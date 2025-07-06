# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
V-JEPA 2 Optimized Demo for Jetson Devices

This script provides an optimized version of V-JEPA 2 inference specifically
designed for NVIDIA Jetson devices, with enhanced performance monitoring
and memory management.
"""

# Standard library imports
import json
import os
import subprocess
import sys
import time
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING

# Add project root to Python path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Third-party imports
import numpy as np

# Conditional imports to avoid linter errors
if TYPE_CHECKING:
    import torch
    import torch.nn.functional as F
    from decord import VideoReader, cpu
    import src.datasets.utils.video.transforms as video_transforms
    import src.datasets.utils.video.volume_transforms as volume_transforms
    from src.models.attentive_pooler import AttentiveClassifier
else:
    try:
        import torch
        import torch.nn.functional as F
        from decord import VideoReader, cpu
        import src.datasets.utils.video.transforms as video_transforms
        import src.datasets.utils.video.volume_transforms as volume_transforms
        from src.models.attentive_pooler import AttentiveClassifier
    except ImportError as e:
        print(f"❌ Required import failed: {e}")
        print("Please install required packages: torch, decord")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Python path: {sys.path[:3]}...")  # Show first 3 entries
        sys.exit(1)

# Constants
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# Configuration flags
USE_MIXED_PREC = True
USE_HF_MODEL = False
HF_REPO_ID = "facebook/vjepa2-vitl-fpc16-256-ssv2"
NUM_FRAMES = 16
TOKEN_REDUCTION = 256
USE_COMPILED_CLASSIFIER = False

# Magic numbers as constants
DEFAULT_FPS = 30
DEFAULT_DUMMY_TOKENS = 8192
CUDA_ROOT_PATH = "/usr/local/cuda-11.4/targets/aarch64-linux/lib"
VIDEO_PATH = os.path.join(SCRIPT_DIR, "sample_video.mp4")
SSV2_CLASSES_PATH = os.path.join(SCRIPT_DIR, "ssv2_classes.json")
SSV2_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/huggingface/label-files/resolve/"
    "d79675f2d50a7b1ecf98923d42c30526a51818e2/something-something-v2-id2label.json"
)

# Global caches
_cached_models: Dict[str, Any] = {}
_cached_hf: Dict[str, Any] = {}


def _fix_nvml_library() -> None:
    """Fix NVML stub warning by prioritizing the real library over stubs."""
    current = os.environ.get("LD_LIBRARY_PATH", "").split(":") if os.environ.get("LD_LIBRARY_PATH") else []
    cleaned = [p for p in current if "stubs" not in p]
    if CUDA_ROOT_PATH not in cleaned:
        cleaned.insert(0, CUDA_ROOT_PATH)
    os.environ["LD_LIBRARY_PATH"] = ":".join(cleaned)
    print("🔧 NVML fix applied → stubs removed; real library prioritized")


def _patch_torchvision_resize_antialias() -> None:
    """Force torchvision.resize antialias=False to silence warnings and stay fast."""
    try:
        import torchvision.transforms.functional as tvf
        if not hasattr(tvf.resize, "__patched_antialias__"):
            _orig_resize = tvf.resize

            def _resize_no_alias(
                img: "torch.Tensor", 
                size: Tuple[int, int], 
                *args: Any, 
                **kwargs: Any
            ) -> "torch.Tensor":
                kwargs.setdefault("antialias", False)
                return _orig_resize(img, size, *args, **kwargs)

            _resize_no_alias.__patched_antialias__ = True
            tvf.resize = _resize_no_alias
            print("🖼️  torchvision.resize patched → antialias=False (fast path)")
    except ImportError:
        print("⚠️  torchvision not available, skipping resize patch")


def _get_host_mem_gb() -> float:
    """Get current process resident memory in GB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1e9
    except ImportError:
        # Fallback: parse /proc/self/status
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        return kb / 1e6
        except (OSError, ValueError):
            return float('nan')


def _get_system_mem_stats_gb() -> Tuple[float, float]:
    """Return (used_gb, total_gb) system-wide memory stats."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.used / 1e9, vm.total / 1e9
    except ImportError:
        return float('nan'), float('nan')


def load_pretrained_vjepa_pt_weights(model: "torch.nn.Module", pretrained_weights: str) -> None:
    """Load pretrained V-JEPA 2 weights for PyTorch model."""
    pretrained_dict = torch.load(pretrained_weights, map_location="cpu")["encoder"]
    pretrained_dict = {
        k.replace("module.", "").replace("backbone.", ""): v 
        for k, v in pretrained_dict.items()
    }
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print(f"Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}")


def load_pretrained_vjepa_classifier_weights(model: "torch.nn.Module", pretrained_weights: str) -> None:
    """Load pretrained classifier weights."""
    pretrained_dict = torch.load(pretrained_weights, map_location="cpu")["classifiers"][0]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print(f"Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}")


def build_pt_video_transform(img_size: int) -> "video_transforms.Compose":
    """Build PyTorch video preprocessing transform."""
    short_side_size = int(256.0 / 224 * img_size)
    return video_transforms.Compose([
        video_transforms.Resize(short_side_size, interpolation="bilinear"),
        video_transforms.CenterCrop(size=(img_size, img_size)),
        volume_transforms.ClipToTensor(),
        video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])


def get_video(num_frames: Optional[int] = None) -> Tuple[np.ndarray, float]:
    """
    Read num_frames equally-spaced RGB frames from the sample clip.

    Returns:
        Tuple of (video_np, clip_duration_s) where video_np is a T×H×W×3 uint8
        ndarray suitable for later tensor conversion.
    """
    vr = VideoReader(VIDEO_PATH, ctx=cpu())
    fps = vr.get_avg_fps() or DEFAULT_FPS
    total_duration = len(vr) / fps if fps else float('nan')

    if num_frames is None:
        num_frames = NUM_FRAMES

    total_frames = len(vr)
    if total_frames <= num_frames:
        frame_idx = np.arange(total_frames)
    else:
        frame_idx = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    video = vr.get_batch(frame_idx).asnumpy()
    return video, total_duration


def forward_vjepa_video(
    model_pt: "torch.nn.Module", 
    pt_transform: "video_transforms.Compose"
) -> Tuple["torch.Tensor", float]:
    """Run V-JEPA 2 forward pass on video."""
    with torch.inference_mode():
        video, video_duration = get_video()
        video = torch.from_numpy(video).permute(0, 3, 1, 2)
        x_pt = pt_transform(video).cuda().unsqueeze(0)
        out_patch_features_pt = model_pt(x_pt)
    return out_patch_features_pt, video_duration


def get_vjepa_video_classification_results(
    classifier: "torch.nn.Module", 
    out_patch_features_pt: "torch.Tensor"
) -> None:
    """Get V-JEPA 2 video classification results."""
    with open(SSV2_CLASSES_PATH, "r") as f:
        SOMETHING_SOMETHING_V2_CLASSES = json.load(f)

    with torch.inference_mode():
        feat = out_patch_features_pt
        if TOKEN_REDUCTION and feat.shape[1] > TOKEN_REDUCTION:
            feat = torch.nn.functional.adaptive_avg_pool1d(
                feat.transpose(1, 2), TOKEN_REDUCTION
            ).transpose(1, 2)

        if USE_MIXED_PREC and torch.cuda.is_available():
            logits = classifier(feat.half())
        else:
            logits = classifier(feat)

        out_classifier = logits.float()

    print("\n" + "="*60)
    print("🎯 CLASSIFICATION RESULTS")
    print("="*60)

    print(f"Classifier output shape: {out_classifier.shape}")

    top5_indices = out_classifier.topk(5).indices[0]
    top5_probs = F.softmax(out_classifier.topk(5).values[0], dim=0) * 100.0
    for rank, (idx, prob) in enumerate(zip(top5_indices, top5_probs), 1):
        class_name = SOMETHING_SOMETHING_V2_CLASSES[str(idx.item())]
        print(f"{rank:2d}. {class_name:<35} {prob:6.1f}%")


def _get_cached_models() -> Tuple["torch.nn.Module", "torch.nn.Module", float, float]:
    """
    Load backbone + classifier (with warm-up) on first call, cache thereafter.
    
    Returns:
        Tuple of (model, classifier, backbone_load_s, classifier_load_s) where the two
        timing values are >0 only the first time this is executed in the Python
        process; subsequent calls return 0.0 for both timings.
    """
    global _cached_models
    if _cached_models:
        return (
            _cached_models["model"],
            _cached_models["classifier"],
            0.0,
            0.0,
        )

    # Paths - use absolute paths from project root
    pt_model_path = os.path.join(PROJECT_ROOT, "models", "vitl_fp16.pt")
    pt_model_orig_path = os.path.join(PROJECT_ROOT, "models", "vitl.pt")
    classifier_model_path = os.path.join(PROJECT_ROOT, "models", "ssv2-vitl-16x2x3.pt")

    # One-time FP32→FP16 conversion
    if not os.path.exists(pt_model_path):
        print("⚙️  Converting vitl.pt to FP16 …")
        ckpt = torch.load(pt_model_orig_path, map_location="cpu")
        for block in ("encoder", "predictor"):
            if block in ckpt:
                ckpt[block] = {k: v.half() for k, v in ckpt[block].items()}
        torch.save(ckpt, pt_model_path)
        print("✅ Saved", pt_model_path)

    # Load architecture (no weights)
    model_pt, _ = torch.hub.load(
        "facebookresearch/vjepa2",
        "vjepa2_vit_large",
        pretrained=False,
    )

    # Load FP16 weights
    load_pretrained_vjepa_pt_weights(model_pt, pt_model_path)
    model_pt = model_pt.cuda().eval()

    print("🔗 Loading classifier head (cold)…")
    t_cls_start = time.time()
    classifier = AttentiveClassifier(
        embed_dim=model_pt.embed_dim, 
        num_heads=16, 
        depth=4, 
        num_classes=174
    ).cuda().eval()
    load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)

    # Optional torch.compile for classifier
    if USE_COMPILED_CLASSIFIER:
        try:
            classifier = torch.compile(classifier, mode="max-autotune")
            print("⚡ Compiled classifier with torch.compile")
        except RuntimeError as e:
            print(f"⚠️  torch.compile failed ({e.__class__.__name__}); falling back to eager")

    if USE_MIXED_PREC and torch.cuda.is_available():
        classifier.half()
        dummy_dtype = torch.float16
    else:
        dummy_dtype = torch.float32

    # One-time kernel warm-up with proper dtype
    with torch.inference_mode():
        dummy_tokens = TOKEN_REDUCTION if TOKEN_REDUCTION > 0 else DEFAULT_DUMMY_TOKENS
        dummy = torch.zeros(
            1, dummy_tokens, model_pt.embed_dim, 
            device="cuda", dtype=dummy_dtype
        )
        _ = classifier(dummy)
    classifier_time = time.time() - t_cls_start

    _cached_models = {"model": model_pt, "classifier": classifier}
    return model_pt, classifier, 0.0, classifier_time


def _get_hf_model() -> Tuple[Optional[Any], Optional[Any], float]:
    """Load HF packaged backbone+classifier (fp16) once and cache."""
    global _cached_hf
    if _cached_hf:
        return _cached_hf["model"], _cached_hf["proc"], 0.0

    t0 = time.time()
    
    # Dynamically import the appropriate processor class
    processor_class = None
    try:
        from transformers import AutoModelForVideoClassification, AutoVideoProcessor
        processor_class = AutoVideoProcessor
    except ImportError:
        try:
            from transformers import AutoModelForVideoClassification, AutoImageProcessor
            processor_class = AutoImageProcessor
        except ImportError:
            try:
                from transformers import AutoModelForVideoClassification, AutoProcessor
                processor_class = AutoProcessor
            except ImportError:
                print("❌ No suitable processor class found")
                return None, None, 0.0

    if processor_class is None:
        print("❌ Failed to import transformers classes")
        return None, None, 0.0

    try:
        model = AutoModelForVideoClassification.from_pretrained(
            HF_REPO_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,
        ).cuda().eval()
        
        processor = processor_class.from_pretrained(HF_REPO_ID)
        dt = time.time() - t0

        _cached_hf.update(model=model, proc=processor)
        return model, processor, dt
        
    except Exception as e:
        print(f"❌ Failed to load HF model: {str(e)}")
        return None, None, 0.0


def _read_clip_for_hf(video_path: str, num_frames: int) -> np.ndarray:
    """Read clip for HF path."""
    vr = VideoReader(video_path, ctx=cpu())
    total = len(vr)
    step = max(1, total // num_frames)
    idx = np.arange(0, step * num_frames, step)[:num_frames]
    clip = vr.get_batch(idx).asnumpy()
    return clip


def _get_gpu_mem_gb() -> Tuple[float, float, float]:
    """Get GPU memory stats in GB."""
    if not torch.cuda.is_available():
        return float('nan'), float('nan'), float('nan')
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return used / 1e9, total / 1e9, free / 1e9


def _run_local() -> None:
    """Run local inference path."""
    start_total = time.time()
    
    # Load (or retrieve cached) backbone & classifier
    model_pt, classifier, backbone_load_time, classifier_load_time = _get_cached_models()
    print(f"🧩 Backbone load: {backbone_load_time:.2f}s | Classifier load: {classifier_load_time:.2f}s (0 means cached)")

    # Build PyTorch preprocessing transform
    pt_video_transform = build_pt_video_transform(img_size=256)

    # Inference on video
    print("🎞️  Running video inference…")
    start_inference = time.time()
    out_patch_features_pt, video_duration = forward_vjepa_video(model_pt, pt_video_transform)
    inference_time = time.time() - start_inference
    print(f"Video inference time: {inference_time:.2f} seconds")

    print(f"PyTorch output shape: {out_patch_features_pt.shape}")

    # Download SSV2 classes if not already present
    if not os.path.exists(SSV2_CLASSES_PATH):
        command = [
            "wget",
            SSV2_DOWNLOAD_URL,
            "-O",
            SSV2_CLASSES_PATH,
        ]
        subprocess.run(command, check=True)
        print("Downloading SSV2 classes")

    # Classification inference
    print("🏷️  Running classification…")
    start_classification = time.time()
    get_vjepa_video_classification_results(classifier, out_patch_features_pt)
    classification_time = time.time() - start_classification
    print(f"Classification time: {classification_time:.2f} seconds")

    # Calculate total time and performance metrics
    total_time = time.time() - start_total
    
    print("\n" + "="*60)
    print("📊 PERFORMANCE METRICS")
    print("="*60)

    print(f"Model loading:        {backbone_load_time:>8.2f}s")
    print(f"Video inference:      {inference_time:>8.2f}s")
    print(f"Classifier loading:   {classifier_load_time:>8.2f}s")
    print(f"Classification:       {classification_time:>8.2f}s")
    print(f"{'─'*60}")
    print(f"Total time:           {total_time:>8.2f}s")
    print(f"Video duration:       {video_duration:>8.2f}s")
    print(f"Real-time factor:     {inference_time/video_duration:>8.2f}x")
    print(f"FPS (inference only): {1.0/inference_time:>8.2f}")
    print(f"FPS (total):          {1.0/total_time:>8.2f}")
    
    if inference_time < video_duration:
        print("✅ Real-time capable (inference < clip length)")
    else:
        print("⚠️  Not real-time (inference slower than clip)")
    
    # GPU memory
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        used_gb, total_gb, free_gb = _get_gpu_mem_gb()
        host_gb = _get_host_mem_gb()
        sys_used_gb, sys_total_gb = _get_system_mem_stats_gb()
        print(
            f"\n🗄️  GPU alloc/reserved: {mem_alloc:.2f}/{mem_reserved:.2f} GB | "
            f"GPU used/total: {used_gb:.2f}/{total_gb:.2f} GB | "
            f"🧠 Host RAM (proc): {host_gb:.2f} GB | "
            f"💻 Sys RAM used/total: {sys_used_gb:.2f}/{sys_total_gb:.2f} GB"
        )

    print("="*60)


def _run_hf() -> None:
    """Run HuggingFace inference path."""
    start_total = time.time()

    model, processor, load_time = _get_hf_model()
    if model is None or processor is None:
        print("❌ HF model loading failed, falling back to PyTorch Hub...")
        _run_local()
        return
    
    print(f"🧩 HF model load: {load_time:.2f}s (0 means cached)")

    clip = _read_clip_for_hf(
        VIDEO_PATH, 
        getattr(model.config, 'frames_per_clip', NUM_FRAMES)
    )
    inputs = processor(list(clip), return_tensors="pt").to(model.device)

    t_inf = time.time()
    with torch.no_grad():
        logits = model(**inputs).logits
    infer_time = time.time() - t_inf

    # Print top-5
    top5_idx = logits.topk(5).indices[0]
    top5_prob = torch.softmax(logits, dim=-1)[0, top5_idx] * 100

    print("\n"+"="*60)
    print("🎯 CLASSIFICATION RESULTS (HF model)")
    print("="*60)
    for r, (i, p) in enumerate(zip(top5_idx, top5_prob), 1):
        class_name = getattr(model.config, 'id2label', {}).get(i.item(), f"Class_{i.item()}")
        print(f"{r:2d}. {class_name:<35} {p:5.1f}%")

    total_time = time.time() - start_total
    print("\n"+"="*60)
    print("📊 PERFORMANCE METRICS (HF)")
    print("="*60)
    print(f"Model loading:        {load_time:>8.2f}s")
    print(f"Forward (incl. preprocess): {infer_time:>8.2f}s")
    print(f"Total time:           {total_time:>8.2f}s")


def run_sample_inference() -> None:
    """Run sample inference with appropriate backend."""
    if USE_HF_MODEL:
        _run_hf()
    else:
        _run_local()


# Initialize environment fixes
_fix_nvml_library()
os.environ["PYTORCH_NVML_DISABLED"] = "1"
print("🔇 PyTorch NVML initialization disabled")
_patch_torchvision_resize_antialias()


if __name__ == "__main__":
    print("===== FIRST (cold) RUN =====")
    run_sample_inference()

    print("\n===== SECOND (warm) RUN =====")
    run_sample_inference() 