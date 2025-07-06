# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import sys, io, contextlib
from functools import lru_cache
from typing import Optional  # add near top? (not needed for runtime but ensure annotations)
# ---------------------------------------------------------------------------
# Runtime patches / environment fixes (must run BEFORE torch/torchvision)
# ---------------------------------------------------------------------------

# 1) Fix NVML stub warning by prioritising the real library over stubs *before* torch loads.
def _fix_nvml_library():
    cuda_root = "/usr/local/cuda-11.4/targets/aarch64-linux/lib"
    current = os.environ.get("LD_LIBRARY_PATH", "").split(":") if os.environ.get("LD_LIBRARY_PATH") else []
    cleaned = [p for p in current if "stubs" not in p]
    if cuda_root not in cleaned:
        cleaned.insert(0, cuda_root)
    os.environ["LD_LIBRARY_PATH"] = ":".join(cleaned)
    print("🔧 NVML fix applied → stubs removed; real library prioritised")

_fix_nvml_library()

# 3) Tell PyTorch not to touch NVML at all to remove final warning
os.environ["PYTORCH_NVML_DISABLED"] = "1"  # must be set before torch import
print("🔇 PyTorch NVML initialisation disabled")

# 4) Silence verbose C-level warnings (libnvidia-ml stub, etc.) printed directly to stderr
# DELETE_START
class _StderrFilter(io.TextIOBase):
    def __init__(self, original, patterns):
        self._orig = original
        self._patterns = patterns
    def write(self, s):
        if any(p in s for p in self._patterns):
            return len(s)
        return self._orig.write(s)
    def flush(self):
        return self._orig.flush()

_patterns_to_suppress = [
    "WARNING:",
    "libnvidia-ml.so in GDK package is a stub",
]
_sys_stderr_orig = sys.stderr
# sys.stderr = _StderrFilter(sys.stderr, _patterns_to_suppress)  # disabled – allow default warnings
# ---------------------------------------------------------------------------
# After critical imports are done, restore stderr so real errors show up.
# ---------------------------------------------------------------------------
# DELETE_END

# Now safe to import the rest of the std-lib modules
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
# from src.models.vision_transformer import vit_large  # unused after switch to torch.hub
from src.models.attentive_pooler import AttentiveClassifier

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# 2) Force torchvision.resize antialias=False to silence future warnings & stay fast.
def _patch_torchvision_resize_antialias():
    import torchvision.transforms.functional as tvf
    if not hasattr(tvf.resize, "__patched_antialias__"):
        _orig_resize = tvf.resize

        def _resize_no_alias(img, size, *args, **kwargs):
            kwargs.setdefault("antialias", False)
            return _orig_resize(img, size, *args, **kwargs)

        _resize_no_alias.__patched_antialias__ = True  # type: ignore
        tvf.resize = _resize_no_alias  # type: ignore
        print("🖼️  torchvision.resize patched → antialias=False (fast path)")

_patch_torchvision_resize_antialias()

# Helper to grab current process resident memory (host RAM)
def _get_host_mem_gb():
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1e9  # GB
    except Exception:
        # Fallback: parse /proc/self/status
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        kb = int(line.split()[1])
                        return kb / 1e6  # convert kB→GB
        except Exception:
            return float('nan')

def _get_system_mem_stats_gb():
    """Return (used_gb, total_gb) system-wide memory stats."""
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        return vm.used / 1e9, vm.total / 1e9
    except Exception:
        return float('nan'), float('nan')

def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    pretrained_dict = torch.load(pretrained_weights, map_location="cpu")["encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))

def load_pretrained_vjepa_classifier_weights(model, pretrained_weights):
    pretrained_dict = torch.load(pretrained_weights, map_location="cpu")["classifiers"][0]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))

def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return eval_transform

def get_video(num_frames: Optional[int] = None):
    """Read *num_frames* equally-spaced RGB frames from the sample clip.

    Returns (video_np, clip_duration_s) where *video_np* is a T×H×W×3 uint8
    ndarray suitable for later tensor conversion.
    """
    vr = VideoReader("notebooks/sample_video.mp4", ctx=cpu())
    fps = vr.get_avg_fps() or 30  # decord may report 0 for some files
    total_duration = len(vr) / fps if fps else float('nan')

    if num_frames is None:
        num_frames = NUM_FRAMES

    total_frames = len(vr)
    if total_frames <= num_frames:
        frame_idx = np.arange(total_frames)  # use everything if very short
    else:
        frame_idx = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    video = vr.get_batch(frame_idx).asnumpy()  # (T, H, W, 3)
    return video, total_duration

def forward_vjepa_video(model_pt, pt_transform):
    with torch.inference_mode():
        video, video_duration = get_video()
        video = torch.from_numpy(video).permute(0, 3, 1, 2)
        x_pt = pt_transform(video).cuda().unsqueeze(0)
        out_patch_features_pt = model_pt(x_pt)
    return out_patch_features_pt, video_duration

# ---------------------------------------------------------------------------
# Configuration flags
# ---------------------------------------------------------------------------
# Use mixed precision (FP16) for the classifier to speed up matmuls on Tensor Cores.
# Set to False for full-precision.
USE_MIXED_PREC = True

# Optional: use Hugging-Face packaged model that already contains backbone +
# attentive classifier in FP16 with 256 tokens.  This requires a bleeding-edge
# Transformers (≥ 4.40) which is not available on Python 3.8 for Jetson, so we
# disable it by default and rely on the torch.hub path instead.
USE_HF_MODEL   = False
HF_REPO_ID     = "facebook/vjepa2-vitl-fpc16-256-ssv2"

# Number of frames fed to the backbone (and expected by the attentive probe).
# The "vjepa2_vit_large" hub entry is pre-trained with 16-frame positional
# embeddings, so we sample the same amount from the video.
NUM_FRAMES = 16

# Reduce tokens when using local path. Ignored when USE_HF_MODEL=True.
TOKEN_REDUCTION = 256   # set to 0 to disable

# Use torch.compile (torch 2.0+) to optimise the classifier network.
USE_COMPILED_CLASSIFIER = False     # Triton not available on Jetson

def get_vjepa_video_classification_results(classifier, out_patch_features_pt):
    SOMETHING_SOMETHING_V2_CLASSES = json.load(open("notebooks/ssv2_classes.json", "r"))

    with torch.inference_mode():
        # Optionally pool tokens to reduce seq length
        feat = out_patch_features_pt
        if TOKEN_REDUCTION and feat.shape[1] > TOKEN_REDUCTION:
            feat = (
                torch.nn.functional.adaptive_avg_pool1d(
                    feat.transpose(1, 2), TOKEN_REDUCTION
                ).transpose(1, 2)
            )

        if USE_MIXED_PREC and torch.cuda.is_available():
            logits = classifier(feat.half())
        else:
            logits = classifier(feat)

        out_classifier = logits.float()  # ensure fp32 for softmax + printing

    print("\n" + "="*60)
    print("🎯 CLASSIFICATION RESULTS")
    print("="*60)

    print(f"Classifier output shape: {out_classifier.shape}")

    top5_indices = out_classifier.topk(5).indices[0]
    top5_probs   = F.softmax(out_classifier.topk(5).values[0], dim=0) * 100.0
    for rank, (idx, prob) in enumerate(zip(top5_indices, top5_probs), 1):
        class_name = SOMETHING_SOMETHING_V2_CLASSES[str(idx.item())]
        print(f"{rank:2d}. {class_name:<35} {prob:6.1f}%")

# --- Simple module-level cache for backbone & classifier ---
_cached_models = {}

# --- Hugging Face model + processor cache ---
_cached_hf = {}

def _get_cached_models():
    """Load backbone + classifier (with warm-up) on first call, cache thereafter.
    Returns (model, classifier, backbone_load_s, classifier_load_s) where the two
    timing values are >0 only the first time this is executed in the Python
    process; subsequent calls return 0.0 for both timings."""
    global _cached_models
    if _cached_models:
        return (
            _cached_models["model"],
            _cached_models["classifier"],
            0.0,
            0.0,
        )

    # --- paths -------------------------------------------------
    pt_model_path         = "models/vitl_fp16.pt"          # NEW
    pt_model_orig_path    = "models/vitl.pt"
    classifier_model_path = "models/ssv2-vitl-16x2x3.pt"

    # --- one-time FP32→FP16 conversion ------------------------
    if not os.path.exists(pt_model_path):
        print("⚙️  Converting vitl.pt to FP16 …")
        ckpt = torch.load(pt_model_orig_path, map_location="cpu")
        # down-cast all tensors inside the two state-dict blocks
        for block in ("encoder", "predictor"):
            if block in ckpt:
                ckpt[block] = {k: v.half() for k, v in ckpt[block].items()}
        torch.save(ckpt, pt_model_path)
        print("✅ Saved", pt_model_path)

    # --- load architecture (no weights) -----------------------
    model_pt, _ = torch.hub.load(
        "facebookresearch/vjepa2",
        "vjepa2_vit_large",
        pretrained=False,               # keep Hub from downloading vitl.pt
    )

    # --- load FP16 weights ------------------------------------
    load_pretrained_vjepa_pt_weights(model_pt, pt_model_path)
    model_pt = model_pt.cuda().eval()
    # weights are already FP16 so we can drop extra .half() call

    print("🔗 Loading classifier head (cold)…")
    t_cls_start = time.time()
    classifier = AttentiveClassifier(embed_dim=model_pt.embed_dim, num_heads=16, depth=4, num_classes=174).cuda().eval()
    load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)

    # Optional torch.compile for classifier
    global USE_COMPILED_CLASSIFIER
    if USE_COMPILED_CLASSIFIER:
        try:
            classifier = torch.compile(classifier, mode="max-autotune")
            print("⚡ Compiled classifier with torch.compile")
        except RuntimeError as e:
            print(
                f"⚠️  torch.compile failed ({e.__class__.__name__}); falling back to eager"
            )
            USE_COMPILED_CLASSIFIER = False

    if USE_MIXED_PREC and torch.cuda.is_available():
        classifier.half()  # permanently store weights in FP16 for speed
        dummy_dtype = torch.float16
    else:
        dummy_dtype = torch.float32

    # one-time kernel warm-up with proper dtype
    with torch.inference_mode():
        dummy_tokens = TOKEN_REDUCTION if TOKEN_REDUCTION > 0 else 8192
        dummy = torch.zeros(1, dummy_tokens, model_pt.embed_dim, device="cuda", dtype=dummy_dtype)
        _ = classifier(dummy)
    classifier_time = time.time() - t_cls_start

    _cached_models = {"model": model_pt, "classifier": classifier}
    return model_pt, classifier, 0.0, classifier_time

def _get_hf_model():
    """Load HF packaged backbone+classifier (fp16) once and cache."""
    if _cached_hf:
        return _cached_hf["model"], _cached_hf["proc"], 0.0

    t0 = time.time()
    # Dynamically import the appropriate processor class: older versions of
    # transformers (<4.39) do not have `AutoVideoProcessor`. We therefore try
    # that first, then fall back to `AutoImageProcessor` (introduced earlier),
    # and finally to the more generic `AutoProcessor`.
    from transformers import AutoModelForVideoClassification  # always exists
    try:
        from transformers import AutoVideoProcessor as _AutoProcessor  # type: ignore
    except ImportError:
        try:
            from transformers import AutoImageProcessor as _AutoProcessor  # type: ignore
        except ImportError:
            from transformers import AutoProcessor as _AutoProcessor  # type: ignore

    model = (
        AutoModelForVideoClassification
        .from_pretrained(
            HF_REPO_ID,
            torch_dtype=torch.float16,
            trust_remote_code=True,  # allow loading custom V-JEPA model class
        )
        .cuda()
        .eval()
    )
    processor = _AutoProcessor.from_pretrained(HF_REPO_ID)
    dt = time.time() - t0

    _cached_hf.update(model=model, proc=processor)
    return model, processor, dt

# Helper: read clip for HF path
def _read_clip_for_hf(video_path: str, num_frames: int):
    vr = VideoReader(video_path, ctx=cpu())
    total = len(vr)
    step = max(1, total // num_frames)
    idx = np.arange(0, step * num_frames, step)[:num_frames]
    clip = vr.get_batch(idx).asnumpy()  # (T, H, W, 3)
    return clip

def run_sample_inference():
    if USE_HF_MODEL:
        _run_hf()
    else:
        _run_local()

# ---------------------------------------------------------------------
# Existing local inference path renamed
# ---------------------------------------------------------------------

def _run_local():
    start_total = time.time()
    
    # ---- Load (or retrieve cached) backbone & classifier ----
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
    ssv2_classes_path = "notebooks/ssv2_classes.json"
    if not os.path.exists(ssv2_classes_path):
        command = [
            "wget",
            "https://huggingface.co/datasets/huggingface/label-files/resolve/d79675f2d50a7b1ecf98923d42c30526a51818e2/something-something-v2-id2label.json",
            "-O",
            "notebooks/ssv2_classes.json",
        ]
        subprocess.run(command)
        print("Downloading SSV2 classes")

    # Classification inference
    print("🏷️  Running classification…")
    start_classification = time.time()
    get_vjepa_video_classification_results(classifier, out_patch_features_pt)
    classification_time = time.time() - start_classification
    print(f"Classification time: {classification_time:.2f} seconds")

    # Calculate total time and performance metrics
    total_time = time.time() - start_total
    video_duration = video_duration  # seconds
    
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
    
    # ---- GPU memory ----
    if torch.cuda.is_available():
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        used_gb, total_gb, _free_gb = _get_gpu_mem_gb()
        host_gb = _get_host_mem_gb()
        sys_used_gb, sys_total_gb = _get_system_mem_stats_gb()
        print(
            f"\n🗄️  GPU alloc/reserved: {mem_alloc:.2f}/{mem_reserved:.2f} GB | "
            f"GPU used/total: {used_gb:.2f}/{total_gb:.2f} GB | "
            f"🧠 Host RAM (proc): {host_gb:.2f} GB | "
            f"💻 Sys RAM used/total: {sys_used_gb:.2f}/{sys_total_gb:.2f} GB"
        )

    print("="*60)

# Helper to collect detailed GPU stats
def _get_gpu_mem_gb():
    if not torch.cuda.is_available():
        return float('nan'), float('nan'), float('nan')
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return used / 1e9, total / 1e9, free / 1e9

# ---------------------------------------------------------------------
# HF inference helper
# ---------------------------------------------------------------------

def _run_hf():
    start_total = time.time()

    model, processor, load_time = _get_hf_model()
    print(f"🧩 HF model load: {load_time:.2f}s (0 means cached)")

    clip = _read_clip_for_hf("notebooks/sample_video.mp4", model.config.frames_per_clip)
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
    for r,(i,p) in enumerate(zip(top5_idx, top5_prob),1):
        print(f"{r:2d}. {model.config.id2label[i.item()]:<35} {p:5.1f}%")

    total_time = time.time() - start_total
    print("\n"+"="*60)
    print("📊 PERFORMANCE METRICS (HF)")
    print("="*60)
    print(f"Model loading:        {load_time:>8.2f}s")
    print(f"Forward (incl. preprocess): {infer_time:>8.2f}s")
    print(f"Total time:           {total_time:>8.2f}s")

if __name__ == "__main__":
    print("===== FIRST (cold) RUN =====")
    run_sample_inference()

    print("\n===== SECOND (warm) RUN =====")
    run_sample_inference() 