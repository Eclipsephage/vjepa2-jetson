# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
V-JEPA 2 HuggingFace Model Demo for Jetson Devices

This script provides an optimized version of V-JEPA 2 inference specifically
designed for NVIDIA Jetson devices using the downloaded HuggingFace model
'vjepa2-vitl-fpc16-256-ssv2.pt' with enhanced performance monitoring
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
else:
    try:
        import torch
        import torch.nn.functional as F
        from decord import VideoReader, cpu
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
USE_MIXED_PREC = True  # Disable mixed precision
NUM_FRAMES = 16
TOKEN_REDUCTION = 256

# Magic numbers as constants
DEFAULT_FPS = 30
CUDA_ROOT_PATH = "/usr/local/cuda-11.4/targets/aarch64-linux/lib"
VIDEO_PATH = os.path.join(SCRIPT_DIR, "sample_video.mp4")
SSV2_CLASSES_PATH = os.path.join(SCRIPT_DIR, "ssv2_classes.json")
SSV2_DOWNLOAD_URL = (
    "https://huggingface.co/datasets/huggingface/label-files/resolve/"
    "d79675f2d50a7b1ecf98923d42c30526a51818e2/something-something-v2-id2label.json"
)

# Model paths
DOWNLOADED_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "vjepa2-vitl-fpc16-256-ssv2.pt")

# Global caches
_cached_model: Optional[Any] = None

# Add global cache for both models
_cached_backbone: Optional[Any] = None
_cached_classifier: Optional[Any] = None

def _fix_nvml_library() -> None:
    """Fix NVML stub warning by prioritizing the real library over stubs."""
    current = os.environ.get("LD_LIBRARY_PATH", "").split(":") if os.environ.get("LD_LIBRARY_PATH") else []
    cleaned = [p for p in current if "stubs" not in p]
    if CUDA_ROOT_PATH not in cleaned:
        cleaned.insert(0, CUDA_ROOT_PATH)
    os.environ["LD_LIBRARY_PATH"] = ":".join(cleaned)
    print("�� NVML fix applied → stubs removed; real library prioritized")


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


def _get_gpu_mem_gb() -> Tuple[float, float, float]:
    """Get GPU memory stats in GB."""
    if not torch.cuda.is_available():
        return float('nan'), float('nan'), float('nan')
    free, total = torch.cuda.mem_get_info()
    used = total - free
    return used / 1e9, total / 1e9, free / 1e9


# Add the weight loading function from the original script
def load_pretrained_vjepa_pt_weights(model: "torch.nn.Module", pretrained_weights: str) -> None:
    """Load pretrained V-JEPA 2 weights for PyTorch model with flexible structure handling."""
    pretrained_dict = torch.load(pretrained_weights, map_location="cpu")
    
    # Handle different model structures
    if isinstance(pretrained_dict, dict):
        # Try different possible key structures
        if "encoder" in pretrained_dict:
            # Original PyTorch Hub format
            weights = pretrained_dict["encoder"]
            print("✅ Found 'encoder' key in model weights")
        elif "model" in pretrained_dict:
            # HuggingFace format
            weights = pretrained_dict["model"]
            print("✅ Found 'model' key in model weights")
        elif "state_dict" in pretrained_dict:
            # Some other format
            weights = pretrained_dict["state_dict"]
            print("✅ Found 'state_dict' key in model weights")
        else:
            # Assume the dict itself contains the weights
            weights = pretrained_dict
            print("✅ Using top-level dict as weights")
    else:
        # Assume it's already the weights
        weights = pretrained_dict
        print("✅ Using loaded object as weights")
    
    # Clean up key names
    weights = {
        k.replace("module.", "").replace("backbone.", ""): v 
        for k, v in weights.items()
    }
    
    msg = model.load_state_dict(weights, strict=False)
    print(f"Pretrained weights found at {pretrained_weights} and loaded with msg: {msg}")

# Add classifier loading function
def load_pretrained_vjepa_classifier_weights(model: "torch.nn.Module", pretrained_weights: str) -> None:
    """Load pretrained classifier weights."""
    pretrained_dict = torch.load(pretrained_weights, map_location="cpu")["classifiers"][0]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print(f"Classifier weights found at {pretrained_weights} and loaded with msg: {msg}")

# Add classifier model loading
def _load_classifier(backbone_model) -> Tuple[Any, float]:
    """Load the SSv2 classifier."""
    classifier_model_path = os.path.join(PROJECT_ROOT, "models", "ssv2-vitl-16x2x3.pt")
    
    if not os.path.exists(classifier_model_path):
        print(f"❌ Classifier model not found at: {classifier_model_path}")
        raise FileNotFoundError(f"Classifier file not found: {classifier_model_path}")
    
    print(" Loading classifier head...")
    t0 = time.time()
    
    # Import the classifier class
    from src.models.attentive_pooler import AttentiveClassifier
    
    classifier = AttentiveClassifier(
        embed_dim=backbone_model.embed_dim, 
        num_heads=16, 
        depth=4, 
        num_classes=174
    ).cuda().eval()
    
    load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)
    
    if USE_MIXED_PREC and torch.cuda.is_available():
        classifier.half()
    
    load_time = time.time() - t0
    print(f"✅ Classifier loaded in {load_time:.2f}s")
    
    return classifier, load_time

def _load_hf_model() -> Tuple[Any, float]:
    """Load the downloaded HuggingFace model weights and apply to PyTorch Hub architecture."""
    global _cached_model
    if _cached_model is not None:
        return _cached_model, 0.0

    # Check if downloaded model exists
    if not os.path.exists(DOWNLOADED_MODEL_PATH):
        print(f"❌ Downloaded model not found at: {DOWNLOADED_MODEL_PATH}")
        print("Please ensure the vjepa2-vitl-fpc16-256-ssv2.pt file is in the models/ directory")
        raise FileNotFoundError(f"Model file not found: {DOWNLOADED_MODEL_PATH}")
    
    print(f" Loading downloaded HuggingFace model weights: {DOWNLOADED_MODEL_PATH}")
    t0 = time.time()
    
    try:
        # Load the model architecture from PyTorch Hub (no weights)
        print("📋 Loading model architecture from PyTorch Hub...")
        model, _ = torch.hub.load(
            "facebookresearch/vjepa2",
            "vjepa2_vit_large",
            pretrained=False,
        )
        
        # Load and apply the downloaded weights
        print("⚙️  Applying downloaded weights...")
        load_pretrained_vjepa_pt_weights(model, DOWNLOADED_MODEL_PATH)
        
        # Move to GPU and set to eval mode
        model = model.cuda().eval()
        
        # Convert to FP16 mode (weights are already FP16, but model needs to be in FP16 mode for FP16 output)
        if USE_MIXED_PREC and torch.cuda.is_available():
            model = model.half()
            print("⚙️  Model set to FP16 mode (weights already FP16)")
        
        load_time = time.time() - t0
        print(f"✅ Model loaded in {load_time:.2f}s")
        
        _cached_model = model
        return model, load_time
        
    except Exception as e:
        print(f"❌ Failed to load model: {str(e)}")
        raise


# Modify the run_inference function to use both backbone and classifier
def _get_cached_models() -> Tuple[Any, Any, float, float]:
    """
    Load backbone + classifier (with warm-up) on first call, cache thereafter.
    
    Returns:
        Tuple of (backbone_model, classifier, backbone_load_s, classifier_load_s) where the two
        timing values are >0 only the first time this is executed in the Python
        process; subsequent calls return 0.0 for both timings.
    """
    global _cached_backbone, _cached_classifier
    
    # If both models are cached, return them immediately
    if _cached_backbone is not None and _cached_classifier is not None:
        return _cached_backbone, _cached_classifier, 0.0, 0.0

    # Load backbone model
    backbone_model, backbone_load_time = _load_hf_model()
    
    # Load classifier
    classifier, classifier_load_time = _load_classifier(backbone_model)
    
    # Cache both models
    _cached_backbone = backbone_model
    _cached_classifier = classifier
    
    return backbone_model, classifier, backbone_load_time, classifier_load_time

# Remove the separate _load_hf_model and _load_classifier functions from run_inference
# and replace with the cached version
def run_inference() -> None:
    """Run inference with the downloaded HuggingFace model."""
    start_total = time.time()
    
    # Load (or retrieve cached) backbone & classifier
    backbone_model, classifier, backbone_load_time, classifier_load_time = _get_cached_models()
    print(f"🧩 Backbone load: {backbone_load_time:.2f}s | Classifier load: {classifier_load_time:.2f}s (0 means cached)")
    
    # Get video
    print("️  Loading video...")
    video, video_duration = get_video()
    print(f"📹 Video loaded: {video.shape} ({video_duration:.2f}s)")
    
    # Prepare video for inference
    print("🔧 Preparing video for inference...")
    # Convert to tensor format expected by the model
    video_tensor = torch.from_numpy(video).permute(3, 0, 1, 2).unsqueeze(0)  # B, C, T, H, W
    video_tensor = video_tensor.cuda()
    
    if USE_MIXED_PREC and torch.cuda.is_available():
        video_tensor = video_tensor.half()
    
    print(f" Video tensor shape: {video_tensor.shape}")
    
    # Run backbone inference
    print("🚀 Running backbone inference...")
    start_inference = time.time()
    
    with torch.inference_mode():
        # Forward pass through backbone
        backbone_features = backbone_model(video_tensor)
    
    inference_time = time.time() - start_inference
    
    print(f"🔍 Backbone output shape: {backbone_features.shape}")
    print(f"⚡ Backbone inference completed in {inference_time:.2f}s")
    
    # Run classifier inference
    print("🏷️  Running classifier inference...")
    start_classification = time.time()
    
    with torch.inference_mode():
        # Apply token reduction if needed
        feat = backbone_features
        if TOKEN_REDUCTION and feat.shape[1] > TOKEN_REDUCTION:
            feat = torch.nn.functional.adaptive_avg_pool1d(
                feat.transpose(1, 2), TOKEN_REDUCTION
            ).transpose(1, 2)
        
        # Convert to FP16 if mixed precision is enabled
        if USE_MIXED_PREC and torch.cuda.is_available():
            feat = feat.half()
        
        # Forward pass through classifier
        logits = classifier(feat)
    
    classification_time = time.time() - start_classification
    
    print(f"🎯 Classifier output shape: {logits.shape}")
    print(f"⚡ Classification completed in {classification_time:.2f}s")
    
    # Download SSV2 classes if not already present
    if not os.path.exists(SSV2_CLASSES_PATH):
        print("📥 Downloading SSV2 class labels...")
        command = [
            "wget",
            SSV2_DOWNLOAD_URL,
            "-O",
            SSV2_CLASSES_PATH,
        ]
        subprocess.run(command, check=True)
        print("✅ SSV2 classes downloaded")
    
    # Load class labels
    with open(SSV2_CLASSES_PATH, "r") as f:
        SOMETHING_SOMETHING_V2_CLASSES = json.load(f)
    
    # Process classification results
    print("🏷️  Processing classification results...")
    
    # Get top predictions
    logits = logits.float()  # Ensure float32 for softmax
    top5_indices = logits.topk(5).indices[0]
    top5_probs = F.softmax(logits.topk(5).values[0], dim=0) * 100.0
    
    # Print results
    print("\n" + "="*60)
    print("🎯 CLASSIFICATION RESULTS")
    print("="*60)
    print(f"Logits shape: {logits.shape}")
    
    for rank, (idx, prob) in enumerate(zip(top5_indices, top5_probs), 1):
        class_name = SOMETHING_SOMETHING_V2_CLASSES.get(str(idx.item()), f"Class_{idx.item()}")
        print(f"{rank:2d}. {class_name:<35} {prob:6.1f}%")
    
    # Calculate total time and performance metrics
    total_time = time.time() - start_total
    
    print("\n" + "="*60)
    print("📊 PERFORMANCE METRICS")
    print("="*60)
    print(f"Backbone loading:     {backbone_load_time:>8.2f}s")
    print(f"Classifier loading:   {classifier_load_time:>8.2f}s")
    print(f"Backbone inference:   {inference_time:>8.2f}s")
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
            f" Host RAM (proc): {host_gb:.2f} GB | "
            f" Sys RAM used/total: {sys_used_gb:.2f}/{sys_total_gb:.2f} GB"
        )

    print("="*60)


# Initialize environment fixes
_fix_nvml_library()
os.environ["PYTORCH_NVML_DISABLED"] = "1"
print("🔇 PyTorch NVML initialization disabled")
_patch_torchvision_resize_antialias()


if __name__ == "__main__":
    print("🚀 V-JEPA 2 HuggingFace Model Demo")
    print("�� Model: vjepa2-vitl-fpc16-256-ssv2.pt (downloaded)")
    print(f"⚙️  Mixed Precision: {USE_MIXED_PREC}")
    print(f"��️  Frames per clip: {NUM_FRAMES}")
    print("="*60)
    
    print("===== FIRST (cold) RUN =====")
    run_inference()

    print("\n===== SECOND (warm) RUN =====")
    run_inference()