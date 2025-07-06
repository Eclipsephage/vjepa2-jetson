# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_large

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

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

def get_video():
    vr = VideoReader("notebooks/sample_video.mp4", ctx=cpu())
    fps = vr.get_avg_fps()
    total_duration = len(vr) / fps

    frame_idx = np.arange(0, 128, 2)
    video = vr.get_batch(frame_idx).asnumpy()
    return video, total_duration

def forward_vjepa_video(model_pt, pt_transform):
    with torch.inference_mode():
        video, video_duration = get_video()
        video = torch.from_numpy(video).permute(0, 3, 1, 2)
        x_pt = pt_transform(video).cuda().unsqueeze(0)
        out_patch_features_pt = model_pt(x_pt)
    return out_patch_features_pt, video_duration

def get_vjepa_video_classification_results(classifier, out_patch_features_pt):
    SOMETHING_SOMETHING_V2_CLASSES = json.load(open("notebooks/ssv2_classes.json", "r"))

    with torch.inference_mode():
        out_classifier = classifier(out_patch_features_pt)

    print(f"Classifier output shape: {out_classifier.shape}")

    print("Top 5 predicted class names:")
    top5_indices = out_classifier.topk(5).indices[0]
    top5_probs = F.softmax(out_classifier.topk(5).values[0], dim=0) * 100.0
    for idx, prob in zip(top5_indices, top5_probs):
        str_idx = str(idx.item())
        print(f"{SOMETHING_SOMETHING_V2_CLASSES[str_idx]} ({prob}%)")

def run_sample_inference():
    start_total = time.time()
    
    pt_model_path = "models/vitl.pt"
    classifier_model_path = "models/ssv2-vitl-16x2x3.pt"

    sample_video_path = "notebooks/sample_video.mp4"
    if not os.path.exists(sample_video_path):
        video_url = "https://huggingface.co/datasets/nateraw/kinetics-mini/resolve/main/val/bowling/-WH-lxmGJVY_000005_000015.mp4"
        command = ["wget", video_url, "-O", sample_video_path]
        subprocess.run(command)
        print("Downloading video")

    # Initialize the PyTorch model
    print("Loading ViT-Large model...")
    start_model_load = time.time()
    model_pt = vit_large(img_size=(256, 256), num_frames=64)
    model_pt.cuda().eval()
    load_pretrained_vjepa_pt_weights(model_pt, pt_model_path)
    model_load_time = time.time() - start_model_load
    print(f"Model loading time: {model_load_time:.2f} seconds")

    # Build PyTorch preprocessing transform
    pt_video_transform = build_pt_video_transform(img_size=256)

    # Inference on video
    print("Running video inference...")
    start_inference = time.time()
    out_patch_features_pt, video_duration = forward_vjepa_video(model_pt, pt_video_transform)
    inference_time = time.time() - start_inference
    print(f"Video inference time: {inference_time:.2f} seconds")

    print(f"PyTorch output shape: {out_patch_features_pt.shape}")

    # Initialize the classifier
    print("Loading classifier...")
    start_classifier_load = time.time()
    classifier = AttentiveClassifier(embed_dim=model_pt.embed_dim, num_heads=16, depth=4, num_classes=174).cuda().eval()
    load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)
    classifier_load_time = time.time() - start_classifier_load
    print(f"Classifier loading time: {classifier_load_time:.2f} seconds")

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
    print("Running classification...")
    start_classification = time.time()
    get_vjepa_video_classification_results(classifier, out_patch_features_pt)
    classification_time = time.time() - start_classification
    print(f"Classification time: {classification_time:.2f} seconds")

    # Calculate total time and performance metrics
    total_time = time.time() - start_total
    video_duration = video_duration  # seconds
    
    print(f"\n{'='*50}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*50}")
    print(f"Model loading:        {model_load_time:>8.2f}s")
    print(f"Video inference:      {inference_time:>8.2f}s")
    print(f"Classifier loading:   {classifier_load_time:>8.2f}s")
    print(f"Classification:       {classification_time:>8.2f}s")
    print(f"{'─'*50}")
    print(f"Total time:           {total_time:>8.2f}s")
    print(f"Video duration:       {video_duration:>8.2f}s")
    print(f"Real-time factor:     {inference_time/video_duration:>8.2f}x")
    print(f"FPS (inference only): {1.0/inference_time:>8.2f}")
    print(f"FPS (total):          {1.0/total_time:>8.2f}")
    
    if inference_time < video_duration:
        print(f"✅ Real-time capable: Inference faster than video duration")
    else:
        print(f"⚠️  Not real-time: Inference slower than video duration")
    
    print(f"{'='*50}")

if __name__ == "__main__":
    run_sample_inference()