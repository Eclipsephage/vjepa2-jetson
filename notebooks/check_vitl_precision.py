#!/usr/bin/env python3
"""
Lightweight script to check the precision of vitl.pt without loading the entire model.
This script samples a few parameters to determine if the model is FP16 or FP32.
"""

import os
import sys
import torch

# Add project root to Python path for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def check_model_precision(model_path: str, sample_size: int = 10) -> None:
    """
    Check the precision of a model by sampling parameters.
    
    Args:
        model_path: Path to the model file
        sample_size: Number of parameters to sample (default: 10)
    """
    print(f"�� Checking precision of: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        return
    
    try:
        # Load the model file
        print("�� Loading model file...")
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Find the encoder weights
        if "encoder" in checkpoint:
            encoder_weights = checkpoint["encoder"]
            print(f"✅ Found encoder with {len(encoder_weights)} parameters")
        else:
            print("❌ No encoder found in checkpoint")
            return
        
        # Sample parameters to check precision
        print(f"🔬 Sampling {sample_size} parameters...")
        fp16_count = 0
        fp32_count = 0
        other_count = 0
        
        sample_params = list(encoder_weights.items())[:sample_size]
        
        for name, param in sample_params:
            if param.dtype == torch.float16:
                fp16_count += 1
            elif param.dtype == torch.float32:
                fp32_count += 1
            else:
                other_count += 1
                print(f"   ⚠️  Unexpected dtype: {name} -> {param.dtype}")
        
        # Report results
        print("\n�� PRECISION ANALYSIS")
        print("=" * 50)
        print(f"Sampled parameters: {len(sample_params)}")
        print(f"FP16 parameters: {fp16_count}")
        print(f"FP32 parameters: {fp32_count}")
        print(f"Other dtypes: {other_count}")
        
        if fp16_count > fp32_count:
            print(f"\n✅ Model appears to be FP16 (majority: {fp16_count}/{len(sample_params)} parameters)")
        elif fp32_count > fp16_count:
            print(f"\n✅ Model appears to be FP32 (majority: {fp32_count}/{len(sample_params)} parameters)")
        else:
            print(f"\n❓ Inconclusive: equal FP16/FP32 counts")
        
        # Show sample parameter details
        print(f"\n🔍 Sample parameter details:")
        for i, (name, param) in enumerate(sample_params[:5]):
            print(f"   {i+1}. {name}: {param.dtype} {param.shape}")
        
        # File size info
        file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"\n📁 File size: {file_size_mb:.1f} MB")
        
    except Exception as e:
        print(f"❌ Error checking model: {e}")

def main():
    """Main function to check vitl.pt precision."""
    print("🚀 V-JEPA 2 Model Precision Checker")
    print("=" * 50)
    
    # Check the original vitl.pt file
    vitl_path = os.path.join(PROJECT_ROOT, "models", "vitl.pt")
    check_model_precision(vitl_path)
    
    print("\n" + "=" * 50)
    print("✅ Precision check complete!")

if __name__ == "__main__":
    main()
