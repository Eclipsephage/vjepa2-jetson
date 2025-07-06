# V-JEPA 2 Model Files

This directory contains model files for V-JEPA 2 inference.

## Required Files

The following model files are required for the demo scripts to work:

- **`vitl.pt`** - V-JEPA 2 ViT-Large backbone weights (FP32)
- **`vitl_fp16.pt`** - V-JEPA 2 ViT-Large backbone weights (FP16, auto-generated)
- **`ssv2-vitl-16x2x3.pt`** - Something-Something V2 classifier weights

## File Sizes

- `vitl.pt`: ~4.8GB
- `vitl_fp16.pt`: ~4.2GB (auto-converted from vitl.pt)
- `ssv2-vitl-16x2x3.pt`: ~189MB

## How to Obtain

1. **Official Repository**: Visit the [V-JEPA 2 repository](https://github.com/facebookresearch/vjepa2)
2. **Download Instructions**: Follow the model download instructions in the official repo
3. **Place Files**: Download and place the model files in this directory

## Git Ignore

The actual model files are ignored by Git due to their large size. Only the dummy file and README are tracked.

## Auto-Conversion

The script will automatically convert `vitl.pt` to `vitl_fp16.pt` on first run if needed. 