# Face Upscaling and Seamless Embedding (FUSE)

FUSE is a collection of advanced face enhancement nodes for ComfyUI that provides seamless face upscaling and embedding capabilities using YOLO face detection and SAM (Segment Anything Model) for precise masking.

## Installation

```bash
cd custom_nodes
git clone https://github.com/WASasquatch/face-upscaling-and-seamless-embedding WAS-FUSE
cd WAS-FUSE
pip install -r requirements.txt
```

## Nodes

### FUSEKSampler

Advanced face-aware sampling node that combines face detection, segmentation, and seamless blending.

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| model | Base model for sampling | MODEL | Required |
| positive | Positive conditioning | CONDITIONING | Required |
| negative | Negative conditioning | CONDITIONING | Required |
| latent | Input latent for sampling | LATENT | Required |
| vae | VAE model | VAE | Required |
| seed | Seed for sampling | INT | Required |
| steps | Number of steps | INT | 20 |
| cfg | CFG Scale | FLOAT | 7.0 |
| sampler_name | Name of the sampler | STRING | "euler" |
| scheduler | Scheduler type | STRING | "normal" |
| denoise | Denoising strength | FLOAT | 1.0 |
| yolo_model_name | YOLO model for face detection | STRING | Required |
| sam_model_name | SAM model name | STRING | Required |
| sam_model_type | SAM model type | STRING | Required |
| blend_amount | Strength of blending | FLOAT | 0.5 |
| blend_mode | Blending method | STRING | "gaussian" |
| use_sam_mask | Use SAM for masking | BOOL | True |

### FUSESamplerMaskOptions

Additional mask processing options for the FUSEKSampler.

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| mask_dilation | Amount to dilate the mask | INT | 0 |
| mask_erosion | Amount to erode the mask | INT | 0 |
| mask_smoothing | Smoothing factor for mask edges | FLOAT | 0.0 |
| mask_fill_holes | Fill holes in the mask | BOOL | True |

## Requirements

- Python >= 3.10
- torch
- numpy
- Pillow
- scipy
- ultralytics (YOLO)
- segment-anything (SAM)

## Models

The following model directories are expected in your ComfyUI models folder:
- `models/yolo-face/` - YOLO face detection models
- `models/sams/` - SAM models
- `models/upscale_models/` - Upscaling models

## License

MIT License - See LICENSE file for details.

## Credits

Created by WASasquatch (https://github.com/WASasquatch)
