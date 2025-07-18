# Face Upscaling and Seamless Embedding (FUSE)

FUSE is an All-in-One Face Fix KSampler for ComfyUI that provides seamless face upscaling and embedding capabilities using YOLO face detection and SAM (Segment Anything Model) for precise masking.

## Installation

Navigate to your ComfyUI's custom nodes
```bash
cd custom_nodes
```

Clone FUSE nodes.
```bash
git clone https://github.com/WASasquatch/face-upscaling-and-seamless-embedding
```

Navigate to the FUSE directory
```bash
cd face-upscaling-and-seamless-embedding
```

Install dependencies to your Python Environment used by ComfyUI
```bash
pip install -r requirements.txt
```

## Nodes

### FUSEKSampler

Advanced face-aware sampling node that combines YOLO face detection, SAM segmentation, and seamless blending. Features include:
- Intelligent face detection and cropping with adjustable padding and size
- Optional square crop enforcement for consistent face processing
- Multiple blending modes (box/radial) with configurable strength
- SAM-based precise face masking
- Color preservation options using various transfer methods (LAB, PDF, mean-std)
- Internal caching system for faster workflow iteration
- Support for processing specific faces using face index

| Parameter | Description | Type | Default |
|-----------|-------------|------|---------|
| model | Base model for sampling | MODEL | Required |
| vae | VAE model for the sampler | VAE | Required |
| images | Input image batch for face detection and upscaling | IMAGE | Required |
| positive | Positive conditioning for the sampler | CONDITIONING | Required |
| negative | Negative conditioning for the sampler | CONDITIONING | Required |
| use_cache | Use internal caching to speed up workflow iteration | BOOLEAN | True |
| seed | Seed for deterministic results | INT | 0 |
| steps | Number of steps for the sampler | INT | 20 |
| cfg | Classifier-Free Guidance scale | FLOAT | 8.0 |
| sampler_name | Name of the sampler | STRING | (from KSampler) |
| scheduler | Scheduler type | STRING | (from KSampler) |
| denoise | Denoising strength (lower values adhere more to input face) | FLOAT | 0.5 |
| yolo_detector | YOLO model for face detection | STRING | Required |
| sam_segmenter | SAM model for face segmentation | STRING | Required |
| sam_model_type | SAM model type (vit_b, vit_l, vit_h) | STRING | "vit_b" |
| face_id | Index of the face to process (0 is first face) | INT | 0 |
| face_size | Resolution to sample the face crop at | INT | 512 |
| face_padding | Padding in pixels around face crop | INT | 20 |
| force_square | Force 1:1 square face crops | BOOLEAN | True |
| blend_amount | Amount of blending for face embedding | FLOAT | 0.3 |
| blend_mode | Feathering mode for blending (box, radial) | STRING | "box" |
| use_sam_mask | Use SAM face mask for blending | BOOLEAN | False |
| face_color_transfer | Color transfer mode (none, lab, pdf, mean_std) | STRING | "none" |

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

MIT License

## Credits

Created by WASasquatch (https://github.com/WASasquatch)
