# Face Upscaling and Seamless Embedding (FUSE)

FUSE is an All-in-One Face Fix KSampler for ComfyUI that provides seamless face upscaling and embedding capabilities using YOLO face detection and SAM (Segment Anything Model) for precise masking.

**NOTE**: While FUSE is aimed at face restoration, it should be noted that it can accept any trained YOLO detection model, and as a result of your prompt, help isolate and fix other areas of images, such as... hands?


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

### Models

#### YOLO Face Models
You'll need to download some YOLO Face models. You can find some from:
 - https://github.com/akanametov/yolo-face
 - https://github.com/Fuyucch1/yolov8_animeface/releases/tag/v1 (Anime Face)
  
Models should be placed in the `models/yolo-face/` directory.

#### SAM Models
You'll also need to download some SAM models. You can find official models from:
 - https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints

Models should be placed in the `models/sams/` directory.

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

| Parameter | Description | Type | Default | Value Range |
|-----------|-------------|------|---------|-------------|
| model | Base model for sampling | MODEL | Required | N/A |
| vae | VAE model for the sampler | VAE | Required | N/A |
| images | Input image batch for face detection and upscaling | IMAGE | Required | N/A |
| positive | Positive conditioning for the sampler | CONDITIONING | Required | N/A |
| negative | Negative conditioning for the sampler | CONDITIONING | Required | N/A |
| use_cache | Use internal caching to speed up workflow iteration | BOOLEAN | True | True/False |
| seed | Seed for deterministic results | INT | 0 | ≥ 0 |
| steps | Number of steps for the sampler | INT | 20 | ≥ 1 |
| cfg | Classifier-Free Guidance scale | FLOAT | 8.0 | Any float value |
| sampler_name | Name of the sampler | STRING | (from KSampler) | Available KSampler options |
| scheduler | Scheduler type | STRING | (from KSampler) | Available KSampler schedulers |
| denoise | Denoising strength (lower values adhere more to input face) | FLOAT | 0.5 | 0.0 - 1.0 |
| yolo_detector | YOLO model for face detection | STRING | Required | Available YOLO face models |
| sam_segmenter | SAM model for face segmentation | STRING | Required | Available SAM models |
| sam_model_type | SAM model type | STRING | "vit_b" | "vit_b", "vit_l", "vit_h" |
| face_id | Index of the face to process (0 is first face) | INT | 0 | ≥ 0 |
| face_size | Resolution to sample the face crop at | INT | 512 | 512, 768, 1024, 1280, 1536 |
| face_padding | Padding in pixels around face crop | INT | 20 | 0 - `nodes.MAX_RESOLUTION` (Currently 16384) |
| force_square | Force 1:1 square face crops | BOOLEAN | True | True/False |
| blend_amount | Amount of blending for face embedding | FLOAT | 0.3 | 0.0 - 1.0 |
| blend_mode | Feathering mode for blending | STRING | "box" | "box", "radial" |
| use_sam_mask | Use SAM face mask for blending | BOOLEAN | False | True/False |
| face_color_transfer | Color transfer mode | STRING | "none" | "none", "lab", "pdf", "mean_std" |

### FUSESamplerMaskOptions

Additional mask processing options for the FUSEKSampler.

| Parameter | Description | Type | Default | Value Range |
|-----------|-------------|------|---------|-------------|
| mask_dilation | Amount to dilate the mask | INT | 0 | 0 - `nodes.MAX_RESOLUTION` (Currently 16384) |
| mask_erosion | Amount to erode the mask | INT | 0 | 0 - `nodes.MAX_RESOLUTION` (Currently 16384) |
| mask_smoothing | Smoothing factor for mask edges | FLOAT | 0.0 | 0.0 - 1024.0 |
| mask_fill_holes | Fill holes in the mask | BOOL | True | True/False |


## Workflows

You can find example workflows in the `workflows` [directory](/workflows).

## Requirements

- Python >= 3.10
- torch
- numpy
- Pillow
- scipy
- ultralytics (YOLO)
- segment-anything (SAM)

## License

MIT License

## Credits

Created by WASasquatch (https://github.com/WASasquatch)