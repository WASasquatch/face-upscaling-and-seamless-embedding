<div align="center">
  <img src="banner.png" alt="FUSE Banner">
</div>

# Face Upscaling and Seamless Embedding (FUSE)

FUSE is an All-in-One Face Fix KSampler for ComfyUI that provides seamless face upscaling and embedding capabilities using YOLO face detection and SAM (Segment Anything Model) for precise masking.

## Before and After Examples

- https://imgsli.com/Mzk5NzU1
- https://imgsli.com/Mzk5NzUx
- https://imgsli.com/Mzk5NzUz


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

#### YOLO Models
You'll need to download some YOLO models. You can find some from:
 - https://github.com/akanametov/yolo-face
 - https://github.com/Fuyucch1/yolov8_animeface/releases/tag/v1 (Anime Face)

Models should be placed in the `models/yolo/` directory.

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
| yolo_confidence | YOLO confidence threshold (higher values detect only more confident faces) | FLOAT | 0.25 | 0.0 - 1.0 |
| yolo_nms_iou | YOLO NMS IoU threshold (lower values remove more overlapping detections) | FLOAT | 0.5 | 0.0 - 1.0 |
| sam_segmenter | SAM model for face segmentation | STRING | Required | Available SAM models |
| sam_model_type | SAM model type | STRING | "vit_b" | "vit_b", "vit_l", "vit_h" |
| face_id | Index of the face to process (0 is first face, -1 is all faces) | INT | 0 | ≥ -1 |
| face_order | Order to process detected faces | STRING | "linear" | "linear", "linear_reverse", "largest_bbox", "smallest_bbox" |
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
| blend_sampling | Resampling method for resizing operations (face/mask crop and paste back) | STRING | "bilinear" | "bilinear", "lanczos", "nearest", "bicubic", "box", "hamming" |

### FUSEKSampler (Video)

Video-aware face enhancement sampler with temporal tracking capabilities. This node processes video inputs (5D tensors) or image batches (4D tensors) and maintains face consistency across frames through intelligent tracking. Each unique person is processed as their own video batch, enabling true temporal coherence.

**Key Features:**
- Temporal face tracking across frames with IoU-based matching
- Per-person video batch processing
- Temporal smoothing of bounding boxes for stable crops
- Support for both face-specific and general YOLO models

**Use Cases:**
- Video face enhancement with temporal consistency
- General object enhancement with temporal consistency
- Multi-person video processing with individual tracking

| Parameter | Description | Type | Default | Value Range |
|-----------|-------------|------|---------|-------------|
| model | Base model for sampling (should support video if input is 5D) | MODEL | Required | N/A |
| vae | VAE model for the sampler (should support video if input is 5D) | VAE | Required | N/A |
| images | Input image batch (4D) or video (5D) for face detection and upscaling | IMAGE | Required | N/A |
| positive | Positive conditioning for the sampler | CONDITIONING | Required | N/A |
| negative | Negative conditioning for the sampler | CONDITIONING | Required | N/A |
| use_cache | Use internal caching to speed up workflow iteration | BOOLEAN | True | True/False |
| seed | Seed for deterministic results | INT | 0 | ≥ 0 |
| steps | Number of steps for the sampler | INT | 20 | ≥ 1 |
| cfg | Classifier-Free Guidance scale | FLOAT | 8.0 | Any float value |
| sampler_name | Name of the sampler | STRING | (from KSampler) | Available KSampler options |
| scheduler | Scheduler type | STRING | (from KSampler) | Available KSampler schedulers |
| denoise | Denoising strength for the sampler | FLOAT | 0.5 | 0.0 - 1.0 |
| yolo_detector | YOLO model for face detection (face/ or yolo/ prefixed) | STRING | Required | Available YOLO models |
| sam_segmenter | SAM model for face segmentation | STRING | Required | Available SAM models |
| sam_model_type | SAM model type | STRING | "vit_b" | "vit_b", "vit_l", "vit_h" |
| face_id | Index of the face to process (-1 for all faces) | INT | 0 | ≥ -1 |
| face_order | Order to process detected faces | STRING | "linear" | "linear", "linear_reverse", "largest_bbox", "smallest_bbox" |
| face_size | Resolution to sample the face crop at | INT | 512 | 512, 768, 1024, 1280, 1536 |
| face_padding | Padding in pixels around face crop | INT | 20 | 0 - `nodes.MAX_RESOLUTION` (Currently 16384) |
| force_square | Force 1:1 square face crops | BOOLEAN | True | True/False |
| temporal_tracking | Enable temporal face tracking for video inputs | BOOLEAN | True | True/False |
| mask_optionals | Optional masking and blending settings | DICT | Optional | From FUSESamplerMaskOptions |
| yolo_optionals | Optional YOLO detection and tracking settings | DICT | Optional | From FUSEYOLOSettings |

### FUSEYOLOSettings

Optional YOLO detection and temporal tracking settings for video processing.

| Parameter | Description | Type | Default | Value Range |
|-----------|-------------|------|---------|-------------|
| confidence | YOLO confidence threshold | FLOAT | 0.25 | 0.0 - 1.0 |
| iou_threshold | YOLO NMS IoU threshold | FLOAT | 0.5 | 0.0 - 1.0 |
| max_detections | Maximum number of detections to keep | INT | 300 | 1 - 1000 |
| class_filter | Comma or newline separated class names to filter | STRING | "" | Any string |
| agnostic_nms | Class-agnostic NMS | BOOLEAN | False | True/False |
| half_precision | Use FP16 for faster inference | BOOLEAN | False | True/False |
| augment | Test Time Augmentation for improved accuracy | BOOLEAN | False | True/False |
| tracking_iou_threshold | IoU threshold for face tracking across frames | FLOAT | 0.3 | 0.0 - 1.0 |
| tracking_smooth_boxes | Apply temporal smoothing to bounding boxes | BOOLEAN | True | True/False |
| tracking_smooth_window | Temporal smoothing window size (must be odd) | INT | 3 | 1 - 11 |

### FUSEKSampler (Generic)

A generic version of the FUSEKSampler designed for use with any YOLO detection model, not just face-specific ones. This node provides the same sampling capabilities as the original FUSEKSampler but with more generic parameter naming to avoid confusion when working with non-face detection models.

**Use Cases:**
- Hand detection and enhancement
- General object isolation and processing
- Any scenario where you need to detect, crop, and enhance specific objects in images

| Parameter | Description | Type | Default | Value Range |
|-----------|-------------|------|---------|-------------|
| model | Base model for sampling | MODEL | Required | N/A |
| vae | VAE model for the sampler | VAE | Required | N/A |
| images | Input image batch for mask detection and upscaling | IMAGE | Required | N/A |
| positive | Positive conditioning for the sampler | CONDITIONING | Required | N/A |
| negative | Negative conditioning for the sampler | CONDITIONING | Required | N/A |
| use_cache | Use internal caching to speed up workflow iteration | BOOLEAN | True | True/False |
| seed | Seed for deterministic results | INT | 0 | ≥ 0 |
| steps | Number of steps for the sampler | INT | 20 | ≥ 1 |
| cfg | Classifier-Free Guidance scale | FLOAT | 8.0 | Any float value |
| sampler_name | Name of the sampler | STRING | (from KSampler) | Available KSampler options |
| scheduler | Scheduler type | STRING | (from KSampler) | Available KSampler schedulers |
| denoise | Denoising strength (lower values adhere more to input mask) | FLOAT | 0.5 | 0.0 - 1.0 |
| yolo_detector | YOLO model for mask detection | STRING | Required | Available YOLO models |
| yolo_confidence | YOLO confidence threshold (higher values detect only more confident objects) | FLOAT | 0.25 | 0.0 - 1.0 |
| yolo_nms_iou | YOLO NMS IoU threshold (lower values remove more overlapping detections) | FLOAT | 0.5 | 0.0 - 1.0 |
| sam_segmenter | SAM model for mask segmentation | STRING | Required | Available SAM models |
| sam_model_type | SAM model type | STRING | "vit_b" | "vit_b", "vit_l", "vit_h" |
| mask_id | Index of the mask to process (0 is first mask found, -1 is all masks) | INT | 0 | ≥ -1 |
| mask_order | Order to process detected masks | STRING | "linear" | "linear", "linear_reverse", "largest_bbox", "smallest_bbox" |
| mask_size | Resolution to sample the mask crop at | INT | 512 | 512, 768, 1024, 1280, 1536 |
| mask_padding | Padding in pixels around mask crop | INT | 20 | 0 - `nodes.MAX_RESOLUTION` (Currently 16384) |
| force_square | Force 1:1 square mask crops | BOOLEAN | True | True/False |
| blend_amount | Amount of blending for mask embedding | FLOAT | 0.3 | 0.0 - 1.0 |
| blend_mode | Feathering mode for blending | STRING | "box" | "box", "radial" |
| use_sam_mask | Use SAM mask for blending | BOOLEAN | False | True/False |
| mask_color_transfer | Color transfer mode | STRING | "none" | "none", "lab", "pdf", "mean_std" |


## Workflows

You can find example workflows in the `workflows` [directory](/workflows).

## Known Issues

### Torch 2.6+ Compatibility

Torch 2.6+ versions will not allow loading pickles with Ultralytics due to security restrictions. This may cause issues when loading YOLO models. A pull request has been submitted to address this issue:

- **Issue**: Ultralytics pickle loading incompatibility with Torch 2.6+
- **Status**: PR submitted to Ultralytics: https://github.com/ultralytics/ultralytics/pull/21260
- **Workaround**: Consider using Torch versions < 2.6 if you encounter pickle loading errors.

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