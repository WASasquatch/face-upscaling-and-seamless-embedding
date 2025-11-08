# Video Support for FUSE Nodes

## Overview

The FUSE nodes now support **temporal-aware video processing** with face tracking across frames. This enables using FUSE with video models like Wan 2.2 for enhanced face/object upscaling in videos.

## New Features

### 1. **FUSEVideoKSampler Node**

A new video-aware sampler that:
- ✅ Supports both **4D image batches** `[N, H, W, C]` and **5D video tensors** `[B, F, H, W, C]`
- ✅ **Tracks faces across frames** using IoU + spatial proximity matching
- ✅ **Maintains temporal consistency** with smoothed bounding boxes
- ✅ **Prevents flickering** through temporal coherence
- ✅ Works with standard image models (processes frame-by-frame)
- ✅ Compatible with video models when they support 5D latents

### 2. **Temporal Tracking Utilities**

New functions in `fuse_utils.py`:

#### `track_faces_in_video(all_frame_boxes, iou_threshold, distance_threshold)`
- Assigns consistent track IDs to faces across all frames
- Handles faces entering/leaving the scene
- Returns tracks with frame indices and bounding boxes

#### `match_faces_across_frames(prev_boxes, curr_boxes, iou_threshold, distance_threshold)`
- Matches faces between consecutive frames
- Uses hybrid IoU + distance scoring
- Prevents duplicate matches

#### `smooth_boxes_temporal(boxes, window_size)`
- Applies moving average smoothing to bounding boxes
- Reduces jitter and improves temporal stability

#### `compute_iou(box1, box2)` & `compute_box_distance(box1, box2)`
- Helper functions for box matching

## How Face Tracking Works

### **Hybrid IoU + Distance Matching**

The tracking algorithm uses a two-stage approach:

1. **Primary Matching (IoU-based)**
   - Computes Intersection over Union between boxes in consecutive frames
   - Matches boxes with IoU ≥ `iou_threshold` (default: 0.3)
   - Prioritizes high IoU matches

2. **Fallback Matching (Distance-based)**
   - For boxes with low IoU, uses center-point distance
   - Matches if distance ≤ `distance_threshold` (default: 100 pixels)
   - Handles fast-moving faces

3. **Track Management**
   - Assigns unique track IDs to each face
   - Maintains tracks across frames
   - Marks tracks inactive when faces leave the scene
   - Creates new tracks for newly appearing faces

### **Temporal Smoothing**

Bounding boxes are smoothed using a moving average window:
- Default window size: 3 frames
- Reduces jitter from detection noise
- Maintains face position stability

## Usage

### Basic Video Processing

```python
# Input: 5D video tensor [1, 16, 1024, 1024, 3]
# - 1 batch
# - 16 frames
# - 1024x1024 resolution
# - 3 channels (RGB)

FUSEVideoKSampler(
    model=video_model,  # e.g., Wan 2.2
    vae=video_vae,
    images=video_input,  # 5D tensor
    positive=positive_cond,
    negative=negative_cond,
    temporal_tracking=True,  # Enable face tracking
    iou_threshold=0.3,       # IoU threshold for matching
    smooth_boxes=True,       # Enable temporal smoothing
    smooth_window=3,         # Smoothing window size
    face_id=-1,              # Process all faces
    # ... other parameters
)
```

### Image Batch Processing (Backward Compatible)

```python
# Input: 4D image batch [10, 512, 512, 3]
# Works exactly like the original FUSEKSampler

FUSEVideoKSampler(
    model=image_model,
    vae=image_vae,
    images=image_batch,  # 4D tensor
    temporal_tracking=False,  # Disable tracking for images
    # ... other parameters
)
```

## Parameters

### Temporal Tracking Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `temporal_tracking` | bool | `True` | Enable face tracking across frames |
| `iou_threshold` | float | `0.3` | IoU threshold for face matching (0-1) |
| `smooth_boxes` | bool | `True` | Apply temporal smoothing to boxes |
| `smooth_window` | int | `3` | Smoothing window size (must be odd) |

### Other Parameters

All standard FUSE parameters are supported:
- Model, VAE, conditioning
- YOLO detector, SAM segmenter
- Face size, padding, order
- Mask options, blending settings
- Caching

## Processing Pipeline

### For Video Inputs (5D)

1. **Detection Phase**
   - Detect faces in all frames using YOLO
   - Cache detection results

2. **Tracking Phase** (if `temporal_tracking=True`)
   - Track faces across frames using IoU + distance
   - Assign consistent track IDs
   - Apply temporal smoothing to boxes

3. **Processing Phase**
   - Process each track across its frames
   - Maintain temporal coherence
   - Apply face enhancement per frame

4. **Blending Phase**
   - Blend enhanced faces back into frames
   - Reshape output to 5D format

### For Image Batches (4D)

- Processes each image independently
- No tracking or temporal smoothing
- Identical to original FUSEKSampler behavior

## Video Model Compatibility

### Current Implementation (Frame-by-Frame)

The current implementation processes video **frame-by-frame**:
- ✅ Works with any image model
- ✅ Maintains temporal consistency via tracking
- ⚠️ Doesn't use video model's temporal capabilities

### Future Enhancement (True 5D Processing)

To fully leverage video models like Wan 2.2:

1. **Batch temporal crops**
   ```python
   # Extract face crops across multiple frames
   face_sequence = [crop_frame_0, crop_frame_1, ..., crop_frame_N]
   # Stack to 5D: [1, N, H, W, C]
   ```

2. **Encode to 5D latents**
   ```python
   # Video VAE: [B, F, H, W, C] -> [B, C, F, H_latent, W_latent]
   latents_5d = video_vae.encode(face_sequence)
   ```

3. **Sample with video model**
   ```python
   # Video model processes temporal dimension
   out_latents = video_model.sample(latents_5d)
   ```

4. **Decode and blend**
   ```python
   enhanced_faces = video_vae.decode(out_latents)
   ```

This would require:
- Video VAE support in ComfyUI
- Video model compatibility
- Temporal crop extraction logic

## Example Use Cases

### 1. Face Enhancement in Video (Wan 2.2)

```python
# Enhance faces in a talking head video
# Input: 30 frames of 1920x1080 video
# Process: Detect, track, and enhance all faces
# Output: Enhanced video with improved face quality
```

### 2. Object Enhancement in Video

```python
# Use FUSEGenericKSampler with video support
# Track and enhance specific objects (cars, products, etc.)
# Maintain temporal consistency across frames
```

### 3. Multi-Face Video Processing

```python
# Process multiple faces in a group video
# Track each person independently
# Apply different enhancements per track
```

## Performance Considerations

### Memory Usage

- Video processing is memory-intensive
- 5D tensors: `[B, F, H, W, C]` can be large
- Consider processing in chunks for long videos

### Caching

- Detection results are cached per frame
- Diffusion results are cached per crop
- Reduces redundant computation on re-runs

### Speed

- Frame-by-frame processing is parallelizable
- YOLO detection is the bottleneck
- Consider using smaller YOLO models for speed

## Limitations

1. **No true 5D latent processing yet**
   - Current implementation processes frames individually
   - Video models' temporal features not fully utilized

2. **Face tracking limitations**
   - Simple IoU + distance matching
   - May lose tracks with extreme motion or occlusions
   - No re-identification after track loss

3. **Memory constraints**
   - Long videos may require chunking
   - High-resolution videos are memory-intensive

## Future Improvements

1. **True 5D latent processing**
   - Extract temporal face crops
   - Process with video models
   - Leverage temporal attention

2. **Advanced tracking**
   - Deep learning trackers (DeepSORT, ByteTrack)
   - Appearance-based re-identification
   - Optical flow integration

3. **Temporal mask smoothing**
   - Smooth masks across frames
   - Reduce blending artifacts

4. **Chunk processing**
   - Process long videos in chunks
   - Maintain track consistency across chunks

## Conclusion

The new `FUSEVideoKSampler` provides a solid foundation for video face enhancement with temporal awareness. While it currently processes frames individually, the tracking system ensures temporal consistency and prevents flickering. Future enhancements will enable true 5D latent processing for full video model support.
