# Temporal Face Processing for Video Models

## FUSEVideoLatentKSampler

A clean, efficient approach to video face enhancement using temporal face sequences.

## Workflow

```
Input Video [N, H, W, C]
    ↓
1. YOLO detects faces in all frames
    ↓
2. Track faces across frames (IoU + distance matching)
    ↓
3. For each face track:
    ├─ Crop face from each frame
    ├─ Stack into temporal sequence [F, H, W, C]
    ├─ Encode with Video VAE → 5D latents
    ├─ Sample with Video Model (temporal coherence!)
    ├─ Decode enhanced sequence
    └─ Paste back into original frames
    ↓
Output Video [N, H, W, C]
```

## Key Advantages

### ✅ **True Temporal Processing**
- Each face is processed as a **temporal sequence**
- Video model sees the face across multiple frames
- Natural temporal coherence and consistency

### ✅ **Efficient**
- Only processes face regions, not entire frames
- Video VAE encodes/decodes small face crops
- Much faster than full-frame processing

### ✅ **Smooth Tracking**
- YOLO detects faces in every frame
- IoU + distance matching tracks faces
- Temporal smoothing reduces jitter
- Handles faces entering/leaving scene

### ✅ **Clean Architecture**
- No complex latent-space manipulation
- Standard VAE encode → sample → decode pipeline
- Works with any video model (Wan 2.2, etc.)

## Usage Example

```python
FUSEVideoLatentKSampler(
    model=wan_2_2_model,      # Video model
    vae=wan_vae,              # Video VAE
    images=video_frames,      # [N, H, W, C] - standard ComfyUI format
    positive=positive_cond,
    negative=negative_cond,
    
    # Face detection
    yolo_detector="yolov8n-face.pt",
    face_id=-1,               # Process all faces
    face_size=512,            # Crop resolution
    face_padding=20,
    force_square=True,
    
    # Temporal tracking
    iou_threshold=0.3,        # IoU threshold for matching
    smooth_boxes=True,        # Smooth bounding boxes
    smooth_window=3,          # Smoothing window
    
    # Sampling
    seed=42,
    steps=20,
    cfg=8.0,
    denoise=0.5,
    sampler_name="euler",
    scheduler="normal",
    
    # Blending
    mask_optionals={
        "blend_amount": 0.3,
        "blend_mode": "box",
    }
)
```

## How It Works

### 1. Face Detection
```python
# Detect faces in all frames
for each frame:
    boxes = yolo.detect(frame)
    all_frame_boxes.append(boxes)
```

### 2. Temporal Tracking
```python
# Track faces across frames
tracks = track_faces_in_video(all_frame_boxes)
# Returns: [
#   {
#     'track_id': 0,
#     'frames': [0, 1, 2, 3, ...],
#     'boxes': [[x1,y1,x2,y2], ...]
#   },
#   ...
# ]
```

### 3. Process Each Track
```python
for track in tracks:
    # Crop face from each frame
    face_crops = []
    for frame_idx, box in zip(track['frames'], track['boxes']):
        crop = extract_and_resize(frames[frame_idx], box, face_size)
        face_crops.append(crop)
    
    # Stack into sequence [F, H, W, C]
    face_sequence = stack(face_crops)
    
    # Encode with video VAE
    latents = vae.encode(face_sequence)  # → 5D latents
    
    # Sample with video model
    enhanced_latents = model.sample(latents)
    
    # Decode
    enhanced_sequence = vae.decode(enhanced_latents)
    
    # Paste back
    for i, frame_idx in enumerate(track['frames']):
        paste_face(frames[frame_idx], enhanced_sequence[i], box)
```

## Face Tracking Details

### IoU + Distance Hybrid Matching
```python
# For each face in frame N-1, find best match in frame N
for prev_face in frame_N_minus_1:
    best_match = None
    best_score = 0
    
    for curr_face in frame_N:
        iou = compute_iou(prev_face, curr_face)
        distance = compute_distance(prev_face, curr_face)
        
        if iou >= 0.3:
            # High overlap - likely same face
            score = iou + (1 / (1 + distance/100))
        elif distance <= 100:
            # Low overlap but close - might be fast motion
            score = 1 / (1 + distance/100)
        
        if score > best_score:
            best_match = curr_face
            best_score = score
    
    if best_match:
        assign_to_same_track(prev_face, best_match)
    else:
        mark_track_inactive(prev_face)
```

### Temporal Smoothing
```python
# Smooth bounding boxes with moving average
def smooth_boxes(boxes, window=3):
    smoothed = []
    for i in range(len(boxes)):
        window_boxes = boxes[max(0, i-1):min(len(boxes), i+2)]
        avg_box = average(window_boxes)
        smoothed.append(avg_box)
    return smoothed
```

## Video VAE Compatibility

The node works with video VAEs that accept standard ComfyUI IMAGE format:
- **Input**: `[N, H, W, C]` - batch of frames
- **Encode**: Produces latents (format depends on VAE)
- **Decode**: Reconstructs `[N, H, W, C]`

For true 5D latent support (e.g., `[B, C, F, H, W]`), the VAE must handle temporal dimensions internally.

## Comparison: FUSEVideoLatentKSampler vs FUSEVideoKSampler

| Feature | FUSEVideoLatentKSampler | FUSEVideoKSampler |
|---------|------------------------|-------------------|
| **Input** | 4D images `[N,H,W,C]` | 4D or 5D |
| **Processing** | Temporal face sequences | Frame-by-frame |
| **VAE** | Encodes face sequences | Encodes individual crops |
| **Temporal coherence** | ✅ Video model sees sequence | ⚠️ Via tracking only |
| **Efficiency** | ✅ Processes face regions | ⚠️ Processes full crops |
| **Best for** | Video models (Wan 2.2) | Image models on video |

## Performance Considerations

### Memory Usage
- **Face crops**: Much smaller than full frames
- **Typical**: 512x512 face vs 1920x1080 frame = ~14x less data
- **Multiple faces**: Processed sequentially (one track at a time)

### Speed
- **YOLO detection**: ~10-50ms per frame (depends on model)
- **VAE encode/decode**: Depends on sequence length and face size
- **Sampling**: Depends on model, steps, and latent size

### Optimization Tips
1. **Use smaller YOLO models** for faster detection
2. **Reduce face_size** if quality allows (512 vs 1024)
3. **Process fewer frames** for long videos (sample keyframes)
4. **Batch multiple tracks** if memory allows (future enhancement)

## Limitations

1. **Track Loss**
   - Faces that leave and re-enter get new track IDs
   - No re-identification after track loss
   - Solution: Use more advanced trackers (DeepSORT, etc.)

2. **Occlusions**
   - Partially occluded faces may lose tracking
   - Solution: Lower IoU threshold or use appearance features

3. **Fast Motion**
   - Very fast motion may break tracking
   - Solution: Increase distance threshold or use optical flow

4. **Multiple Faces**
   - Processed sequentially, not in parallel
   - Solution: Batch processing (future enhancement)

## Future Enhancements

1. **Batch Processing**
   - Process multiple tracks in parallel
   - Encode all face sequences together

2. **Advanced Tracking**
   - Appearance-based re-identification
   - Optical flow integration
   - Deep learning trackers

3. **Adaptive Cropping**
   - Adjust crop size based on face size
   - Handle varying face scales

4. **Temporal Padding**
   - Add context frames before/after track
   - Improve temporal coherence at track boundaries

## Conclusion

`FUSEVideoLatentKSampler` provides a clean, efficient approach to video face enhancement by:
- Tracking faces across frames
- Processing each face as a temporal sequence
- Leveraging video models' temporal capabilities
- Pasting enhanced faces back seamlessly

This is the **recommended approach** for using FUSE with video models like Wan 2.2.
