import hashlib
import numpy as np
import os
import torch
import gc
from python_color_transfer.color_transfer import ColorTransfer

from PIL import Image, ImageFilter, ImageOps
from scipy import ndimage
from ultralytics import YOLO, SAM
from segment_anything import sam_model_registry, SamPredictor

from nodes import VAEDecode, VAEEncode
from folder_paths import models_dir

nodes_dir = os.path.dirname(__file__)
yolo_face_models_path = os.path.join(models_dir, "yolo-face")
yolo_models_path = os.path.join(models_dir, "yolo")
sam_models_path = os.path.join(models_dir, "sams")
upscale_models_path = os.path.join(models_dir, "upscale_models")

# PyTorch 2.6+ Pickle fix for YOLO/Ultralytics
if hasattr(torch.serialization, "add_safe_globals"):
    try:
        from ultralytics.nn.tasks import DetectionModel
        torch.serialization.add_safe_globals([DetectionModel])
        print("[FUSE] Added ultralytics.nn.tasks.DetectionModel to torch.serialization.add_safe_globals")
        print("[FUSE] Loading of pickle YOLO models enabled.")
    except Exception:
        print("[FUSE] Failed to add ultralytics.nn.tasks.DetectionModel to torch.serialization.add_safe_globals")
        print("[FUSE] Pickle models will not be loaded with Ultralytics YOLO.")
        pass

def tensor2pil(img_tensor):
    arr = img_tensor.cpu().numpy()
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def pil2tensor(pil_img):
    arr = np.array(pil_img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    return torch.from_numpy(arr)

def compute_tensor_hash(tensor):
    return hashlib.md5(tensor.cpu().numpy().tobytes()).hexdigest()

def compute_diffusion_hash(seed, steps, cfg, sampler_name, scheduler, denoise):
    params = f"{seed}_{steps}_{cfg}_{sampler_name}_{scheduler}_{denoise}"
    return hashlib.md5(params.encode()).hexdigest()

def compute_blend_hash(blend_amount, blend_mode, use_sam_mask, mask_fill_holes,
                       mask_dilation=0, mask_erosion=0, mask_smoothing=0.0):
    params = f"{blend_amount}_{blend_mode}_{use_sam_mask}_{mask_fill_holes}_{mask_dilation}_{mask_erosion}_{mask_smoothing}"
    return hashlib.md5(params.encode()).hexdigest()

def get_model_identity(model):
    if hasattr(model, 'patches_uuid'):
        return str(model.patches_uuid)
    if hasattr(model, 'model') and hasattr(model.model, 'patches_uuid'):
        return str(model.model.patches_uuid)
    return f"{model.__class__.__name__}_nocache"

def compute_conditioning_hash(cond):
    try:
        return hashlib.md5(str(cond).encode()).hexdigest()
    except Exception:
        return str(id(cond)) 

def sort_boxes_by_order(boxes, order_mode):
    def box_center(b):
        return ((b[1] + b[3]) / 2.0, (b[0] + b[2]) / 2.0)
    def box_area(b):
        return (b[2] - b[0]) * (b[3] - b[1])

    linear_sorted = sorted(boxes, key=box_center)

    if order_mode == "linear":
        result = linear_sorted
    elif order_mode == "linear_reverse":
        result = linear_sorted[::-1]
    elif order_mode == "largest_bbox":
        result = sorted(boxes, key=box_area, reverse=True)
    elif order_mode == "smallest_bbox":
        result = sorted(boxes, key=box_area)
    else:
        result = linear_sorted

    return result

def transfer_color(source_img, target_img, mode='lab'):
    source_cv = np.array(source_img)[:, :, ::-1]
    target_cv = np.array(target_img)[:, :, ::-1]
    ct = ColorTransfer()

    if mode == 'lab':
        result = ct.lab_transfer(img_arr_in=target_cv, img_arr_ref=source_cv)

    elif mode == 'pdf':
        result = ct.pdf_transfer(img_arr_in=target_cv, img_arr_ref=source_cv, regrain=True)

    else:  # mean_std
        result = ct.mean_std_transfer(img_arr_in=target_cv, img_arr_ref=source_cv)

    return Image.fromarray(result[:, :, ::-1])

def load_yolo_face_models_list():
    if not os.path.isdir(yolo_face_models_path):
        return []

    return sorted([
        fname for fname in os.listdir(yolo_face_models_path)
        if (fname.lower().endswith(".pth") or fname.lower().endswith(".pt"))
           and os.path.isfile(os.path.join(yolo_face_models_path, fname))
    ])

def load_yolo_models_list():
    if not os.path.isdir(yolo_models_path):
        return []

    return sorted([
        fname for fname in os.listdir(yolo_models_path)
        if (fname.lower().endswith(".pth") or fname.lower().endswith(".pt"))
           and os.path.isfile(os.path.join(yolo_models_path, fname))
    ])

def load_all_yolo_models_list():
    face_models = load_yolo_face_models_list()
    regular_models = load_yolo_models_list()
    
    combined = []
    if face_models:
        combined.extend([f"face/{m}" for m in face_models])
    if regular_models:
        combined.extend([f"yolo/{m}" for m in regular_models])
    
    return combined if combined else ["No models found"]

def load_sam_models_list():
    if not os.path.isdir(sam_models_path):
        return []

    return sorted([
        fname for fname in os.listdir(sam_models_path)
        if (fname.lower().endswith(".pth") or fname.lower().endswith(".pt"))
           and os.path.isfile(os.path.join(sam_models_path, fname))
    ])

def build_feather_mask(w, h, blend_amount, blend_mode):
    if blend_amount <= 0.0:
        return np.ones((h, w), dtype=np.float32)

    if blend_mode == "box":
        mask = np.ones((h, w), dtype=np.float32)
        feather_pixels = max(1, int(blend_amount * min(w, h)))

        for yy in range(h):
            for xx in range(w):
                d = min(xx, yy, w - xx - 1, h - yy - 1)
                if d < feather_pixels:
                    mask[yy, xx] = d / feather_pixels

        return mask

    mask = np.zeros((h, w), dtype=np.float32)
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    max_radius = min(w, h) / 2.0
    feather_pixels = blend_amount * max_radius

    for yy in range(h):
        for xx in range(w):
            r = np.hypot(xx - cx, yy - cy)

            if r <= max_radius - feather_pixels:
                mask[yy, xx] = 1.0

            elif r <= max_radius:
                mask[yy, xx] = max(0.0, (max_radius - r) / feather_pixels)

    return mask

def contract_and_blur_mask(mask_arr, blend_amount):
    if blend_amount <= 0.0:
        return np.clip(mask_arr, 0, 1)

    h, w = mask_arr.shape
    px = min(w, h)
    fa_curve = blend_amount ** 2.2
    contract_px = max(1, int(round(fa_curve * px * 0.1)))
    blur_radius = max(0.5, fa_curve * px * 0.7)
    mask_bin = (mask_arr > 0.5).astype(np.float32)

    if contract_px > 0:
        mask_contracted = ndimage.binary_erosion(mask_bin, iterations=contract_px).astype(np.float32)
    
    else:
        mask_contracted = mask_bin

    mask_pil = Image.fromarray((mask_contracted * 255).astype(np.uint8), mode="L")
    
    if blur_radius > 0.0:
        mask_pil = mask_pil.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    mask_final = np.array(mask_pil).astype(np.float32) / 255.0
    
    return np.clip(mask_final, 0, 1)


def apply_edge_blending(mask_arr, blend_amount, near_white_threshold=0.001):
    """
    Apply edge blending to mask borders where mask touches the edges.
    Fades edge pixels to zero, ramping up to the original mask value over a blend width proportional to blend_amount.
    Args:
        mask_arr: 2D numpy array (0-1 range)
        blend_amount: (0-1 range), controls width of fade
        near_white_threshold: threshold for edge detection (default 0.001)
    Returns:
        Modified mask array with seamless edge fade
    """
    if blend_amount <= 0:
        return mask_arr
    h, w = mask_arr.shape
    modified_mask = mask_arr.copy()
    blend_width = max(1, int(min(h, w) * blend_amount * 0.2))
    borders_to_blend = {
        'top': np.any(mask_arr[0, :] >= near_white_threshold),
        'bottom': np.any(mask_arr[-1, :] >= near_white_threshold),
        'left': np.any(mask_arr[:, 0] >= near_white_threshold),
        'right': np.any(mask_arr[:, -1] >= near_white_threshold)
    }
    for border, should_blend in borders_to_blend.items():
        if not should_blend:
            continue
        if border == 'top':
            for i in range(min(blend_width, h)):
                alpha = (i + 1) / blend_width  # 0 at edge, 1 at blend_width
                modified_mask[i, :] = mask_arr[i, :] * alpha
        elif border == 'bottom':
            for i in range(min(blend_width, h)):
                row_idx = h - 1 - i
                alpha = (i + 1) / blend_width
                modified_mask[row_idx, :] = mask_arr[row_idx, :] * alpha
        elif border == 'left':
            for i in range(min(blend_width, w)):
                alpha = (i + 1) / blend_width
                modified_mask[:, i] = mask_arr[:, i] * alpha
        elif border == 'right':
            for i in range(min(blend_width, w)):
                col_idx = w - 1 - i
                alpha = (i + 1) / blend_width
                modified_mask[:, col_idx] = mask_arr[:, col_idx] * alpha
    return np.clip(modified_mask, 0, 1)


def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) between two bounding boxes.
    Args:
        box1, box2: [x1, y1, x2, y2] format
    Returns:
        IoU score (0-1)
    """
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    if x2_inter <= x1_inter or y2_inter <= y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0


def compute_box_distance(box1, box2):
    """
    Compute center distance between two bounding boxes.
    Args:
        box1, box2: [x1, y1, x2, y2] format
    Returns:
        Euclidean distance between centers
    """
    cx1 = (box1[0] + box1[2]) / 2.0
    cy1 = (box1[1] + box1[3]) / 2.0
    cx2 = (box2[0] + box2[2]) / 2.0
    cy2 = (box2[1] + box2[3]) / 2.0
    return np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)


def match_faces_across_frames(prev_boxes, curr_boxes, iou_threshold=0.3, distance_threshold=100.0):
    """
    Match faces between consecutive frames using IoU and spatial proximity.
    Args:
        prev_boxes: List of boxes from previous frame [[x1,y1,x2,y2], ...]
        curr_boxes: List of boxes from current frame
        iou_threshold: Minimum IoU to consider a match
        distance_threshold: Maximum center distance to consider a match
    Returns:
        List of tuples (prev_idx, curr_idx) representing matches
    """
    if not prev_boxes or not curr_boxes:
        return []
    
    matches = []
    used_curr = set()
    
    # For each previous box, find best match in current frame
    for prev_idx, prev_box in enumerate(prev_boxes):
        best_score = -1
        best_curr_idx = -1
        
        for curr_idx, curr_box in enumerate(curr_boxes):
            if curr_idx in used_curr:
                continue
            
            iou = compute_iou(prev_box, curr_box)
            distance = compute_box_distance(prev_box, curr_box)
            
            # Combined score: prioritize IoU, use distance as tiebreaker
            if iou >= iou_threshold:
                score = iou + (1.0 / (1.0 + distance / 100.0))  # Normalize distance
                if score > best_score:
                    best_score = score
                    best_curr_idx = curr_idx
            elif distance <= distance_threshold:
                # Fallback to distance-based matching if IoU is low
                score = 1.0 / (1.0 + distance / 100.0)
                if score > best_score:
                    best_score = score
                    best_curr_idx = curr_idx
        
        if best_curr_idx >= 0:
            matches.append((prev_idx, best_curr_idx))
            used_curr.add(best_curr_idx)
    
    return matches


def track_faces_in_video(all_frame_boxes, iou_threshold=0.3, distance_threshold=100.0):
    """
    Track faces across all video frames, assigning consistent IDs.
    Args:
        all_frame_boxes: List of frame detections, each frame is list of boxes [[x1,y1,x2,y2],...]
        iou_threshold: IoU threshold for matching
        distance_threshold: Distance threshold for matching
    Returns:
        List of tracks, where each track is a dict:
        {
            'track_id': int,
            'frames': [frame_idx, ...],
            'boxes': [box, ...],  # One box per frame
        }
    """
    if not all_frame_boxes:
        return []
    
    tracks = []
    next_track_id = 0
    
    # Initialize tracks from first frame
    for box in all_frame_boxes[0]:
        tracks.append({
            'track_id': next_track_id,
            'frames': [0],
            'boxes': [box],
            'active': True
        })
        next_track_id += 1
    
    # Process subsequent frames
    for frame_idx in range(1, len(all_frame_boxes)):
        curr_boxes = all_frame_boxes[frame_idx]
        
        # Get active tracks from previous frame
        active_tracks = [t for t in tracks if t['active']]
        prev_boxes = [t['boxes'][-1] for t in active_tracks]
        
        # Match current detections to previous tracks
        matches = match_faces_across_frames(prev_boxes, curr_boxes, iou_threshold, distance_threshold)
        
        matched_curr = set()
        for prev_idx, curr_idx in matches:
            track = active_tracks[prev_idx]
            track['frames'].append(frame_idx)
            track['boxes'].append(curr_boxes[curr_idx])
            matched_curr.add(curr_idx)
        
        # Mark unmatched tracks as inactive
        for prev_idx, track in enumerate(active_tracks):
            if not any(m[0] == prev_idx for m in matches):
                track['active'] = False
        
        # Create new tracks for unmatched detections
        for curr_idx, box in enumerate(curr_boxes):
            if curr_idx not in matched_curr:
                tracks.append({
                    'track_id': next_track_id,
                    'frames': [frame_idx],
                    'boxes': [box],
                    'active': True
                })
                next_track_id += 1
    
    # Clean up active flags
    for track in tracks:
        del track['active']
    
    return tracks


def smooth_boxes_temporal(boxes, window_size=3):
    """
    Apply temporal smoothing to bounding boxes using moving average.
    Args:
        boxes: List of boxes [x1, y1, x2, y2] across frames
        window_size: Size of smoothing window (must be odd)
    Returns:
        Smoothed boxes
    """
    if len(boxes) <= 1 or window_size <= 1:
        return boxes
    
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    half_window = window_size // 2
    
    smoothed = []
    for i in range(len(boxes)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(boxes), i + half_window + 1)
        window_boxes = boxes[start_idx:end_idx]
        
        # Average coordinates
        avg_box = [
            sum(b[0] for b in window_boxes) / len(window_boxes),
            sum(b[1] for b in window_boxes) / len(window_boxes),
            sum(b[2] for b in window_boxes) / len(window_boxes),
            sum(b[3] for b in window_boxes) / len(window_boxes),
        ]
        smoothed.append(avg_box)
    
    return smoothed


class FUSEBase:
    def __init__(self):
        self.cache = {}
        self.vae_encode = VAEEncode()
        self.vae_decode = VAEDecode()
        self.yolo_model = None
        self.current_yolo_model_name = None
        self.sam_model = None
        self.sam_predictor = None
        self.current_sam_model_name = None
        self.current_sam_model_type = None
        self.sam_is_ultralytics = False

    def _load_yolo_model(self, model_name, device, is_face_model=False):
        if self.yolo_model is None or self.current_yolo_model_name != model_name:
            if self.yolo_model is not None:
                self._move_yolo_to_cpu()
                del self.yolo_model
                gc.collect()

            # Handle prefixed model names (face/ or yolo/)
            if model_name.startswith("face/"):
                actual_model_name = model_name[5:]  # Remove "face/" prefix
                model_path = os.path.join(yolo_face_models_path, actual_model_name)
            elif model_name.startswith("yolo/"):
                actual_model_name = model_name[5:]  # Remove "yolo/" prefix
                model_path = os.path.join(yolo_models_path, actual_model_name)
            else:
                # Fallback to old behavior for backward compatibility
                model_path = os.path.join(yolo_face_models_path if is_face_model else yolo_models_path, model_name)
            
            self.yolo_model = YOLO(model_path)
            self.current_yolo_model_name = model_name

        try:
            dev = getattr(self.yolo_model.model, 'device', getattr(self.yolo_model, 'device', None))
            if dev is None or dev != device:
                self.yolo_model.to(device)
        except:
            try:
                self.yolo_model.to(device)
            except:
                pass

    
    def _apply_yolo_settings(self, yolo_settings=None):
        """Apply YOLO settings to the loaded model"""
        if self.yolo_model is None or yolo_settings is None:
            return
            
        self.yolo_model.conf = yolo_settings.get('confidence', 0.25)
        self.yolo_model.iou = yolo_settings.get('iou_threshold', 0.5)
        self.yolo_model.max_det = yolo_settings.get('max_detections', 300)
        self.yolo_model.agnostic_nms = yolo_settings.get('agnostic_nms', False)
        
        if hasattr(self.yolo_model, 'half') and yolo_settings.get('half_precision', False):
            self.yolo_model.half()
        
        self._yolo_class_filter = yolo_settings.get('filtered_classes', [])
        self._yolo_augment = yolo_settings.get('augment', False)
    
    def _filter_detections_by_class(self, results):
        """Filter YOLO detection results by specified classes"""
        if not hasattr(self, '_yolo_class_filter') or not self._yolo_class_filter:
            return results
            
        if not results or len(results) == 0:
            return results
            
        try:
            model_names = self.yolo_model.names 
            if not model_names:
                return results
                
            name_to_id = {name.lower(): class_id for class_id, name in model_names.items()}
            filter_class_ids = []
            
            for filter_name in self._yolo_class_filter:
                filter_name_lower = filter_name.lower()
                if filter_name_lower in name_to_id:
                    filter_class_ids.append(name_to_id[filter_name_lower])
                else:
                    for name, class_id in name_to_id.items():
                        if filter_name_lower in name or name in filter_name_lower:
                            filter_class_ids.append(class_id)
                            break
            
            if not filter_class_ids:
                return results 
                
            filtered_results = []
            for result in results:
                if hasattr(result, 'boxes') and result.boxes is not None:
                    if hasattr(result.boxes, 'cls'):
                        class_ids = result.boxes.cls.cpu().numpy()
                        mask = np.isin(class_ids, filter_class_ids)
                        
                        if mask.any():
                            filtered_result = type(result)()
                            for attr in dir(result):
                                if not attr.startswith('_') and hasattr(result, attr):
                                    setattr(filtered_result, attr, getattr(result, attr))
                            
                            if hasattr(result.boxes, 'xyxy'):
                                filtered_result.boxes.xyxy = result.boxes.xyxy[mask]
                            if hasattr(result.boxes, 'conf'):
                                filtered_result.boxes.conf = result.boxes.conf[mask]
                            if hasattr(result.boxes, 'cls'):
                                filtered_result.boxes.cls = result.boxes.cls[mask]
                                
                            filtered_results.append(filtered_result)
                else:
                    filtered_results.append(result)
                    
            return filtered_results
            
        except Exception as e:
            print(f"[FUSE] Warning: Class filtering failed: {e}")
            return results
        
        return results

    def _load_sam_model(self, model_name, model_type, device):
        if (self.sam_model is None or
            self.current_sam_model_name != model_name or
            self.current_sam_model_type != model_type):

            if self.sam_model is not None:
                self._move_sam_to_cpu()
                del self.sam_model
                if hasattr(self, 'sam_predictor') and self.sam_predictor is not None:
                    del self.sam_predictor
                gc.collect()

            checkpoint = os.path.join(sam_models_path, model_name)
            
            is_ultralytics_sam = self._is_ultralytics_sam_model(model_name)
            
            if is_ultralytics_sam:
                print(f"[FUSE] Loading Ultralytics SAM model: {model_name}")
                try:
                    self.sam_model = SAM(checkpoint)
                    self.sam_predictor = None
                    self.sam_is_ultralytics = True
                except Exception as e:
                    print(f"[FUSE] Failed to load Ultralytics SAM model: {e}")
                    print(f"[FUSE] Falling back to traditional SAM loading...")
                    self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
                    self.sam_predictor = SamPredictor(self.sam_model)
                    self.sam_is_ultralytics = False
            else:
                print(f"[FUSE] Loading traditional SAM model: {model_name}")
                self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
                self.sam_predictor = SamPredictor(self.sam_model)
                self.sam_is_ultralytics = False
                
            self.current_sam_model_name = model_name
            self.current_sam_model_type = model_type
            
        if not self.sam_is_ultralytics and self.sam_model.device != device:
            self.sam_model.to(device)
        elif self.sam_is_ultralytics:
            try:
                self.sam_model.to(device)
            except:
                pass
    
    def _is_ultralytics_sam_model(self, model_name):
        """Detect if a SAM model is an Ultralytics model based on filename"""
        model_name_lower = model_name.lower()
        ultralytics_indicators = [
            'sam2', 'sam2.1', 'sam_2', 'mobile_sam', 'fast_sam', 'fastsam'
        ]
        return any(indicator in model_name_lower for indicator in ultralytics_indicators)
    
    def _segment_with_sam(self, image_pil, bbox, device):
        """Unified SAM segmentation method for both traditional and Ultralytics SAM"""
        if self.sam_model is None:
            return None
            
        try:
            if self.sam_is_ultralytics:
                results = self.sam_model(image_pil, bboxes=[bbox])
                if len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                    mask = results[0].masks.data[0].cpu().numpy()
                    return (mask > 0.5).astype(np.uint8) * 255
                return None
            else:
                self.sam_predictor.set_image(np.array(image_pil))
                x1, y1, x2, y2 = bbox
                input_box = np.array([x1, y1, x2, y2])
                masks, _, _ = self.sam_predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=input_box[None, :],
                    multimask_output=False,
                )
                return (masks[0] * 255).astype(np.uint8)
        except Exception as e:
            print(f"[FUSE] SAM segmentation failed: {e}")
            return None

    def _move_yolo_to_cpu(self):
        if self.yolo_model is not None:
            try:
                self.yolo_model.to('cpu')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    def _move_yolo_to_device(self, device):
        if self.yolo_model is not None:
            try:
                self.yolo_model.to(device)
            except:
                pass

    def _move_sam_to_cpu(self):
        if self.sam_model is not None:
            try:
                self.sam_model.to('cpu')
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except:
                pass

    def _move_sam_to_device(self, device):
        if self.sam_model is not None:
            self.sam_model.to(device)

    def _process_mask(self, mask_arr, mask_dilation=0, mask_erosion=0, mask_smoothing=0.0, mask_fill_holes=True):
        if mask_arr.max() > 1.0:
            mask_arr = mask_arr / 255.0

        if mask_fill_holes:
            mask_arr = ndimage.binary_fill_holes(mask_arr > 0.5).astype(np.float32)

        else:
            mask_arr = (mask_arr > 0.5).astype(np.float32)

        if mask_dilation > 0:
            mask_arr = ndimage.binary_dilation(mask_arr > 0.5, iterations=mask_dilation).astype(np.float32)
        
        if mask_erosion > 0:
            mask_arr = ndimage.binary_erosion(mask_arr > 0.5, iterations=mask_erosion).astype(np.float32)
        
        if mask_smoothing > 0.0:
            sm = ndimage.gaussian_filter(mask_arr, sigma=mask_smoothing)
            mask_arr = (sm > 0.5).astype(np.float32)
        
        return mask_arr

    def _is_mask_center(self, mask, min_center_coverage=0.5, max_total_coverage=0.8):
        h, w = mask.shape
        cy, cx = h // 2, w // 2
        center_h, center_w = int(h * 0.5), int(w * 0.5)
        center_mask = mask[cy - center_h//2: cy + center_h//2, cx - center_w//2: cx + center_w//2]
        center_coverage = center_mask.mean()
        total_coverage = mask.mean()
        return center_coverage > min_center_coverage and total_coverage < max_total_coverage

    def clear_cache(self):
        self.cache.clear()
    
    def cleanup_cache(self, unique_id, images):
        if unique_id not in self.cache:
            return
            
        node_cache = self.cache[unique_id]
        current_image_hashes = set()
        for i in range(images.shape[0]):
            img = images[i]
            img_hash = compute_tensor_hash(img)
            current_image_hashes.add(img_hash)
        
        keys_to_remove = []
        for cache_key in list(node_cache.keys()):
            if cache_key.startswith('detection_'):
                remaining = cache_key[10:]
                if len(remaining) >= 32:
                    cached_img_hash = remaining[:32]
                    if cached_img_hash not in current_image_hashes:
                        keys_to_remove.append(cache_key)
        
        for key in keys_to_remove:
            del node_cache[key]

    def __del__(self):
        if hasattr(self, 'yolo_model') and self.yolo_model is not None:
            del self.yolo_model
        if hasattr(self, 'sam_model') and self.sam_model is not None:
            del self.sam_model
        if hasattr(self, 'sam_predictor') and self.sam_predictor is not None:
            del self.sam_predictor
        gc.collect()
