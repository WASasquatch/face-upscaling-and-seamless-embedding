import hashlib
import numpy as np
import os
import torch
import gc
from python_color_transfer.color_transfer import ColorTransfer

from PIL import Image, ImageFilter
from scipy import ndimage
from ultralytics import YOLO
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
    return str(id(model))

def compute_conditioning_hash(cond):
    try:
        return hashlib.md5(str(cond).encode()).hexdigest()
    except Exception:
        return str(id(cond)) 

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

    def _load_yolo_model(self, model_name, device, is_face_model=False):
        if self.yolo_model is None or self.current_yolo_model_name != model_name:
            if self.yolo_model is not None:
                self._move_yolo_to_cpu()
                del self.yolo_model
                gc.collect()

            model_path = yolo_face_models_path if is_face_model else yolo_models_path
            weights = os.path.join(model_path, model_name)
            self.yolo_model = YOLO(weights)
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

    def _load_sam_model(self, model_name, model_type, device):
        if (self.sam_model is None or
            self.current_sam_model_name != model_name or
            self.current_sam_model_type != model_type):

            if self.sam_model is not None:
                self._move_sam_to_cpu()
                del self.sam_model
                del self.sam_predictor
                gc.collect()

            checkpoint = os.path.join(sam_models_path, model_name)
            self.sam_model = sam_model_registry[model_type](checkpoint=checkpoint)
            self.sam_predictor = SamPredictor(self.sam_model)
            self.current_sam_model_name = model_name
            self.current_sam_model_type = model_type
            
        if self.sam_model.device != device:
            self.sam_model.to(device)

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

    def __del__(self):
        if hasattr(self, 'yolo_model') and self.yolo_model is not None:
            del self.yolo_model
        if hasattr(self, 'sam_model') and self.sam_model is not None:
            del self.sam_model
        if hasattr(self, 'sam_predictor') and self.sam_predictor is not None:
            del self.sam_predictor
        gc.collect()
