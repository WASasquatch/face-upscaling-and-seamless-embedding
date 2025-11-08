import hashlib
import numpy as np
import torch

from PIL import Image

import comfy
from nodes import common_ksampler, MAX_RESOLUTION

from .fuse_utils import (
    tensor2pil, pil2tensor, compute_tensor_hash, compute_diffusion_hash, compute_blend_hash,
    get_model_identity, compute_conditioning_hash, load_yolo_face_models_list, load_yolo_models_list, 
    load_all_yolo_models_list, load_sam_models_list, build_feather_mask, contract_and_blur_mask, 
    transfer_color, sort_boxes_by_order, apply_edge_blending, FUSEBase, track_faces_in_video, 
    smooth_boxes_temporal,
)

SAMPLING_MAP = {
    "bilinear": Image.BILINEAR or Image.Resampling.BILINEAR,
    "lanczos": Image.LANCZOS or Image.Resampling.LANCZOS,
    "nearest": Image.NEAREST or Image.Resampling.NEAREST,
    "bicubic": Image.BICUBIC or Image.Resampling.BICUBIC,
    "box": Image.BOX or Image.Resampling.BOX,
    "hamming": Image.HAMMING or Image.Resampling.HAMMING
}

BLUR_EXTRA_MARGIN = 8

class FUSEKSampler(FUSEBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Base model for sampling."}),
                "vae": ("VAE", {"tooltip": "VAE model for the sampler."}),
                "images": ("IMAGE", {"tooltip": "Input image batch for face detection and upscaling."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning for the sampler."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning for the sampler."}),
                "use_cache": ("BOOLEAN", {"default": True, "tooltip": "Use internal caching to speed up workflow iteration."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "tooltip": "Seed for the sampler. Used for determinisitc results with some models."}),
                "steps": ("INT", {"default": 20, "min": 1, "tooltip": "Number of steps for the sampler."}),
                "cfg": ("FLOAT", {"default": 8.0, "tooltip": "Classifier-Free Guidance scale for the sampler."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Choose the sampler to use for sampling."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Choose the scheduler to use for sampling."}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoising strength for the sampler. When doing img2img like FUSE, lower values adhere more to the input image (face)."}),
                "yolo_detector": (load_yolo_face_models_list(), {"tooltip": "Choose the YOLO detector to use for face detection. Must be trained on faces. See: https://github.com/akanametov/yolo-face"}),
                "sam_segmenter": (load_sam_models_list(), {"tooltip": "Choose the SAM segmentation model to use for face segmentation. See: https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints"}),
                "sam_model_type": (["vit_b", "vit_l", "vit_h"], {"default": "vit_b", "tooltip": "SAM model type"}),
                "face_id": ("INT", {"default": 0, "min": -1, "tooltip": "Index of the face to process in the image (0 is the first face found, -1 is all faces)."}),
                "face_order": (["linear", "linear_reverse", "largest_bbox", "smallest_bbox"], {"default": "linear", "tooltip": "Order to process detected faces: linear (top-to-bottom, left-to-right), linear_reverse (bottom-to-top, right-to-left), largest_bbox (largest face first), smallest_bbox (smallest face first)."}),
                "face_size": ([512, 768, 1024, 1280, 1536], {"default": 512, "tooltip": "The resolution to sample the face crop at."}),
                "face_padding": ("INT", {"default": 20, "min": 0, "max": MAX_RESOLUTION, "tooltip": "Padding in pixels (int) to pad the face crop with."}),
                "force_square": ("BOOLEAN", {"default": True, "tooltip": "Force 1:1 square face crops"}),
            },
            "optional": {
                "mask_optionals": ("DICT", {"tooltip": "Optional masking and blending settings from FUSESamplerMaskOptions node."}),
                "yolo_optionals": ("DICT", {"tooltip": "Optional YOLO detection settings from FUSEYOLOSettings node."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")

    FUNCTION = "execute"
    CATEGORY = "Face Enhancement"

    def execute(
        self, model, vae, images, 
        positive, negative, use_cache, seed, steps, cfg, sampler_name, scheduler, denoise,
        yolo_detector, sam_segmenter, sam_model_type, face_id, face_size, face_padding,
        force_square, face_order, unique_id,
        mask_optionals=None, yolo_optionals=None
    ):
        device = comfy.model_management.get_torch_device()
        opts = mask_optionals or {}
        mask_fill_holes = bool(opts.get('mask_fill_holes', True))
        mask_dilation = max(int(opts.get('mask_dilation', 0)), 0)
        mask_erosion = max(int(opts.get('mask_erosion', 0)), 0)
        mask_smoothing = max(float(opts.get('mask_smoothing', 0.0)), 0.0)
        sampling_method = SAMPLING_MAP.get(opts.get('blend_sampling', 'bilinear'), Image.BILINEAR)
        blend_amount = float(opts.get('blend_amount', 0.3))
        blend_mode = opts.get('blend_mode', 'box')
        use_sam_mask = bool(opts.get('use_sam_mask', False))
        face_color_transfer = opts.get('color_transfer', 'none')

        self._load_yolo_model(yolo_detector, device, is_face_model=True)
        self._apply_yolo_settings(yolo_optionals)

        if use_sam_mask:
            self._load_sam_model(sam_segmenter, sam_model_type, device)

        if unique_id not in self.cache:
            self.cache[unique_id] = {}

        node_cache = self.cache[unique_id]
        if use_cache:
            self.cleanup_cache(unique_id, images)
            
        N, H, W, C = images.shape
        out_images = images.clone()
        out_masks = torch.zeros((N, H, W), dtype=torch.float32, device=device)

        for i in range(N):
            img = images[i]
            img_np = img.mul(255).byte().cpu().numpy()
            img_hash = compute_tensor_hash(img)
            detection_key = f"detection_{img_hash}_{yolo_detector}"

            if use_cache and detection_key in node_cache:
                detection_cache = node_cache[detection_key]
                boxes_raw = detection_cache['boxes_raw']
                base_pil = detection_cache['base_pil']
            else:
                self._move_yolo_to_device(device)
                results = self.yolo_model(img_np)
                self._move_yolo_to_cpu()
                if len(results) == 0:
                    continue
                boxes_raw = results[0].boxes.xyxy.cpu().numpy().tolist()
                base_pil = tensor2pil(img)
                if use_cache:
                    node_cache[detection_key] = {
                        'boxes_raw': boxes_raw,
                        'base_pil': base_pil
                    }

            boxes = sort_boxes_by_order(boxes_raw, face_order)

            crops_cache_key = f"crops_{detection_key}_{face_size}_{face_padding}_{force_square}_{mask_dilation}_{mask_smoothing}"
            if use_cache and crops_cache_key in node_cache:
                crops = node_cache[crops_cache_key]['crops']
                crop_info = node_cache[crops_cache_key]['crop_info']
            else:
                crops = []
                crop_info = []

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                if (x2 - x1) < 32 or (y2 - y1) < 32:
                    continue

                x1 = max(0, x1 - face_padding)
                y1 = max(0, y1 - face_padding)
                x2 = min(W, x2 + face_padding)
                y2 = min(H, y2 + face_padding)

                if force_square:
                    side = max(x2 - x1, y2 - y1)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    x1 = max(0, cx - side // 2)
                    y1 = max(0, cy - side // 2)
                    x2 = min(W, cx + side // 2)
                    y2 = min(H, cy + side // 2)

                if (x2 - x1) < 32 or (y2 - y1) < 32:
                    continue

                face_crop = base_pil.crop((x1, y1, x2, y2))
                original_size = (x2 - x1, y2 - y1)
                face_resized = face_crop.resize((face_size, face_size), sampling_method)
                crop_tensor = pil2tensor(face_resized).to(device)
                crops.append(crop_tensor)

                crop_info.append({
                    'paste_coords': (x1, y1),
                    'paste_size': original_size,
                    'processing_size': (face_size, face_size),
                    'mask_full': None,
                })

            if use_cache and crops:
                node_cache[crops_cache_key] = {
                    'crops': crops,
                    'crop_info': crop_info
                }

            if not crops:
                continue

            if face_id == -1:
                process_indices = list(range(len(crops)))
            else:
                if face_id >= len(crops):
                    idx = -1
                else:
                    idx = face_id
                process_indices = [idx]

            crops_to_fix = [crops[j] for j in process_indices]
            crop_info_to_use = [crop_info[j] for j in process_indices]
            crops_hash = hashlib.md5(b''.join([c.cpu().numpy().tobytes() for c in crops_to_fix])).hexdigest()

            params_hash = compute_diffusion_hash(seed, steps, cfg, sampler_name, scheduler, denoise)
            model_hash = get_model_identity(model)
            vae_hash = get_model_identity(vae)
            pos_hash = compute_conditioning_hash(positive)
            neg_hash = compute_conditioning_hash(negative)
            diffusion_key = f"diffusion_{crops_hash}_{params_hash}_{model_hash}_{vae_hash}_{pos_hash}_{neg_hash}"

            if use_cache and diffusion_key in node_cache:
                print(f"[FUSE DEBUG] Using cached diffusion result")
                faces_fixed = node_cache[diffusion_key]
            else:
                batch = torch.stack([c.to(device) for c in crops_to_fix], dim=0)
                enc_out = self.vae_encode.encode(vae, batch)
                latents = enc_out[0]["samples"]
                out_latents = common_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, {"samples": latents}, denoise=denoise
                )[0]["samples"]
                faces_fixed = self.vae_decode.decode(vae, {"samples": out_latents})[0]
                if use_cache:
                    print(f"[FUSE DEBUG] Caching diffusion result")
                    node_cache[diffusion_key] = faces_fixed

            blend_hash = compute_blend_hash(
                blend_amount, blend_mode, use_sam_mask, mask_fill_holes,
                mask_dilation, mask_erosion, mask_smoothing
            )
            blend_key = f"blend_{detection_key}_{diffusion_key}_{blend_hash}"

            if use_cache and blend_key in node_cache:
                out_images[i] = node_cache[blend_key]['image']
                out_masks[i] = node_cache[blend_key]['mask']
                continue

            base_pil = node_cache[detection_key]['base_pil'].copy() if use_cache else tensor2pil(img)
            combined_mask = np.zeros((H, W), dtype=np.float32)

            for face_img, info in zip(faces_fixed, crop_info_to_use):
                pil_face = tensor2pil(face_img).resize(info['paste_size'], sampling_method)
                x0, y0 = info['paste_coords']
                w, h = info['paste_size']

                if face_color_transfer != 'none':
                    source_region = base_pil.crop((x0, y0, x0 + w, y0 + h))
                    pil_face = transfer_color(source_region, pil_face, mode=face_color_transfer)

                region = base_pil.crop((x0, y0, x0 + w, y0 + h))

                if use_sam_mask:
                    self._move_sam_to_device(device)
                    face_np = np.array(pil_face)
                    self.sam_predictor.set_image(face_np)
                    margin = 0.1
                    full_box = np.array([[int(w * margin), int(h * margin), int(w * (1 - margin)), int(h * (1 - margin))]])
                    masks, _, _ = self.sam_predictor.predict(box=full_box, multimask_output=False)
                    mask = masks[0].astype(np.float32)
                    if not self._is_mask_center(mask):
                        tight_margin = 0.15
                        x_min = int(w * tight_margin)
                        y_min = int(h * tight_margin)
                        x_max = int(w * (1 - tight_margin))
                        y_max = int(h * (1 - tight_margin))
                        tight_box = np.array([[x_min, y_min, x_max, y_max]])
                        masks, _, _ = self.sam_predictor.predict(box=tight_box, multimask_output=False)
                        mask = masks[0].astype(np.float32)
                    self._move_sam_to_cpu()
                    mask_arr = self._process_mask(mask, mask_dilation, mask_erosion, mask_smoothing, mask_fill_holes)
                    mask_arr = apply_edge_blending(mask_arr, blend_amount)
                    mask_arr = contract_and_blur_mask(mask_arr, blend_amount)
                    mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                    base_pil.paste(pil_face, (x0, y0), mask_pil)
                    combined_mask[y0:y0+h, x0:x0+w] = np.maximum(combined_mask[y0:y0+h, x0:x0+w], mask_arr)
                else:
                    mask_arr = build_feather_mask(w, h, blend_amount, blend_mode)
                    mask_arr = apply_edge_blending(mask_arr, blend_amount)
                    mask_arr = contract_and_blur_mask(mask_arr, blend_amount)
                    mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                    blended = Image.composite(pil_face, region, mask_pil)
                    base_pil.paste(blended, (x0, y0))
                    combined_mask[y0:y0+h, x0:x0+w] = np.maximum(combined_mask[y0:y0+h, x0:x0+w], mask_arr)

            out_image = pil2tensor(base_pil).to(device)
            out_mask = torch.from_numpy(combined_mask).to(device)
            if use_cache:
                node_cache[blend_key] = {'image': out_image, 'mask': out_mask}
            out_images[i] = out_image
            out_masks[i] = out_mask

        return (out_images, out_masks)


class FUSEGenericKSampler(FUSEBase):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Base model for sampling."}),
                "vae": ("VAE", {"tooltip": "VAE model for the sampler."}),
                "images": ("IMAGE", {"tooltip": "Input image batch for mask detection and upscaling."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning for the sampler."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning for the sampler."}),
                "use_cache": ("BOOLEAN", {"default": True, "tooltip": "Use internal caching to speed up workflow iteration."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "tooltip": "Seed for the sampler. Used for determinisitc results with some models."}),
                "steps": ("INT", {"default": 20, "min": 1, "tooltip": "Number of steps for the sampler."}),
                "cfg": ("FLOAT", {"default": 8.0, "tooltip": "Classifier-Free Guidance scale for the sampler."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Choose the sampler to use for sampling."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Choose the scheduler to use for sampling."}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoising strength for the sampler. When doing img2img like FUSE, lower values adhere more to the input image (mask)."}),
                "yolo_detector": (load_yolo_models_list(), {"tooltip": "Choose the YOLO detector to use for mask detection."}),
                "sam_segmenter": (load_sam_models_list(), {"tooltip": "Choose the SAM segmentation model to use for mask segmentation. See: https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints"}),
                "sam_model_type": (["vit_b", "vit_l", "vit_h"], {"default": "vit_b", "tooltip": "SAM model type"}),
                "mask_id": ("INT", {"default": 0, "min": -1, "tooltip": "Index of the mask to process in the image (0 is the first mask found, -1 is all masks)."}),
                "mask_order": (["linear", "linear_reverse", "largest_bbox", "smallest_bbox"], {"default": "linear", "tooltip": "Order to process detected masks: linear (top-to-bottom, left-to-right), linear_reverse (bottom-to-top, right-to-left), largest_bbox (largest mask first), smallest_bbox (smallest mask first)."}),
                "mask_size": ([512, 768, 1024, 1280, 1536], {"default": 512, "tooltip": "The resolution to sample the mask crop at."}),
                "mask_padding": ("INT", {"default": 20, "min": 0, "max": MAX_RESOLUTION, "tooltip": "Padding in pixels (int) to pad the mask crop with."}),
                "force_square": ("BOOLEAN", {"default": True, "tooltip": "Force 1:1 square mask crops"}),
            },
            "optional": {
                "mask_optionals": ("DICT", {"tooltip": "Optional masking and blending settings from FUSESamplerMaskOptions node."}),
                "yolo_optionals": ("DICT", {"tooltip": "Optional YOLO detection settings from FUSEYOLOSettings node."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")

    FUNCTION = "execute"
    CATEGORY = "Generic Enhancement"

    def execute(
        self, model, vae, images, 
        positive, negative, use_cache, seed, steps, cfg, sampler_name, scheduler, denoise,
        yolo_detector, sam_segmenter, sam_model_type, mask_id, mask_order, mask_size, mask_padding,
        force_square, unique_id,
        mask_optionals=None, yolo_optionals=None
    ):
        device = comfy.model_management.get_torch_device()
        opts = mask_optionals or {}
        mask_fill_holes = bool(opts.get('mask_fill_holes', True))
        mask_dilation = max(int(opts.get('mask_dilation', 0)), 0)
        mask_erosion = max(int(opts.get('mask_erosion', 0)), 0)
        mask_smoothing = max(float(opts.get('mask_smoothing', 0.0)), 0.0)
        sampling_method = SAMPLING_MAP.get(opts.get('blend_sampling', 'bilinear'), Image.BILINEAR)
        blend_amount = float(opts.get('blend_amount', 0.3))
        blend_mode = opts.get('blend_mode', 'box')
        use_sam_mask = bool(opts.get('use_sam_mask', False))
        mask_color_transfer = opts.get('color_transfer', 'none')

        self._load_yolo_model(yolo_detector, device)
        self._apply_yolo_settings(yolo_optionals)

        if use_sam_mask:
            self._load_sam_model(sam_segmenter, sam_model_type, device)

        if unique_id not in self.cache:
            self.cache[unique_id] = {}

        node_cache = self.cache[unique_id]
        if use_cache:
            self.cleanup_cache(unique_id, images)
            
        N, H, W, C = images.shape
        out_images = images.clone()
        out_masks = torch.zeros((N, H, W), dtype=torch.float32, device=device)

        for i in range(N):
            img = images[i]
            img_np = img.mul(255).byte().cpu().numpy()
            img_hash = compute_tensor_hash(img)
            detection_key = f"detection_{img_hash}_{yolo_detector}"

            if use_cache and detection_key in node_cache:
                detection_cache = node_cache[detection_key]
                boxes_raw = detection_cache['boxes_raw']
                base_pil = detection_cache['base_pil']
            else:
                self._move_yolo_to_device(device)
                results = self.yolo_model(img_np)
                self._move_yolo_to_cpu()
                if len(results) == 0:
                    continue
                boxes_raw = results[0].boxes.xyxy.cpu().numpy().tolist()
                base_pil = tensor2pil(img)
                if use_cache:
                    node_cache[detection_key] = {
                        'boxes_raw': boxes_raw,
                        'base_pil': base_pil
                    }

            boxes = sort_boxes_by_order(boxes_raw, mask_order)

            crops_cache_key = f"crops_{detection_key}_{mask_size}_{mask_padding}_{force_square}_{mask_dilation}_{mask_smoothing}"
            if use_cache and crops_cache_key in node_cache:
                crops = node_cache[crops_cache_key]['crops']
                crop_info = node_cache[crops_cache_key]['crop_info']
            else:
                crops = []
                crop_info = []

            for box in boxes:
                x1, y1, x2, y2 = map(int, box)

                if (x2 - x1) < 32 or (y2 - y1) < 32:
                    continue

                x1 = max(0, x1 - mask_padding)
                y1 = max(0, y1 - mask_padding)
                x2 = min(W, x2 + mask_padding)
                y2 = min(H, y2 + mask_padding)

                if force_square:
                    side = max(x2 - x1, y2 - y1)
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    x1 = max(0, cx - side // 2)
                    y1 = max(0, cy - side // 2)
                    x2 = min(W, cx + side // 2)
                    y2 = min(H, cy + side // 2)

                if (x2 - x1) < 32 or (y2 - y1) < 32:
                    continue

                mask_crop = base_pil.crop((x1, y1, x2, y2))
                original_size = (x2 - x1, y2 - y1)
                mask_resized = mask_crop.resize((mask_size, mask_size), sampling_method)
                crop_tensor = pil2tensor(mask_resized).to(device)
                crops.append(crop_tensor)

                crop_info.append({
                    'paste_coords': (x1, y1),
                    'paste_size': original_size,
                    'processing_size': (mask_size, mask_size),
                    'mask_full': None,
                })

            if use_cache and crops:
                node_cache[crops_cache_key] = {
                    'crops': crops,
                    'crop_info': crop_info
                    }

            if not crops:
                continue

            if mask_id == -1:
                process_indices = list(range(len(crops)))
            else:
                if mask_id >= len(crops):
                    idx = -1
                else:
                    idx = mask_id
                process_indices = [idx]

            crops_to_fix = [crops[j] for j in process_indices]
            crop_info_to_use = [crop_info[j] for j in process_indices]
            crops_hash = hashlib.md5(b''.join([c.cpu().numpy().tobytes() for c in crops_to_fix])).hexdigest()

            params_hash = compute_diffusion_hash(seed, steps, cfg, sampler_name, scheduler, denoise)
            model_hash = get_model_identity(model)
            vae_hash = get_model_identity(vae)
            pos_hash = compute_conditioning_hash(positive)
            neg_hash = compute_conditioning_hash(negative)
            diffusion_key = f"diffusion_{crops_hash}_{params_hash}_{model_hash}_{vae_hash}_{pos_hash}_{neg_hash}"

            if use_cache and diffusion_key in node_cache:
                masks_fixed = node_cache[diffusion_key]
            else:
                batch = torch.stack([c.to(device) for c in crops_to_fix], dim=0)
                enc_out = self.vae_encode.encode(vae, batch)
                latents = enc_out[0]["samples"]
                out_latents = common_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, {"samples": latents}, denoise=denoise
                )[0]["samples"]
                masks_fixed = self.vae_decode.decode(vae, {"samples": out_latents})[0]
                if use_cache:
                    node_cache[diffusion_key] = masks_fixed

            blend_hash = compute_blend_hash(
                blend_amount, blend_mode, use_sam_mask, mask_fill_holes,
                mask_dilation, mask_erosion, mask_smoothing
            )
            blend_key = f"blend_{detection_key}_{diffusion_key}_{blend_hash}"

            if use_cache and blend_key in node_cache:
                out_images[i] = node_cache[blend_key]['image']
                out_masks[i] = node_cache[blend_key]['mask']
                continue

            base_pil = node_cache[detection_key]['base_pil'].copy() if use_cache else tensor2pil(img)
            combined_mask = np.zeros((H, W), dtype=np.float32)

            for mask_img, info in zip(masks_fixed, crop_info_to_use):
                pil_mask = tensor2pil(mask_img).resize(info['paste_size'], sampling_method)
                x0, y0 = info['paste_coords']
                w, h = info['paste_size']

                if mask_color_transfer != 'none':
                    source_region = base_pil.crop((x0, y0, x0 + w, y0 + h))
                    pil_mask = transfer_color(source_region, pil_mask, mode=mask_color_transfer)

                region = base_pil.crop((x0, y0, x0 + w, y0 + h))

                if use_sam_mask:
                    self._move_sam_to_device(device)
                    mask_np = np.array(pil_mask)
                    self.sam_predictor.set_image(mask_np)
                    margin = 0.1
                    full_box = np.array([[int(w * margin), int(h * margin), int(w * (1 - margin)), int(h * (1 - margin))]])
                    masks, _, _ = self.sam_predictor.predict(box=full_box, multimask_output=False)
                    mask = masks[0].astype(np.float32)
                    if not self._is_mask_center(mask):
                        tight_margin = 0.15
                        x_min = int(w * tight_margin)
                        y_min = int(h * tight_margin)
                        x_max = int(w * (1 - tight_margin))
                        y_max = int(h * (1 - tight_margin))
                        tight_box = np.array([[x_min, y_min, x_max, y_max]])
                        masks, _, _ = self.sam_predictor.predict(box=tight_box, multimask_output=False)
                        mask = masks[0].astype(np.float32)
                    self._move_sam_to_cpu()
                    mask_arr = self._process_mask(mask, mask_dilation, mask_erosion, mask_smoothing, mask_fill_holes)
                    mask_arr = apply_edge_blending(mask_arr, blend_amount)
                    mask_arr = contract_and_blur_mask(mask_arr, blend_amount)
                    mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                    base_pil.paste(pil_mask, (x0, y0), mask_pil)
                    combined_mask[y0:y0+h, x0:x0+w] = np.maximum(combined_mask[y0:y0+h, x0:x0+w], mask_arr)
                else:
                    mask_arr = build_feather_mask(w, h, blend_amount, blend_mode)
                    mask_arr = apply_edge_blending(mask_arr, blend_amount)
                    mask_arr = contract_and_blur_mask(mask_arr, blend_amount)
                    mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                    blended = Image.composite(pil_mask, region, mask_pil)
                    base_pil.paste(blended, (x0, y0))
                    combined_mask[y0:y0+h, x0:x0+w] = np.maximum(combined_mask[y0:y0+h, x0:x0+w], mask_arr)

            out_image = pil2tensor(base_pil).to(device)
            out_mask = torch.from_numpy(combined_mask).to(device)
            if use_cache:
                node_cache[blend_key] = {'image': out_image, 'mask': out_mask}
            out_images[i] = out_image
            out_masks[i] = out_mask

        return (out_images, out_masks)


class FUSESamplerMaskOptions:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "mask_dilation": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, "tooltip": "Dilate (expand) the mask by this many pixels."}),
                "mask_erosion": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 1, "tooltip": "Erode (shrink) the mask by this many pixels."}),
                "mask_smoothing": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1024.0, "step": 0.01, "tooltip": "Gaussian blur radius for mask smoothing."}),
                "mask_fill_holes": ("BOOLEAN", {"default": True, "tooltip": "Fill holes in the detected mask."}),
                "blend_sampling": (["bilinear", "lanczos", "nearest", "bicubic", "box", "hamming"], {"default": "bilinear", "tooltip": "Resampling method for resizing operations (face/mask crop and paste back)."}),
                "blend_amount": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "The amount of blending to apply to the face embedding process."}),
                "blend_mode": (["box", "radial"], {"default": "box", "tooltip": "The feathering mode to use for blending."}),
                "use_sam_mask": ("BOOLEAN", {"default": False, "tooltip": "Use the SAM face mask for blending instead of YOLO bounding box."}),
                "color_transfer": (["none", "lab", "pdf", "mean_std"], {"default": "none", "tooltip": "Color transfer mode to use for maintaining original colors."}),
            }
        }
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("mask_optionals",)
    FUNCTION = "execute"
    CATEGORY = "Face Enhancement"
    
    def execute(self, mask_dilation=0, mask_erosion=0, mask_smoothing=0.0, mask_fill_holes=True, 
                blend_sampling="bilinear", blend_amount=0.3, blend_mode="box", use_sam_mask=False, color_transfer="none"):
        return ({
            "mask_dilation": mask_dilation,
            "mask_erosion": mask_erosion,
            "mask_smoothing": mask_smoothing,
            "mask_fill_holes": mask_fill_holes,
            "blend_sampling": blend_sampling,
            "blend_amount": blend_amount,
            "blend_mode": blend_mode,
            "use_sam_mask": use_sam_mask,
            "color_transfer": color_transfer
        },)


class FUSEYOLOSettings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "confidence": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "YOLO confidence threshold (0-1). Higher values detect only more confident objects."}),
                "iou_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "YOLO NMS IoU threshold (0-1). Lower values remove more overlapping detections."}),
                "max_detections": ("INT", {"default": 300, "min": 1, "max": 1000, "tooltip": "Maximum number of detections to keep after NMS."}),
                "class_filter": ("STRING", {"default": "", "multiline": True, "tooltip": "Optional: Comma or newline separated class names to filter detections. Leave empty to use all classes."}),
                "agnostic_nms": ("BOOLEAN", {"default": False, "tooltip": "Class-agnostic NMS (merge boxes from different classes)."}),
                "half_precision": ("BOOLEAN", {"default": False, "tooltip": "Use half precision (FP16) for faster inference on compatible GPUs."}),
                "augment": ("BOOLEAN", {"default": False, "tooltip": "Test Time Augmentation (TTA) for improved accuracy at cost of speed."}),
                "tracking_iou_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "IoU threshold for face tracking across frames (video only)."}),
                "tracking_smooth_boxes": ("BOOLEAN", {"default": True, "tooltip": "Apply temporal smoothing to face bounding boxes (video only)."}),
                "tracking_smooth_window": ("INT", {"default": 3, "min": 1, "max": 11, "step": 2, "tooltip": "Temporal smoothing window size, must be odd (video only)."}),
            }
        }
    
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("yolo_optionals",)
    FUNCTION = "execute"
    CATEGORY = "Face Enhancement"
    
    def execute(self, confidence=0.25, iou_threshold=0.5, max_detections=300, class_filter="", 
                agnostic_nms=False, half_precision=False, augment=False,
                tracking_iou_threshold=0.3, tracking_smooth_boxes=True, tracking_smooth_window=3):
        filtered_classes = []
        if class_filter.strip():
            classes = [cls.strip() for cls in class_filter.replace('\n', ',').split(',')]
            filtered_classes = [cls for cls in classes if cls]
        
        return ({
            "confidence": confidence,
            "iou_threshold": iou_threshold,
            "max_detections": max_detections,
            "filtered_classes": filtered_classes,
            "agnostic_nms": agnostic_nms,
            "half_precision": half_precision,
            "augment": augment,
            "tracking_iou_threshold": tracking_iou_threshold,
            "tracking_smooth_boxes": tracking_smooth_boxes,
            "tracking_smooth_window": tracking_smooth_window
        },)


class FUSEVideoKSampler(FUSEBase):
    """
    Video-aware face enhancement sampler with temporal tracking.
    Supports both 4D (image batches) and 5D (video) inputs.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Base model for sampling. Should support video if input is 5D."}),
                "vae": ("VAE", {"tooltip": "VAE model for the sampler. Should support video if input is 5D."}),
                "images": ("IMAGE", {"tooltip": "Input image batch (4D) or video (5D) for face detection and upscaling."}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning for the sampler."}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning for the sampler."}),
                "use_cache": ("BOOLEAN", {"default": True, "tooltip": "Use internal caching to speed up workflow iteration."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2**32 - 1, "tooltip": "Seed for the sampler."}),
                "steps": ("INT", {"default": 20, "min": 1, "tooltip": "Number of steps for the sampler."}),
                "cfg": ("FLOAT", {"default": 8.0, "tooltip": "Classifier-Free Guidance scale for the sampler."}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"tooltip": "Choose the sampler to use for sampling."}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"tooltip": "Choose the scheduler to use for sampling."}),
                "denoise": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoising strength for the sampler."}),
                "yolo_detector": (load_all_yolo_models_list(), {"tooltip": "Choose the YOLO detector to use for face detection. Prefix 'face/' for face-specific models, 'yolo/' for general YOLO models."}),
                "sam_segmenter": (load_sam_models_list(), {"tooltip": "Choose the SAM segmentation model to use for face segmentation."}),
                "sam_model_type": (["vit_b", "vit_l", "vit_h"], {"default": "vit_b", "tooltip": "SAM model type"}),
                "face_id": ("INT", {"default": 0, "min": -1, "tooltip": "Index of the face to process (-1 for all faces)."}),
                "face_order": (["linear", "linear_reverse", "largest_bbox", "smallest_bbox"], {"default": "linear", "tooltip": "Order to process detected faces."}),
                "face_size": ([512, 768, 1024, 1280, 1536], {"default": 512, "tooltip": "The resolution to sample the face crop at."}),
                "face_padding": ("INT", {"default": 20, "min": 0, "max": MAX_RESOLUTION, "tooltip": "Padding in pixels to pad the face crop with."}),
                "force_square": ("BOOLEAN", {"default": True, "tooltip": "Force 1:1 square face crops"}),
                "temporal_tracking": ("BOOLEAN", {"default": True, "tooltip": "Enable temporal face tracking for video inputs."}),
            },
            "optional": {
                "mask_optionals": ("DICT", {"tooltip": "Optional masking and blending settings from FUSESamplerMaskOptions node."}),
                "yolo_optionals": ("DICT", {"tooltip": "Optional YOLO detection settings from FUSEYOLOSettings node."}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("images", "masks")
    FUNCTION = "execute"
    CATEGORY = "Face Enhancement/Video"

    def execute(
        self, model, vae, images, 
        positive, negative, use_cache, seed, steps, cfg, sampler_name, scheduler, denoise,
        yolo_detector, sam_segmenter, sam_model_type, face_id, face_size, face_padding,
        force_square, face_order, temporal_tracking,
        unique_id, mask_optionals=None, yolo_optionals=None
    ):
        device = comfy.model_management.get_torch_device()
        
        # Detect if input is video (5D) or image batch (4D)
        is_video = images.ndim == 5
        
        if is_video:
            print(f"[FUSE Video] Processing 5D video input: {images.shape}")
            B, F, H, W, C = images.shape
            # Reshape to process as batch of frames
            images_flat = images.reshape(B * F, H, W, C)
            total_frames = B * F
        else:
            print(f"[FUSE Video] Processing 4D image batch: {images.shape}")
            N, H, W, C = images.shape
            images_flat = images
            total_frames = N
        
        # Get mask options
        opts = mask_optionals or {}
        mask_fill_holes = bool(opts.get('mask_fill_holes', True))
        mask_dilation = max(int(opts.get('mask_dilation', 0)), 0)
        mask_erosion = max(int(opts.get('mask_erosion', 0)), 0)
        mask_smoothing = max(float(opts.get('mask_smoothing', 0.0)), 0.0)
        sampling_method = SAMPLING_MAP.get(opts.get('blend_sampling', 'bilinear'), Image.BILINEAR)
        blend_amount = float(opts.get('blend_amount', 0.3))
        blend_mode = opts.get('blend_mode', 'box')
        use_sam_mask = bool(opts.get('use_sam_mask', False))
        face_color_transfer = opts.get('color_transfer', 'none')
        
        # Get YOLO options with defaults for temporal tracking parameters
        yolo_opts = yolo_optionals or {}
        iou_threshold = float(yolo_opts.get('tracking_iou_threshold', 0.3))  # IoU threshold for face matching
        smooth_boxes = bool(yolo_opts.get('tracking_smooth_boxes', True))    # Apply temporal smoothing
        smooth_window = int(yolo_opts.get('tracking_smooth_window', 3))      # Smoothing window size

        self._load_yolo_model(yolo_detector, device, is_face_model=True)
        self._apply_yolo_settings(yolo_optionals)

        if use_sam_mask:
            self._load_sam_model(sam_segmenter, sam_model_type, device)

        if unique_id not in self.cache:
            self.cache[unique_id] = {}

        node_cache = self.cache[unique_id]
        
        print(f"[FUSE Video] Detecting faces in {total_frames} frames...")
        all_frame_boxes = []
        frame_hashes = []
        
        for i in range(total_frames):
            img = images_flat[i]
            img_np = img.mul(255).byte().cpu().numpy()
            img_hash = compute_tensor_hash(img)
            frame_hashes.append(img_hash)
            detection_key = f"detection_{img_hash}_{yolo_detector}"

            if use_cache and detection_key in node_cache:
                boxes_raw = node_cache[detection_key]['boxes_raw']
            else:
                self._move_yolo_to_device(device)
                results = self.yolo_model(img_np)
                self._move_yolo_to_cpu()
                if len(results) == 0:
                    boxes_raw = []
                else:
                    boxes_raw = results[0].boxes.xyxy.cpu().numpy().tolist()
                
                if use_cache:
                    base_pil = tensor2pil(img)
                    node_cache[detection_key] = {
                        'boxes_raw': boxes_raw,
                        'base_pil': base_pil
                    }
            
            all_frame_boxes.append(boxes_raw)
        
        if is_video and temporal_tracking and len(all_frame_boxes) > 1:
            print(f"[FUSE Video] Tracking faces across frames...")
            tracks = track_faces_in_video(all_frame_boxes, iou_threshold=iou_threshold)
            print(f"[FUSE Video] Found {len(tracks)} face tracks")
            
            if smooth_boxes:
                for track in tracks:
                    track['boxes'] = smooth_boxes_temporal(track['boxes'], window_size=smooth_window)
        else:
            tracks = None
        
        print(f"[FUSE Video] Processing faces...")
        out_images = images_flat.clone()
        out_masks = torch.zeros((total_frames, H, W), dtype=torch.float32, device=device)
        
        if tracks and temporal_tracking:
            # Process each tracked face as separate video batch
            tracks_to_process = [t for t in tracks if face_id == -1 or t['track_id'] == face_id]
            
            if not tracks_to_process:
                print(f"[FUSE Video] No tracks to process, returning original images")
                if is_video:
                    return (images.reshape(B, F, H, W, C), torch.zeros((B, F, H, W), dtype=torch.float32, device=device))
                else:
                    return (images_flat, torch.zeros((total_frames, H, W), dtype=torch.float32, device=device))
            
            for track in tracks_to_process:
                track_id = track['track_id']
                track_frames = track['frames']
                track_boxes = track['boxes']
                
                print(f"[FUSE Video] Processing track {track_id} ({len(track_frames)} frames)")
                
                track_face_crops = []
                track_crop_info = []
                
                for frame_idx, box in zip(track_frames, track_boxes):
                    x1, y1, x2, y2 = map(int, box)
                    
                    x1 = max(0, x1 - face_padding)
                    y1 = max(0, y1 - face_padding)
                    x2 = min(W, x2 + face_padding)
                    y2 = min(H, y2 + face_padding)
                    
                    if force_square:
                        side = max(x2 - x1, y2 - y1)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        x1 = max(0, cx - side // 2)
                        y1 = max(0, cy - side // 2)
                        x2 = min(W, cx + side // 2)
                        y2 = min(H, cy + side // 2)
                    
                    if (x2 - x1) < 32 or (y2 - y1) < 32:
                        continue
                    
                    img_pil = tensor2pil(images_flat[frame_idx])
                    face_crop = img_pil.crop((x1, y1, x2, y2))
                    original_size = (x2 - x1, y2 - y1)
                    face_resized = face_crop.resize((face_size, face_size), sampling_method)
                    
                    track_face_crops.append(pil2tensor(face_resized).to(device))
                    track_crop_info.append({
                        'frame_idx': frame_idx,
                        'paste_coords': (x1, y1),
                        'original_size': original_size,
                    })
                
                if not track_face_crops:
                    print(f"[FUSE Video] Track {track_id}: No valid crops, skipping")
                    continue
                
                face_sequence = torch.stack(track_face_crops, dim=0)
                print(f"[FUSE Video] Track {track_id}: Encoding {len(track_face_crops)} crops to video latent...")
                
                enc_out = self.vae_encode.encode(vae, face_sequence)
                latents = enc_out[0]["samples"]
                
                out_latents = common_ksampler(
                    model, seed, steps, cfg, sampler_name, scheduler,
                    positive, negative, {"samples": latents}, denoise=denoise
                )[0]["samples"]
                
                enhanced_batch = self.vae_decode.decode(vae, {"samples": out_latents})[0]
                print(f"[FUSE Video] Track {track_id}: Pasting enhanced faces back...")
                
                for i, info in enumerate(track_crop_info):
                    frame_idx = info['frame_idx']
                    x0, y0 = info['paste_coords']
                    original_size = info['original_size']
                    
                    enhanced_face = tensor2pil(enhanced_batch[i]).resize(original_size, sampling_method)
                    base_pil = tensor2pil(out_images[frame_idx])
                    w, h = original_size
                    
                    if face_color_transfer != 'none':
                        source_region = base_pil.crop((x0, y0, x0 + w, y0 + h))
                        enhanced_face = transfer_color(source_region, enhanced_face, mode=face_color_transfer)
                    
                    if use_sam_mask:
                        self._move_sam_to_device(device)
                        face_np = np.array(enhanced_face)
                        self.sam_predictor.set_image(face_np)
                        
                        margin = 0.25
                        center_box = np.array([[
                            int(w * margin), 
                            int(h * margin), 
                            int(w * (1 - margin)), 
                            int(h * (1 - margin))
                        ]])
                        
                        masks, scores, _ = self.sam_predictor.predict(
                            box=center_box, 
                            multimask_output=True
                        )
                        
                        best_mask = None
                        best_score = -1
                        for m, s in zip(masks, scores):
                            if self._is_mask_center(m.astype(np.float32)) and s > best_score:
                                best_mask = m
                                best_score = s
                        
                        if best_mask is None:
                            best_mask = masks[0]
                        
                        mask = best_mask.astype(np.float32)
                        self._move_sam_to_cpu()
                        mask_arr = self._process_mask(mask, mask_dilation, mask_erosion, mask_smoothing, mask_fill_holes)
                        mask_arr = apply_edge_blending(mask_arr, blend_amount)
                        mask_arr = contract_and_blur_mask(mask_arr, blend_amount)
                        mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                        base_pil.paste(enhanced_face, (x0, y0), mask_pil)
                    else:
                        mask_arr = build_feather_mask(w, h, blend_amount, blend_mode)
                        mask_arr = apply_edge_blending(mask_arr, blend_amount)
                        mask_arr = contract_and_blur_mask(mask_arr, blend_amount)
                        mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                        region = base_pil.crop((x0, y0, x0 + w, y0 + h))
                        blended = Image.composite(enhanced_face, region, mask_pil)
                        base_pil.paste(blended, (x0, y0))
                    
                    out_images[frame_idx] = pil2tensor(base_pil).to(device)
                    
                    mask_full = np.zeros((H, W), dtype=np.float32)
                    mask_full[y0:y0+h, x0:x0+w] = mask_arr
                    out_masks[frame_idx] = torch.maximum(
                        out_masks[frame_idx],
                        torch.from_numpy(mask_full).to(device)
                    )
        else:
            print(f"[FUSE Video] No tracking - processing all face crops as single batch...")
            all_face_crops = []
            all_crop_info = []
            
            for i in range(total_frames):
                boxes = all_frame_boxes[i]
                if not boxes:
                    continue
                
                boxes_sorted = sort_boxes_by_order(boxes, face_order)
                
                if face_id == -1:
                    process_boxes = boxes_sorted
                else:
                    if face_id < len(boxes_sorted):
                        process_boxes = [boxes_sorted[face_id]]
                    else:
                        continue
                
                for box in process_boxes:
                    x1, y1, x2, y2 = map(int, box)
                    
                    x1 = max(0, x1 - face_padding)
                    y1 = max(0, y1 - face_padding)
                    x2 = min(W, x2 + face_padding)
                    y2 = min(H, y2 + face_padding)
                    
                    if force_square:
                        side = max(x2 - x1, y2 - y1)
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        x1 = max(0, cx - side // 2)
                        y1 = max(0, cy - side // 2)
                        x2 = min(W, cx + side // 2)
                        y2 = min(H, cy + side // 2)
                    
                    if (x2 - x1) < 32 or (y2 - y1) < 32:
                        continue
                    
                    img_pil = tensor2pil(images_flat[i])
                    face_crop = img_pil.crop((x1, y1, x2, y2))
                    original_size = (x2 - x1, y2 - y1)
                    face_resized = face_crop.resize((face_size, face_size), sampling_method)
                    
                    all_face_crops.append(pil2tensor(face_resized).to(device))
                    all_crop_info.append({
                        'frame_idx': i,
                        'paste_coords': (x1, y1),
                        'original_size': original_size,
                    })
            
            if not all_face_crops:
                print(f"[FUSE Video] No face crops found, returning original images")
                if is_video:
                    return (images.reshape(B, F, H, W, C), torch.zeros((B, F, H, W), dtype=torch.float32, device=device))
                else:
                    return (images_flat, torch.zeros((total_frames, H, W), dtype=torch.float32, device=device))
            
            face_sequence = torch.stack(all_face_crops, dim=0)
            print(f"[FUSE Video] Encoding {len(all_face_crops)} crops to video latent...")
            
            enc_out = self.vae_encode.encode(vae, face_sequence)
            latents = enc_out[0]["samples"]
            
            out_latents = common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler,
                positive, negative, {"samples": latents}, denoise=denoise
            )[0]["samples"]
            
            enhanced_batch = self.vae_decode.decode(vae, {"samples": out_latents})[0]
            print(f"[FUSE Video] Pasting enhanced faces back...")
            
            for i, info in enumerate(all_crop_info):
                frame_idx = info['frame_idx']
                x0, y0 = info['paste_coords']
                original_size = info['original_size']
                
                enhanced_face = tensor2pil(enhanced_batch[i]).resize(original_size, sampling_method)
                base_pil = tensor2pil(out_images[frame_idx])
                w, h = original_size
                
                if face_color_transfer != 'none':
                    source_region = base_pil.crop((x0, y0, x0 + w, y0 + h))
                    enhanced_face = transfer_color(source_region, enhanced_face, mode=face_color_transfer)
                
                if use_sam_mask:
                    self._move_sam_to_device(device)
                    face_np = np.array(enhanced_face)
                    self.sam_predictor.set_image(face_np)
                    
                    margin = 0.25
                    center_box = np.array([[
                        int(w * margin), 
                        int(h * margin), 
                        int(w * (1 - margin)), 
                        int(h * (1 - margin))
                    ]])
                    
                    masks, scores, _ = self.sam_predictor.predict(
                        box=center_box, 
                        multimask_output=True
                    )
                    
                    best_mask = None
                    best_score = -1
                    for m, s in zip(masks, scores):
                        if self._is_mask_center(m.astype(np.float32)) and s > best_score:
                            best_mask = m
                            best_score = s
                    
                    if best_mask is None:
                        best_mask = masks[0]
                    
                    mask = best_mask.astype(np.float32)
                    self._move_sam_to_cpu()
                    mask_arr = self._process_mask(mask, mask_dilation, mask_erosion, mask_smoothing, mask_fill_holes)
                    mask_arr = apply_edge_blending(mask_arr, blend_amount)
                    mask_arr = contract_and_blur_mask(mask_arr, blend_amount)
                    mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                    base_pil.paste(enhanced_face, (x0, y0), mask_pil)
                else:
                    mask_arr = build_feather_mask(w, h, blend_amount, blend_mode)
                    mask_arr = apply_edge_blending(mask_arr, blend_amount)
                    mask_arr = contract_and_blur_mask(mask_arr, blend_amount)
                    mask_pil = Image.fromarray((mask_arr * 255).astype(np.uint8), mode="L")
                    region = base_pil.crop((x0, y0, x0 + w, y0 + h))
                    blended = Image.composite(enhanced_face, region, mask_pil)
                    base_pil.paste(blended, (x0, y0))
                
                out_images[frame_idx] = pil2tensor(base_pil).to(device)
                
                mask_full = np.zeros((H, W), dtype=np.float32)
                mask_full[y0:y0+h, x0:x0+w] = mask_arr
                out_masks[frame_idx] = torch.maximum(
                    out_masks[frame_idx],
                    torch.from_numpy(mask_full).to(device)
                )
        
        if is_video:
            out_images = out_images.reshape(B, F, H, W, C)
            out_masks = out_masks.reshape(B, F, H, W)
        
        print(f"[FUSE Video] Processing complete!")
        return (out_images, out_masks)


NODE_CLASS_MAPPINGS = {
    "FUSEKSampler": FUSEKSampler,
    "FUSEGenericKSampler": FUSEGenericKSampler,
    "FUSESamplerMaskOptions": FUSESamplerMaskOptions,
    "FUSEYOLOSettings": FUSEYOLOSettings,
    "FUSEVideoKSampler": FUSEVideoKSampler,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FUSEKSampler": "FUSE KSampler",
    "FUSEGenericKSampler": "FUSE KSampler (Generic)",
    "FUSESamplerMaskOptions": "FUSE Mask Optionals",
    "FUSEYOLOSettings": "FUSE YOLO Optionals",
    "FUSEVideoKSampler": "FUSE KSampler (Video)",
}
