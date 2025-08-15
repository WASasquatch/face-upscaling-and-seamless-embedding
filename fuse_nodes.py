import hashlib
import numpy as np
import torch

from PIL import Image

import comfy
from nodes import common_ksampler, MAX_RESOLUTION

from .fuse_utils import (
    tensor2pil, pil2tensor, compute_tensor_hash, compute_diffusion_hash, compute_blend_hash,
    get_model_identity, compute_conditioning_hash, load_yolo_face_models_list, load_yolo_models_list, 
    load_sam_models_list, build_feather_mask, contract_and_blur_mask, transfer_color, sort_boxes_by_order, 
    apply_edge_blending, FUSEBase,
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
            }
        }
    
    RETURN_TYPES = ("DICT",)
    RETURN_NAMES = ("yolo_optionals",)
    FUNCTION = "execute"
    CATEGORY = "Face Enhancement"
    
    def execute(self, confidence=0.25, iou_threshold=0.5, max_detections=300, class_filter="", 
                agnostic_nms=False, half_precision=False, augment=False):
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
            "augment": augment
        },)


NODE_CLASS_MAPPINGS = {
    "FUSEKSampler": FUSEKSampler,
    "FUSEGenericKSampler": FUSEGenericKSampler,
    "FUSESamplerMaskOptions": FUSESamplerMaskOptions,
    "FUSEYOLOSettings": FUSEYOLOSettings,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FUSEKSampler": "FUSE KSampler",
    "FUSEGenericKSampler": "FUSE KSampler (Generic)",
    "FUSESamplerMaskOptions": "FUSE Mask Optionals",
    "FUSEYOLOSettings": "FUSE YOLO Optionals"
}
