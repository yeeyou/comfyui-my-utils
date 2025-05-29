# my_image_mask_utils.py

import torch
import numpy as np
from PIL import Image, ImageFilter, ImageDraw # ImageFilter and ImageDraw are used by the new node
import json
# --- Helper Functions for VideoFrameZoomerWithCropWindowMask ---
def tensor_to_pil(tensor_image, batch_index=0):
    if tensor_image.is_cuda:
        tensor_image = tensor_image.cpu()
    if tensor_image.requires_grad:
        tensor_image = tensor_image.detach()
    
    # Assuming tensor is B, H, W, C for IMAGE or B, H, W for MASK
    image_np = tensor_image[batch_index].numpy() 
    image_np = np.clip(image_np, 0.0, 1.0)

    if image_np.ndim == 3 and image_np.shape[-1] in [1, 3, 4]: # H, W, C (IMAGE or MASK with channel)
        if image_np.shape[-1] == 1: # Grayscale with channel dim (could be MASK)
            image_np = image_np.squeeze(axis=-1) # H, W
    # if image_np.ndim == 2: # H, W (MASK) - handled by squeeze above or already 2D
    
    if image_np.dtype == np.float32 or image_np.dtype == np.float64:
        image_np = (image_np * 255).astype(np.uint8)
    
    try:
        if image_np.ndim == 2: # Grayscale, for MASKs
             image_pil = Image.fromarray(image_np, mode='L')
        elif image_np.ndim == 3 and image_np.shape[-1] == 3: # RGB
             image_pil = Image.fromarray(image_np, mode='RGB')
        elif image_np.ndim == 3 and image_np.shape[-1] == 4: # RGBA
             image_pil = Image.fromarray(image_np, mode='RGBA')
        else:
            raise ValueError(f"Unsupported NumPy array shape for PIL conversion: {image_np.shape}")
    except Exception as e:
        print(f"Error creating PIL image: {e}, array shape: {image_np.shape}, dtype: {image_np.dtype}")
        return Image.new("RGB", (64, 64), "magenta") 

    return image_pil


def pil_to_tensor(image_pil):
    image_np = np.array(image_pil).astype(np.float32) / 255.0
    if image_np.ndim == 2: # L mode (masks)
        # ComfyUI MASK is B, H, W
        tensor_image = torch.from_numpy(image_np).unsqueeze(0) 
    elif image_np.ndim == 3: # RGB or RGBA
        # ComfyUI IMAGE is B, H, W, C
        tensor_image = torch.from_numpy(image_np).unsqueeze(0)
    else:
        raise ValueError("Unsupported PIL image for tensor conversion (ndim not 2 or 3).")
    return tensor_image


def get_mask_bbox(mask_pil, threshold=0.5, padding_percent=0.1, original_image_width=None, original_image_height=None):
    if mask_pil.mode != 'L':
        mask_pil = mask_pil.convert('L')
    mask_array = np.array(mask_pil) > (threshold * 255)
    rows = np.any(mask_array, axis=1)
    cols = np.any(mask_array, axis=0)
    if not np.any(rows) or not np.any(cols):
        if original_image_width and original_image_height:
            # print("Warning: get_mask_bbox found no active pixels. Using center quarter.") # Less verbose
            w, h = original_image_width // 2, original_image_height // 2
            x, y = original_image_width // 4, original_image_height // 4
            return float(x), float(y), float(w), float(h)
        return None
    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]
    bbox_w, bbox_h = float(xmax - xmin + 1), float(ymax - ymin + 1)
    pad_w_amount = bbox_w * padding_percent
    pad_h_amount = bbox_h * padding_percent
    padded_x = float(xmin) - pad_w_amount
    padded_y = float(ymin) - pad_h_amount
    padded_w = bbox_w + 2 * pad_w_amount
    padded_h = bbox_h + 2 * pad_h_amount
    return padded_x, padded_y, padded_w, padded_h

def round_to_multiple(number, multiple, min_val=None):
    if multiple == 0: return int(number) 
    rounded = round(number / multiple) * multiple
    if min_val is not None:
        return max(min_val, int(rounded))
    return int(rounded)


# --- Node 1: MaskGetCoords ---
class MaskGetCoords:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("INT", "INT", "INT", "INT")
    RETURN_NAMES = ("top_y", "bottom_y", "left_x", "right_x")
    FUNCTION = "get_coords"
    CATEGORY = "utils/mask" 

    def get_coords(self, mask, threshold):
        # Ensure mask is B,H,W
        if mask.dim() == 4 and mask.shape[-1] == 1: # B,H,W,C (where C=1)
            mask = mask.squeeze(-1) # B,H,W
        elif mask.dim() == 2: # H,W
            mask = mask.unsqueeze(0) # 1,H,W
        
        if mask.dim() != 3:
             raise ValueError(f"MaskGetCoords: Unsupported mask dimension: {mask.shape}. Expected (N, H, W) or (H, W) or (N,H,W,1).")

        # Work with the first mask in the batch
        mask_to_process = mask[0] 
        
        mask_binarized = mask_to_process > threshold
        non_zero_positions = torch.nonzero(mask_binarized)

        if non_zero_positions.shape[0] == 0:
            # print("Warning: MaskGetCoords found no active pixels above threshold.") # Less verbose
            return (0, 0, 0, 0)

        min_y = torch.min(non_zero_positions[:, 0]).item()
        max_y = torch.max(non_zero_positions[:, 0]).item()
        min_x = torch.min(non_zero_positions[:, 1]).item()
        max_x = torch.max(non_zero_positions[:, 1]).item()

        return (int(min_y), int(max_y), int(min_x), int(max_x))

## --- Node 2: OverlayImageAtPosition ---
class OverlayImageAtPosition:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_foreground": ("IMAGE",), 
                "image_background": ("IMAGE",), 
                "position_y": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}),
                "position_x": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}), 
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_combined",)
    FUNCTION = "overlay_image"
    CATEGORY = "utils/image"

    def overlay_image(self, image_foreground, image_background, position_y, position_x):
        if image_foreground.dim() != 4 or image_background.dim() != 4:
            raise ValueError("Both foreground and background images must be 4D tensors (N, H, W, C)")

        fg_n, fg_h, fg_w, fg_c = image_foreground.shape
        bg_n, bg_h, bg_w, bg_c = image_background.shape

        if fg_n == 0 or bg_n == 0:
             raise ValueError("Input tensors cannot be empty")

        fg = image_foreground[0]
        bg = image_background[0].clone() 
        output_device = image_background.device

        if fg_c != 4: # Require RGBA for foreground for its alpha channel
            # If RGB, add full alpha
            if fg_c == 3:
                # print("Warning: OverlayImageAtPosition converting RGB foreground to RGBA with full alpha.") # Less verbose
                alpha_fg = torch.ones((fg_h, fg_w, 1), dtype=fg.dtype, device=fg.device)
                fg = torch.cat((fg, alpha_fg), dim=-1)
                fg_c = 4
            else:
                raise ValueError(f"Foreground image must have 3 (RGB) or 4 (RGBA) channels, but got {fg_c}")


        if bg_c == 3:
            alpha_bg = torch.ones((bg_h, bg_w, 1), dtype=bg.dtype, device=bg.device)
            bg = torch.cat((bg, alpha_bg), dim=-1)
            bg_c = 4
        elif bg_c != 4:
            raise ValueError(f"Background image must have 3 (RGB) or 4 (RGBA) channels, but got {bg_c}")

        target_y_start = position_y
        target_y_end = position_y + fg_h
        target_x_start = position_x
        target_x_end = position_x + fg_w

        source_y_start = 0
        source_y_end = fg_h
        source_x_start = 0
        source_x_end = fg_w

        if target_y_start < 0:
            source_y_start = -target_y_start
            target_y_start = 0
        if target_y_end > bg_h:
            source_y_end = fg_h - (target_y_end - bg_h)
            target_y_end = bg_h

        if target_x_start < 0:
            source_x_start = -target_x_start
            target_x_start = 0
        if target_x_end > bg_w:
            source_x_end = fg_w - (target_x_end - bg_w)
            target_x_end = bg_w

        if target_y_start >= target_y_end or target_x_start >= target_x_end or \
           source_y_start >= source_y_end or source_x_start >= source_x_end or \
           source_y_end <= 0 or source_x_end <= 0:
            # print("Warning: Foreground image placement is entirely outside. Returning background.") # Less verbose
            return bg.unsqueeze(0).to(output_device) 

        fg_slice = fg[source_y_start:source_y_end, source_x_start:source_x_end, :]
        bg_slice = bg[target_y_start:target_y_end, target_x_start:target_x_end, :]

        fg_color = fg_slice[..., :3]
        fg_alpha = fg_slice[..., 3:] 
        bg_color = bg_slice[..., :3]
        bg_alpha = bg_slice[..., 3:] 

        composite_color = fg_color * fg_alpha + bg_color * (1.0 - fg_alpha)
        composite_alpha = fg_alpha + bg_alpha * (1.0 - fg_alpha)
        composite_area = torch.cat((composite_color, composite_alpha), dim=-1)
        bg[target_y_start:target_y_end, target_x_start:target_x_end, :] = composite_area
        result_image = bg.unsqueeze(0).to(output_device)
        return (result_image,)

# --- Node 3: VideoFrameZoomerWithCropWindowMask (NEW NODE) ---
class VideoFrameZoomerWithCropWindowMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mask": ("MASK",), 
                "target_zoom_factor": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "roi_padding_percent": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 2.0, "step": 0.05}),
                "mask_threshold_for_bbox": ("FLOAT", {"default": 0.5, "min": 0.01, "max": 0.99, "step": 0.01}),
                "dilate_sampler_mask_pixels": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
                "feather_crop_window_mask_pixels": ("INT", {"default": 0, "min": 0, "max": 64, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "FLOAT", "INT", "INT", "INT", "INT") 
    RETURN_NAMES = ("ZOOMED_IMAGE", "SAMPLER_INPAINT_MASK", "CROP_WINDOW_ON_ORIGINAL_IMAGE_MASK",
                    "APPLIED_ZOOM_FACTOR", 
                    "ROI_X_ON_ORIGINAL", "ROI_Y_ON_ORIGINAL",
                    "ZOOMED_IMAGE_WIDTH", "ZOOMED_IMAGE_HEIGHT")
    FUNCTION = "process_frame"
    CATEGORY = "utils/image" # Changed category to match others, or use "VideoTools/Detailer"

    def process_frame(self, image: torch.Tensor, mask: torch.Tensor, 
                      target_zoom_factor: float,
                      roi_padding_percent: float, mask_threshold_for_bbox: float,
                      dilate_sampler_mask_pixels: int,
                      feather_crop_window_mask_pixels: int):

        # 支持batch输入，image: (N,H,W,C), mask: (N,H,W) 或 (N,H,W,1)
        if mask.dim() == 4 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)  # (N,H,W)
        if image.dim() == 4 and image.shape[-1] == 1:
            image = image.squeeze(-1)  # (N,H,W) 但一般IMAGE是(N,H,W,C)

        batch_size = image.shape[0]
        # 合并所有mask，得到联合区域
        merged_mask = torch.max(mask, dim=0)[0]  # (H,W)
        merged_mask_pil = tensor_to_pil(merged_mask.unsqueeze(0))  # (1,H,W) -> PIL

        # 取第一帧的尺寸作为原图尺寸
        img_pil = tensor_to_pil(image, batch_index=0)
        orig_img_w, orig_img_h = img_pil.size

        padded_roi_orig_coords = get_mask_bbox(
            merged_mask_pil, mask_threshold_for_bbox, roi_padding_percent, orig_img_w, orig_img_h
        )

        dummy_output_w = round_to_multiple(orig_img_w / 2, 16, 64) 
        dummy_output_h = round_to_multiple(orig_img_h / 2, 16, 64)

        if padded_roi_orig_coords is None:
            dummy_zi = pil_to_tensor(Image.new("RGB", (dummy_output_w, dummy_output_h), "black"))
            dummy_sim = pil_to_tensor(Image.new("L", (dummy_output_w, dummy_output_h), "black"))
            dummy_cwom = pil_to_tensor(Image.new("L", (orig_img_w, orig_img_h), "black"))
            # 输出batch
            return (dummy_zi.repeat(batch_size,1,1,1), dummy_sim.repeat(batch_size,1,1), dummy_cwom, 1.0, 0, 0, dummy_output_w, dummy_output_h)
        
        pr_x, pr_y, pr_w, pr_h = padded_roi_orig_coords 
        if pr_w <=0 or pr_h <=0:
            pr_x, pr_y, pr_w, pr_h = 0.0, 0.0, float(orig_img_w), float(orig_img_h)

        initial_target_w = pr_w * target_zoom_factor
        initial_target_h = pr_h * target_zoom_factor
        zoomed_image_width = round_to_multiple(initial_target_w, 16, min_val=16)
        zoomed_image_height = round_to_multiple(initial_target_h, 16, min_val=16)

        target_aspect = float(zoomed_image_width) / zoomed_image_height if zoomed_image_height > 0 else 1.0
        padded_roi_center_x = pr_x + pr_w / 2.0
        padded_roi_center_y = pr_y + pr_h / 2.0

        if pr_w == 0 or pr_h == 0 : # Avoid division by zero if pr_h is 0
             fc_w = pr_w 
             fc_h = pr_h
        elif pr_w / pr_h > target_aspect:
            fc_w = pr_w
            fc_h = pr_w / target_aspect if target_aspect > 0 else pr_h # Avoid division by zero
        else:
            fc_h = pr_h
            fc_w = pr_h * target_aspect
        
        fc_x = padded_roi_center_x - fc_w / 2.0
        fc_y = padded_roi_center_y - fc_h / 2.0

        # 下面对每一帧都做同样的crop+resize
        zoomed_images = []
        sampler_inpaint_masks = []
        for i in range(batch_size):
            img_pil = tensor_to_pil(image, batch_index=i)
            mask_pil = tensor_to_pil(mask, batch_index=i) if mask.dim()==3 else tensor_to_pil(mask.unsqueeze(-1), batch_index=i)
            img_canvas_unscaled = Image.new("RGB", (int(round(fc_w)), int(round(fc_h))), (0,0,0))
            mask_canvas_unscaled = Image.new("L", (int(round(fc_w)), int(round(fc_h))), 0) 

            src_crop_x = max(0, fc_x)
            src_crop_y = max(0, fc_y)
            src_crop_x_end = min(orig_img_w, fc_x + fc_w)
            src_crop_y_end = min(orig_img_h, fc_y + fc_h)
            dst_paste_x_on_canvas = src_crop_x - fc_x
            dst_paste_y_on_canvas = src_crop_y - fc_y

            if (src_crop_x_end-src_crop_x)>0 and (src_crop_y_end-src_crop_y)>0:
                img_cropped = img_pil.crop((int(round(src_crop_x)), int(round(src_crop_y)), int(round(src_crop_x_end)), int(round(src_crop_y_end))))
                mask_cropped = mask_pil.crop((int(round(src_crop_x)), int(round(src_crop_y)), int(round(src_crop_x_end)), int(round(src_crop_y_end))))
                img_canvas_unscaled.paste(img_cropped, (int(round(dst_paste_x_on_canvas)), int(round(dst_paste_y_on_canvas))))
                mask_canvas_unscaled.paste(mask_cropped, (int(round(dst_paste_x_on_canvas)), int(round(dst_paste_y_on_canvas))))
            zoomed_image_pil = img_canvas_unscaled.resize((zoomed_image_width, zoomed_image_height), Image.LANCZOS)
            sampler_inpaint_mask_pil = mask_canvas_unscaled.resize((zoomed_image_width, zoomed_image_height), Image.NEAREST)
            if dilate_sampler_mask_pixels > 0:
                filter_size = dilate_sampler_mask_pixels * 2 + 1
                if filter_size > 1: 
                    sampler_inpaint_mask_pil = sampler_inpaint_mask_pil.filter(ImageFilter.MaxFilter(size=filter_size))
            zoomed_images.append(pil_to_tensor(zoomed_image_pil))
            sampler_inpaint_masks.append(pil_to_tensor(sampler_inpaint_mask_pil))

        # 合并batch
        zoomed_image_tensor = torch.cat(zoomed_images, dim=0)
        sampler_inpaint_mask_tensor = torch.cat(sampler_inpaint_masks, dim=0)

        # crop window mask 只需一张
        crop_window_mask_pil = Image.new("L", (orig_img_w, orig_img_h), 0)
        draw_crop_window = ImageDraw.Draw(crop_window_mask_pil)
        fc_box_rect_on_orig = (
            int(round(fc_x)), int(round(fc_y)),
            int(round(fc_x + fc_w)), int(round(fc_y + fc_h))
        )
        draw_crop_window.rectangle(fc_box_rect_on_orig, fill=255) 
        if feather_crop_window_mask_pixels > 0:
            radius = feather_crop_window_mask_pixels # Treat as radius for GaussianBlur
            if radius > 0:
                crop_window_mask_pil = crop_window_mask_pil.filter(ImageFilter.GaussianBlur(radius=radius))
        crop_window_mask_tensor = pil_to_tensor(crop_window_mask_pil)

        applied_zoom_factor = float(zoomed_image_width) / fc_w if fc_w > 0 else 1.0
        roi_x_on_original = int(round(fc_x))
        roi_y_on_original = int(round(fc_y))

        return (zoomed_image_tensor, sampler_inpaint_mask_tensor, crop_window_mask_tensor,
                applied_zoom_factor, 
                roi_x_on_original, roi_y_on_original,
                zoomed_image_width, zoomed_image_height)


# --- Node 4: SegsToBBOXesModule (NEW NODE) ---
class SegsToCoordinatesAndBBOXes: # Renamed for clarity
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segs": ("SEGS",), # The top-level tuple: ( (img_dims), [SEG_obj, ...] )
                "index": ("STRING", {"default": "", "multiline": False, "dynamicPrompts": False}), # Comma-separated list of indices, or empty for all
            },
            # "optional": { # Could add a batch flag if you ever process lists of SEGS, but current SEGS is usually for one image
            #     "batch_processing": ("BOOLEAN", {"default": False}),
            # }
        }

    RETURN_TYPES = ("STRING", "BBOX")
    RETURN_NAMES = ("center_coordinates", "bboxes")
    FUNCTION = "extract_data"
    CATEGORY = "ImpactPackUtils" # Or your preferred category

    def extract_data(self, segs, index):
        extracted_bboxes = []
        center_points_data = []

        default_empty_coords_json = json.dumps([{'x': 0, 'y': 0}]) # Default if no valid data

        if segs is None:
            print(f"{self.__class__.__name__}: Input 'segs' is None. Returning default empty data.")
            return (default_empty_coords_json, [])

        if not isinstance(segs, tuple) or len(segs) < 2:
            print(f"{self.__class__.__name__}: Expected 'segs' to be a tuple with at least 2 elements. Got {type(segs)}. Returning default.")
            return (default_empty_coords_json, [])
        
        seg_object_list = segs[1]

        if not isinstance(seg_object_list, list):
            print(f"{self.__class__.__name__}: Expected item at index 1 of 'segs' (seg_object_list) to be a list. Got {type(seg_object_list)}. Returning default.")
            return (default_empty_coords_json, [])
        
        if not seg_object_list:
            print(f"{self.__class__.__name__}: The list of SEG objects is empty. Returning default empty data.")
            return (default_empty_coords_json, [])

        # Process indices
        target_indices = []
        if index.strip(): # If index string is provided and not empty
            try:
                target_indices = [int(i.strip()) for i in index.split(",")]
            except ValueError:
                print(f"{self.__class__.__name__}: Warning - Invalid format in 'index' string: '{index}'. Processing all bboxes instead.")
                target_indices = list(range(len(seg_object_list))) # Fallback to all if index is malformed
        else: # If index string is empty, process all bboxes
            target_indices = list(range(len(seg_object_list)))

        print(f"{self.__class__.__name__}: Target indices to process: {target_indices}")

        for i, seg_object in enumerate(seg_object_list):
            if i not in target_indices: # Only process selected indices
                continue

            if not hasattr(seg_object, 'bbox'):
                print(f"{self.__class__.__name__}: Warning - SEG object at original index {i} does not have 'bbox' attribute. Skipping.")
                continue
            
            bbox_data = seg_object.bbox

            # Validate and convert bbox_data (similar to previous version)
            bbox_coords_list = None
            if isinstance(bbox_data, np.ndarray):
                if bbox_data.ndim == 1 and bbox_data.shape[0] == 4:
                    bbox_coords_list = bbox_data.tolist()
                else:
                    print(f"{self.__class__.__name__}: Warning - 'bbox' for SEG object {i} (numpy) not shape (1,4). Shape: {bbox_data.shape}. Skipping.")
                    continue
            elif isinstance(bbox_data, (list, tuple)):
                if len(bbox_data) == 4:
                    bbox_coords_list = list(bbox_data)
                else:
                    print(f"{self.__class__.__name__}: Warning - 'bbox' for SEG object {i} (list/tuple) not length 4. Len: {len(bbox_data)}. Skipping.")
                    continue
            else:
                print(f"{self.__class__.__name__}: Warning - 'bbox' for SEG object {i} not list/tuple/numpy array. Type: {type(bbox_data)}. Skipping.")
                continue

            valid_coords_format = True
            numeric_bbox_for_calc = []
            for coord_idx, coord_val in enumerate(bbox_coords_list):
                if not isinstance(coord_val, (int, float, np.number)):
                    print(f"{self.__class__.__name__}: Warning - Coord {coord_idx} in bbox of SEG object {i} not number. Val: {coord_val}, Type: {type(coord_val)}. Skipping bbox.")
                    valid_coords_format = False
                    break
                numeric_bbox_for_calc.append(float(coord_val))
            
            if valid_coords_format:
                # bbox is [x_min, y_min, x_max, y_max]
                min_x, min_y, max_x, max_y = numeric_bbox_for_calc
                
                # Calculate center coordinates
                center_x = int((min_x + max_x) / 2)
                center_y = int((min_y + max_y) / 2)
                
                center_points_data.append({"x": center_x, "y": center_y})
                extracted_bboxes.append(numeric_bbox_for_calc) # Use the float version for consistency
            else:
                # If an index was specified but the bbox at that index was invalid,
                # we might want to indicate this more clearly or return a specific error.
                # For now, it's just skipped.
                pass
        
        if not extracted_bboxes: # If loop finished and nothing was extracted (e.g., all specified indices were invalid/out of bounds)
            print(f"{self.__class__.__name__}: No valid bboxes found for the specified indices or overall. Returning default empty data.")
            return (default_empty_coords_json, [])
        
        coordinates_json_string = json.dumps(center_points_data)
        print(f"{self.__class__.__name__}: Extracted {len(extracted_bboxes)} bboxes and {len(center_points_data)} center coordinates.")
        print(f"{self.__class__.__name__}: Center Coordinates JSON: {coordinates_json_string}")
        print(f"{self.__class__.__name__}: BBOXes: {extracted_bboxes}")
            
        return (coordinates_json_string, extracted_bboxes)

# --- Registration: Update mappings ---
NODE_CLASS_MAPPINGS = {
    "Mask Get Bounding Coords": MaskGetCoords,
    "OverlayImageAtPosition": OverlayImageAtPosition,
    "VideoFrameZoomerWithCropWindowMask": VideoFrameZoomerWithCropWindowMask,
    "SegsToCoordinatesAndBBOXes (Impact)": SegsToCoordinatesAndBBOXes # Added new node
}

# --- Display Name Mappings: Update display name ---
NODE_DISPLAY_NAME_MAPPINGS = {
    "Mask Get Bounding Coords": "Get Mask Coords",
    "OverlayImageAtPosition": "Overlay Image at Position",
    "VideoFrameZoomerWithCropWindowMask": "Zoom Frame for Detailer",
    "SegsToCoordinatesAndBBOXes (Impact)": "SEGS to BBOXes and Coordinates  (Impact)" # Added display name
}

# (Keep the print statement at the end if you like)
print("--- My Custom Nodes Loaded: Mask/Image Utils (with Zoomer) ---")