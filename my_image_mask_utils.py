# my_image_mask_utils.py

import torch
import numpy as np # Import numpy if either node uses it

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
    CATEGORY = "utils/mask" # You can choose your own category

    def get_coords(self, mask, threshold):
        if mask.dim() != 3:
             print(f"Warning: MaskGetCoords expected mask shape (N, H, W), got {mask.shape}. Proceeding with first slice.")
             if mask.dim() == 2:
                 mask = mask.unsqueeze(0)
             elif mask.dim() != 3:
                  raise ValueError(f"Unsupported mask dimension: {mask.dim()}. Expected (N, H, W) or (H, W).")

        n, h, w = mask.shape
        mask_binarized = mask[0] > threshold
        non_zero_positions = torch.nonzero(mask_binarized)

        if non_zero_positions.shape[0] == 0:
            print("Warning: MaskGetCoords found no active pixels above threshold.")
            return (0, 0, 0, 0)

        min_y = torch.min(non_zero_positions[:, 0]).item()
        max_y = torch.max(non_zero_positions[:, 0]).item()
        min_x = torch.min(non_zero_positions[:, 1]).item()
        max_x = torch.max(non_zero_positions[:, 1]).item()

        return (int(min_y), int(max_y), int(min_x), int(max_x))

## --- Node 2: OverlayImageAtPosition (Modified from OverlayImageAtY) ---
class OverlayImageAtPosition: # Renamed class
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_foreground": ("IMAGE",), # Expecting RGBA (N, H, W, 4)
                "image_background": ("IMAGE",), # Can be RGB or RGBA (N, H, W, 3 or 4)
                "position_y": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}), # Increased range just in case
                "position_x": ("INT", {"default": 0, "min": -99999, "max": 99999, "step": 1}), # Added X position input
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_combined",)
    FUNCTION = "overlay_image"
    CATEGORY = "utils/image" # Or keep image/compositing

    def overlay_image(self, image_foreground, image_background, position_y, position_x): # Added position_x parameter
        # --- Input Validation and Preparation ---
        if image_foreground.dim() != 4 or image_background.dim() != 4:
            raise ValueError("Both foreground and background images must be 4D tensors (N, H, W, C)")

        fg_n, fg_h, fg_w, fg_c = image_foreground.shape
        bg_n, bg_h, bg_w, bg_c = image_background.shape

        if fg_n == 0 or bg_n == 0:
             raise ValueError("Input tensors cannot be empty")

        # Work on the first image in the batch
        fg = image_foreground[0]
        bg = image_background[0].clone() # Clone to avoid modifying original bg
        output_device = image_background.device

        # Ensure foreground is RGBA
        if fg_c != 4:
            raise ValueError(f"Foreground image must have 4 channels (RGBA), but got {fg_c}")

        # Ensure background is RGBA for compositing
        if bg_c == 3:
            alpha = torch.ones((bg_h, bg_w, 1), dtype=bg.dtype, device=bg.device)
            bg = torch.cat((bg, alpha), dim=-1)
            bg_c = 4
        elif bg_c != 4:
            raise ValueError(f"Background image must have 3 (RGB) or 4 (RGBA) channels, but got {bg_c}")

        # --- Calculate Placement and Clipping ---
        # Define the target area on the background based on inputs
        target_y_start = position_y
        target_y_end = position_y + fg_h
        target_x_start = position_x # Use position_x for the left edge
        target_x_end = position_x + fg_w # Calculate right edge based on position_x

        # Define the source area from the foreground (initially the whole image)
        source_y_start = 0
        source_y_end = fg_h
        source_x_start = 0
        source_x_end = fg_w

        # --- Clip coordinates against background boundaries ---

        # Clip Y (Vertical) - Adjust source and target if necessary
        if target_y_start < 0:
            source_y_start = -target_y_start # Start reading fg later
            target_y_start = 0             # Start writing bg at the top
        if target_y_end > bg_h:
            source_y_end = fg_h - (target_y_end - bg_h) # Read less of fg
            target_y_end = bg_h                        # Write up to bg bottom

        # Clip X (Horizontal) - Adjust source and target if necessary
        if target_x_start < 0:
            source_x_start = -target_x_start # Start reading fg later (from the right)
            target_x_start = 0             # Start writing bg at the left edge
        if target_x_end > bg_w:
            source_x_end = fg_w - (target_x_end - bg_w) # Read less of fg (from the left)
            target_x_end = bg_w                        # Write up to bg right edge


        # --- Check if there is any overlap left after clipping ---
        if target_y_start >= target_y_end or target_x_start >= target_x_end or \
           source_y_start >= source_y_end or source_x_start >= source_x_end or \
           source_y_end <= 0 or source_x_end <= 0: # Added check for negative source end indices
            print("Warning: Foreground image placement is entirely outside the background boundaries. Returning original background.")
            return bg.unsqueeze(0).to(output_device) # Return original (potentially RGBA converted) background

        # --- Perform Alpha Compositing ---

        # Extract the relevant slices based on calculated source/target ranges
        fg_slice = fg[source_y_start:source_y_end, source_x_start:source_x_end, :]
        bg_slice = bg[target_y_start:target_y_end, target_x_start:target_x_end, :]

        # Separate color and alpha channels
        fg_color = fg_slice[..., :3]
        fg_alpha = fg_slice[..., 3:] # Keep the last dimension (H_clip, W_clip, 1)

        bg_color = bg_slice[..., :3]
        bg_alpha = bg_slice[..., 3:] # Background alpha

        # Calculate composite color: C = Cf*Af + Cb*(1-Af)
        composite_color = fg_color * fg_alpha + bg_color * (1.0 - fg_alpha)

        # Calculate composite alpha: A = Af + Ab*(1-Af)
        composite_alpha = fg_alpha + bg_alpha * (1.0 - fg_alpha)

        # Combine composite color and alpha
        composite_area = torch.cat((composite_color, composite_alpha), dim=-1)

        # Place the composited area back onto the background clone
        bg[target_y_start:target_y_end, target_x_start:target_x_end, :] = composite_area

        # --- Return Result ---
        # Add the batch dimension back and ensure it's on the right device
        result_image = bg.unsqueeze(0).to(output_device)
        return (result_image,)


# --- Registration: Update mappings ---
NODE_CLASS_MAPPINGS = {
    "Mask Get Bounding Coords": MaskGetCoords,
    "OverlayImageAtPosition": OverlayImageAtPosition # Updated class name
}

# --- Display Name Mappings: Update display name ---
NODE_DISPLAY_NAME_MAPPINGS = {
    "Mask Get Bounding Coords": "Get Mask Coords",
    "OverlayImageAtPosition": "Overlay Image at Position" # Updated display name
}

# (Keep the print statement at the end if you like)
print("--- My Custom Nodes Loaded: Mask/Image Utils ---")