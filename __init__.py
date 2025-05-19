#__init__.py
# Import the mappings from your node file
from .my_image_mask_utils import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Optionally print a confirmation message (helps debugging)
print("--- Loading nodes from comfyui-my-utils ---")

# Make sure __all__ is defined if you want to be explicit (optional but good practice)
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']