"""
Exclusion Blend Mode Effect
Applies an exclusion blend mode with a user-defined RGB color
"""
import numpy as np
import cv2
from __main__ import GlitchEffect
from typing import Any

class ExclusionEffect(GlitchEffect):
    """Applies an exclusion blend mode with a user-defined RGB color"""
    
    name = "Exclusion Blend"
    description = "Applies an exclusion blend mode with a user-defined RGB color"
    
    # Define blend mode mapping
    BLEND_MODES = {
        0: "normal",      # Simple weighted average
        1: "multiply",    # Multiply pixel values
        2: "screen",      # Inverse of multiply
        3: "overlay",     # Combination of multiply and screen
        4: "hard_light",  # Similar to overlay but more intense
        5: "soft_light",  # Softer version of overlay
        6: "difference",  # Absolute difference between pixels
        7: "exclusion",   # Similar to difference but lower contrast
        8: "add",         # Simple addition
        9: "subtract",    # Simple subtraction
        10: "divide",     # Divide pixel values
        11: "color_dodge",# Brighten based on bottom layer
        12: "color_burn", # Darken based on bottom layer
        13: "vivid_light",# Combination of color dodge and burn
        14: "linear_light"# Similar to vivid light but more linear
    }
    
    parameters = {
        "exclusion_blend_mode": {
            "type": int,
            "min": 0,
            "max": len(BLEND_MODES) - 1,
            "default": 7,  # Default to exclusion
            "label": "Blend Mode",
            "description": "Select the blend mode to apply"
        },
        "red": {"type": int, "min": 0, "max": 255, "default": 128},
        "green": {"type": int, "min": 0, "max": 255, "default": 128},
        "blue": {"type": int, "min": 0, "max": 255, "default": 128},
        "blend_alpha": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5}
    }
    
    def __init__(self):
        super().__init__()
        self._last_blend_mode = None
        # Ensure default value is set
        self.params["exclusion_blend_mode"] = self.parameters["exclusion_blend_mode"]["default"]
        print(f"Exclusion effect initialized with blend mode index: {self.params['exclusion_blend_mode']} ({self.BLEND_MODES[self.params['exclusion_blend_mode']]})")
    
    def set_param(self, name: str, value: Any) -> None:
        """Set a parameter value with type safety"""
        print(f"\n[ExclusionEffect] set_param called with name={name}, value={value}")
        print(f"[ExclusionEffect] Current params before change: {self.params}")
        
        if name in self.params:
            expected_type = self.parameters[name]["type"]

            try:
                # Auto-cast value to expected type
                if expected_type == int:
                    value = int(value)
                    # For blend mode, ensure it's within valid range
                    if name == "exclusion_blend_mode":
                        value = max(0, min(value, len(self.BLEND_MODES) - 1))
                    self.params[name] = value
                    if name == "exclusion_blend_mode":
                        print(f"[ExclusionEffect] Blend mode index changed to: {value} ({self.BLEND_MODES[value]})")
                        self._last_blend_mode = value
                elif expected_type == float:
                    self.params[name] = float(value)
                elif expected_type == bool:
                    self.params[name] = bool(value)
                elif expected_type == str:
                    self.params[name] = str(value)
                else:
                    self.params[name] = value  # fallback
                    
                print(f"[ExclusionEffect] Params after change: {self.params}")
            except Exception as e:
                print(f"[WARNING] Failed to cast {name} to {expected_type}: {e}")
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Create a solid color image with the same dimensions as the input frame
        height, width = frame.shape[:2]
        blend_color = np.zeros((height, width, 3), dtype=np.uint8)
        blend_color[:, :, 0] = self.params["blue"]  # OpenCV uses BGR
        blend_color[:, :, 1] = self.params["green"]
        blend_color[:, :, 2] = self.params["red"]
        
        # Convert both images to float32 for calculations
        frame_float = frame.astype(np.float32) / 255.0
        blend_float = blend_color.astype(np.float32) / 255.0
        
        # Get current blend mode index
        current_blend_mode = self.params["exclusion_blend_mode"]
        print(f"\n[ExclusionEffect] Processing frame with blend mode index: {current_blend_mode} ({self.BLEND_MODES[current_blend_mode]})")
        
        # Apply blend mode based on index
        if current_blend_mode == 0:  # normal
            result = frame_float * (1 - self.params["blend_alpha"]) + blend_float * self.params["blend_alpha"]
        elif current_blend_mode == 1:  # multiply
            result = frame_float * blend_float
        elif current_blend_mode == 2:  # screen
            result = 1 - (1 - frame_float) * (1 - blend_float)
        elif current_blend_mode == 3:  # overlay
            mask = frame_float > 0.5
            result = np.where(mask,
                            1 - 2 * (1 - frame_float) * (1 - blend_float),
                            2 * frame_float * blend_float)
        elif current_blend_mode == 4:  # hard_light
            mask = blend_float > 0.5
            result = np.where(mask,
                            1 - 2 * (1 - frame_float) * (1 - blend_float),
                            2 * frame_float * blend_float)
        elif current_blend_mode == 5:  # soft_light
            result = np.where(blend_float <= 0.5,
                            frame_float - (1 - 2 * blend_float) * frame_float * (1 - frame_float),
                            frame_float + (2 * blend_float - 1) * (np.sqrt(frame_float) - frame_float))
        elif current_blend_mode == 6:  # difference
            result = np.abs(frame_float - blend_float)
        elif current_blend_mode == 7:  # exclusion
            result = frame_float + blend_float - 2 * frame_float * blend_float
        elif current_blend_mode == 8:  # add
            result = frame_float + blend_float
        elif current_blend_mode == 9:  # subtract
            result = frame_float - blend_float
        elif current_blend_mode == 10:  # divide
            result = np.where(blend_float > 0, frame_float / blend_float, 0)
        elif current_blend_mode == 11:  # color_dodge
            result = np.where(blend_float < 1, frame_float / (1 - blend_float), 1)
        elif current_blend_mode == 12:  # color_burn
            result = np.where(blend_float > 0, 1 - (1 - frame_float) / blend_float, 0)
        elif current_blend_mode == 13:  # vivid_light
            result = np.where(blend_float <= 0.5,
                            np.where(blend_float > 0, 1 - (1 - frame_float) / (2 * blend_float), 0),
                            np.where(blend_float < 1, frame_float / (2 * (1 - blend_float)), 1))
        elif current_blend_mode == 14:  # linear_light
            result = frame_float + 2 * blend_float - 1
        
        # Apply alpha blending if not using normal blend mode
        if current_blend_mode != 0:  # not normal
            result = frame_float * (1 - self.params["blend_alpha"]) + result * self.params["blend_alpha"]
        
        # Convert back to uint8
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result 