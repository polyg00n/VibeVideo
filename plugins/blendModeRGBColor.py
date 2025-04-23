import numpy as np
import cv2
from __main__ import GlitchEffect
from typing import Dict, Any, Optional

class BlendModeRGBColorPlugin(GlitchEffect):
    """Advanced RGB color manipulation with blend modes"""
    
    name = "RGB Blend Modes"
    description = "Advanced RGB color manipulation with blend modes and channel separation"
    
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
    
    def __init__(self):
        super().__init__()
        self.author = "VibeVideo"
        self.version = "1.0.0"
        self._last_blend_mode = None
        
        # Initialize parameters
        self.parameters = {
            "rgb_blend_mode": {
                "type": int,
                "min": 0,
                "max": len(self.BLEND_MODES) - 1,
                "default": 0,  # Default to normal
                "label": "Blend Mode",
                "description": "Select the blend mode to apply"
            },
            "blend_color_r": {
                "type": int,
                "min": 0,
                "max": 255,
                "default": 128,
                "label": "Blend Color Red"
            },
            "blend_color_g": {
                "type": int,
                "min": 0,
                "max": 255,
                "default": 128,
                "label": "Blend Color Green"
            },
            "blend_color_b": {
                "type": int,
                "min": 0,
                "max": 255,
                "default": 128,
                "label": "Blend Color Blue"
            },
            "blend_alpha": {
                "type": float,
                "min": 0.0,
                "max": 1.0,
                "default": 0.5,
                "label": "Blend Alpha"
            },
            "r_offset": {
                "type": int,
                "min": -150,
                "max": 150,
                "default": 0,
                "label": "Red Channel Offset"
            },
            "g_offset": {
                "type": int,
                "min": -150,
                "max": 150,
                "default": 0,
                "label": "Green Channel Offset"
            },
            "b_offset": {
                "type": int,
                "min": -150,
                "max": 150,
                "default": 0,
                "label": "Blue Channel Offset"
            },
            "channel_scale": {
                "type": float,
                "min": 0.0,
                "max": 3.0,
                "default": 1.0,
                "label": "Channel Scale"
            }
        }
        
        # Initialize with default parameter values
        self.params = {name: details["default"] for name, details in self.parameters.items()}
        print(f"RGB Blend effect initialized with blend mode index: {self.params['rgb_blend_mode']}")
        
        # Store reference frame for temporal effects
        self.last_frame = None

    def set_param(self, name: str, value: Any) -> None:
        """Set a parameter value with type safety"""
        print(f"\n[RGBBlend] set_param called with name={name}, value={value}")
        print(f"[RGBBlend] Current params before change: {self.params}")
        
        if name in self.params:
            expected_type = self.parameters[name]["type"]

            try:
                # Auto-cast value to expected type
                if expected_type == int:
                    value = int(value)
                    # For blend mode, ensure it's within valid range
                    if name == "rgb_blend_mode":
                        value = max(0, min(value, len(self.BLEND_MODES) - 1))
                    self.params[name] = value
                    if name == "rgb_blend_mode":
                        print(f"[RGBBlend] Blend mode index changed to: {value} ({self.BLEND_MODES[value]})")
                        self._last_blend_mode = value
                elif expected_type == float:
                    self.params[name] = float(value)
                elif expected_type == bool:
                    self.params[name] = bool(value)
                elif expected_type == str:
                    self.params[name] = str(value)
                else:
                    self.params[name] = value  # fallback
                    
                print(f"[RGBBlend] Params after change: {self.params}")
            except Exception as e:
                print(f"[WARNING] Failed to cast {name} to {expected_type}: {e}")

    def _create_blend_color(self, height: int, width: int) -> np.ndarray:
        """Create a solid color image for blending"""
        # Create RGB color image
        blend_color = np.zeros((height, width, 3), dtype=np.uint8)
        blend_color[:, :, 0] = self.params["blend_color_b"]  # OpenCV uses BGR
        blend_color[:, :, 1] = self.params["blend_color_g"]
        blend_color[:, :, 2] = self.params["blend_color_r"]
        return blend_color

    def _blend_images(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Apply different blend modes using NumPy operations"""
        # Normalize images to float32 for calculations
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        # Get current blend mode index
        current_blend_mode = self.params["rgb_blend_mode"]
        print(f"\n[RGBBlend] Processing frame with blend mode index: {current_blend_mode} ({self.BLEND_MODES[current_blend_mode]})")
        
        # Apply blend mode based on index
        if current_blend_mode == 0:  # normal
            result = img1 * (1 - self.params["blend_alpha"]) + img2 * self.params["blend_alpha"]
        elif current_blend_mode == 1:  # multiply
            result = img1 * img2
        elif current_blend_mode == 2:  # screen
            result = 1 - (1 - img1) * (1 - img2)
        elif current_blend_mode == 3:  # overlay
            mask = img1 > 0.5
            result = np.where(mask,
                            1 - 2 * (1 - img1) * (1 - img2),
                            2 * img1 * img2)
        elif current_blend_mode == 4:  # hard_light
            mask = img2 > 0.5
            result = np.where(mask,
                            1 - 2 * (1 - img1) * (1 - img2),
                            2 * img1 * img2)
        elif current_blend_mode == 5:  # soft_light
            result = np.where(img2 <= 0.5,
                            img1 - (1 - 2 * img2) * img1 * (1 - img1),
                            img1 + (2 * img2 - 1) * (np.sqrt(img1) - img1))
        elif current_blend_mode == 6:  # difference
            result = np.abs(img1 - img2)
        elif current_blend_mode == 7:  # exclusion
            result = img1 + img2 - 2 * img1 * img2
        elif current_blend_mode == 8:  # add
            result = img1 + img2
        elif current_blend_mode == 9:  # subtract
            result = img1 - img2
        elif current_blend_mode == 10:  # divide
            result = np.where(img2 > 0, img1 / img2, 0)
        elif current_blend_mode == 11:  # color_dodge
            result = np.where(img2 < 1, img1 / (1 - img2), 1)
        elif current_blend_mode == 12:  # color_burn
            result = np.where(img2 > 0, 1 - (1 - img1) / img2, 0)
        elif current_blend_mode == 13:  # vivid_light
            result = np.where(img2 <= 0.5,
                            np.where(img2 > 0, 1 - (1 - img1) / (2 * img2), 0),
                            np.where(img2 < 1, img1 / (2 * (1 - img2)), 1))
        elif current_blend_mode == 14:  # linear_light
            result = img1 + 2 * img2 - 1
        
        # Apply alpha blending if not using normal blend mode
        if current_blend_mode != 0:  # not normal
            result = img1 * (1 - self.params["blend_alpha"]) + result * self.params["blend_alpha"]
        
        # Convert back to uint8
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def _apply_channel_offsets(self, frame: np.ndarray) -> np.ndarray:
        """Apply RGB channel offsets and scaling"""
        # Split into color channels
        b, g, r = cv2.split(frame)
        
        # Apply offsets
        if self.params["r_offset"] != 0:
            r = np.roll(r, self.params["r_offset"], axis=1)
        if self.params["g_offset"] != 0:
            g = np.roll(g, self.params["g_offset"], axis=1)
        if self.params["b_offset"] != 0:
            b = np.roll(b, self.params["b_offset"], axis=1)
        
        # Apply channel scaling
        if self.params["channel_scale"] != 1.0:
            scale = self.params["channel_scale"]
            r = np.clip(r * scale, 0, 255).astype(np.uint8)
            g = np.clip(g * scale, 0, 255).astype(np.uint8)
            b = np.clip(b * scale, 0, 255).astype(np.uint8)
        
        # Merge channels back together
        return cv2.merge([b, g, r])

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Apply channel offsets and scaling
        result = self._apply_channel_offsets(frame)
        
        # Create blend color image
        height, width = frame.shape[:2]
        blend_color = self._create_blend_color(height, width)
        
        # Blend with the custom color
        result = self._blend_images(result, blend_color)
        
        # If we have a last frame, apply temporal blending
        if self.last_frame is not None:
            result = self._blend_images(result, self.last_frame)
        
        # Store current frame for next iteration
        self.last_frame = frame.copy()
        
        return result 