"""
RGB Blend Modes - Apply various RGB blend modes to the video
"""
import numpy as np
import cv2
from __main__ import GlitchEffect
from typing import Any, Dict, Optional
import logging

class RGBBlendModesEffect(GlitchEffect):
    """Apply various RGB blend modes to the video"""
    
    name = "RGB Blend Modes"
    description = "Apply various RGB blend modes to the video"
    parameters: Dict[str, Any] = {
        "blend_mode": {
            "type": int,
            "min": 0,
            "max": 7,
            "default": 0,
            "description": "Blend mode to apply (0: normal, 1: multiply, 2: screen, 3: overlay, 4: hard_light, 5: soft_light, 6: difference, 7: exclusion)"
        },
        "intensity": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
            "description": "Effect intensity"
        },
        "red_shift": {
            "type": float,
            "min": -1.0,
            "max": 1.0,
            "default": 0.0,
            "description": "Red channel shift"
        },
        "green_shift": {
            "type": float,
            "min": -1.0,
            "max": 1.0,
            "default": 0.0,
            "description": "Green channel shift"
        },
        "blue_shift": {
            "type": float,
            "min": -1.0,
            "max": 1.0,
            "default": 0.0,
            "description": "Blue channel shift"
        }
    }
    
    # Define blend modes as a class constant
    BLEND_MODES = {
        0: "normal",      # Normal blend
        1: "multiply",    # Multiply blend
        2: "screen",      # Screen blend
        3: "overlay",     # Overlay blend
        4: "hard_light",  # Hard light blend
        5: "soft_light",  # Soft light blend
        6: "difference",  # Difference blend
        7: "exclusion"    # Exclusion blend
    }
    
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
    
    def _add_text_feedback(self, frame: np.ndarray, mode_name: str, params: Dict[str, Any]) -> np.ndarray:
        """Add text feedback to the frame showing current mode and parameters"""
        # Create a copy to avoid modifying the original
        result = frame.copy()
        
        # Define text properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        color = (255, 255, 255)  # White text
        bg_color = (0, 0, 0)     # Black background
        
        # Prepare text lines
        lines = [
            f"Mode: {mode_name}",
            f"Intensity: {params['intensity']:.2f}",
            f"Red Shift: {params['red_shift']:.2f}",
            f"Green Shift: {params['green_shift']:.2f}",
            f"Blue Shift: {params['blue_shift']:.2f}"
        ]
        
        # Add text to frame
        y = 30
        for line in lines:
            # Get text size for background
            (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
            
            # Draw background rectangle
            cv2.rectangle(result, (10, y - text_height - 5), 
                         (10 + text_width, y + 5), bg_color, -1)
            
            # Draw text
            cv2.putText(result, line, (10, y), font, font_scale, color, thickness)
            y += 30
        
        return result
    
    def _normal_blend(self, base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Normal blend mode"""
        return blend
    
    def _multiply_blend(self, base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Multiply blend mode"""
        return base * blend / 255
    
    def _screen_blend(self, base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Screen blend mode"""
        return 255 - (255 - base) * (255 - blend) / 255
    
    def _overlay_blend(self, base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Overlay blend mode"""
        return np.where(base < 128,
                       2 * base * blend / 255,
                       255 - 2 * (255 - base) * (255 - blend) / 255)
    
    def _hard_light_blend(self, base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Hard light blend mode"""
        return np.where(blend < 128,
                       2 * base * blend / 255,
                       255 - 2 * (255 - base) * (255 - blend) / 255)
    
    def _soft_light_blend(self, base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Soft light blend mode"""
        return np.where(blend < 128,
                       base * (blend + 128) / 255,
                       255 - (255 - base) * (255 - blend) / 255)
    
    def _difference_blend(self, base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Difference blend mode"""
        return np.abs(base - blend)
    
    def _exclusion_blend(self, base: np.ndarray, blend: np.ndarray) -> np.ndarray:
        """Exclusion blend mode"""
        return base + blend - 2 * base * blend / 255
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame with the selected blend mode"""
        try:
            # Get current blend mode
            mode = self.params["blend_mode"]
            mode_name = self.BLEND_MODES.get(mode, "normal")  # Default to normal if invalid mode
            
            # Create a color-shifted version of the frame
            shifted = frame.copy().astype(np.float32)
            
            # Apply color shifts
            shifted[:, :, 0] = np.clip(shifted[:, :, 0] * (1 + self.params["red_shift"]), 0, 255)
            shifted[:, :, 1] = np.clip(shifted[:, :, 1] * (1 + self.params["green_shift"]), 0, 255)
            shifted[:, :, 2] = np.clip(shifted[:, :, 2] * (1 + self.params["blue_shift"]), 0, 255)
            
            # Convert to uint8
            shifted = shifted.astype(np.uint8)
            
            # Apply selected blend mode
            if mode_name == "normal":
                result = self._normal_blend(frame, shifted)
            elif mode_name == "multiply":
                result = self._multiply_blend(frame, shifted)
            elif mode_name == "screen":
                result = self._screen_blend(frame, shifted)
            elif mode_name == "overlay":
                result = self._overlay_blend(frame, shifted)
            elif mode_name == "hard_light":
                result = self._hard_light_blend(frame, shifted)
            elif mode_name == "soft_light":
                result = self._soft_light_blend(frame, shifted)
            elif mode_name == "difference":
                result = self._difference_blend(frame, shifted)
            elif mode_name == "exclusion":
                result = self._exclusion_blend(frame, shifted)
            else:
                result = frame
            
            # Apply intensity
            intensity = self.params["intensity"]
            result = cv2.addWeighted(frame, 1 - intensity, result, intensity, 0)
            
            # Add text feedback
            result = self._add_text_feedback(result, mode_name, self.params)
            
            return result
        except Exception as e:
            self._logger.error(f"Error processing frame: {e}")
            return frame