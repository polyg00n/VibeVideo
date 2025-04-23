import numpy as np
from typing import Dict, Any, Optional
from __main__ import GlitchEffect
import cv2

class DatamoshAPlugin(GlitchEffect):
    """Creates authentic datamoshing effects by manipulating I-frames and P-frames"""
    
    name = "Datamosh A"
    description = "Creates authentic datamoshing effects by manipulating I-frames and P-frames"
    parameters = {
        "i_frame_interval": {
            "type": int,
            "min": 1,
            "max": 30,
            "default": 10,
            "label": "I-Frame Interval"
        },
        "p_frame_repeat": {
            "type": int,
            "min": 1,
            "max": 10,
            "default": 3,
            "label": "P-Frame Repeat"
        },
        "effect_type": {
            "type": "choice",
            "options": ["i_frame_removal", "p_frame_repeat", "both"],
            "default": "both",
            "label": "Effect Type"
        },
        "blend_mode": {
            "type": "choice",
            "options": [
                "normal",      # Simple weighted average
                "multiply",    # Multiply pixel values
                "screen",      # Inverse of multiply
                "overlay",     # Combination of multiply and screen
                "hard_light",  # Similar to overlay but more intense
                "soft_light",  # Softer version of overlay
                "difference",  # Absolute difference between pixels
                "exclusion",   # Similar to difference but lower contrast
                "add",         # Simple addition
                "subtract"     # Simple subtraction
            ],
            "default": "normal",
            "label": "Blend Mode"
        },
        "blend_contrast": {
            "type": float,
            "min": 0.5,
            "max": 2.0,
            "default": 1.0,
            "label": "Blend Contrast"
        },
        "block_size": {
            "type": int,
            "min": 4,
            "max": 32,
            "default": 16,
            "label": "Block Size"
        },
        "motion_scale": {
            "type": float,
            "min": 0.1,
            "max": 2.0,
            "default": 1.0,
            "label": "Motion Scale"
        },
        "bloom_effect": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "default": 0.3,
            "label": "Bloom Effect"
        }
    }
    
    def __init__(self):
        super().__init__()
        self.author = "VibeVideo"
        self.version = "1.0.0"
        
        # Store reference frames
        self.last_i_frame = None
        self.last_p_frame = None
        self.frame_count = 0
        self.p_frame_repeat_count = 0
        
    def _blend_images(self, img1: np.ndarray, img2: np.ndarray, alpha: float) -> np.ndarray:
        """Apply different blend modes using NumPy operations"""
        # Normalize images to float32 for calculations
        img1 = img1.astype(np.float32) / 255.0
        img2 = img2.astype(np.float32) / 255.0
        
        # Apply contrast adjustment
        contrast = self.params["blend_contrast"]
        img1 = (img1 - 0.5) * contrast + 0.5
        img2 = (img2 - 0.5) * contrast + 0.5
        
        # Clip values to valid range
        img1 = np.clip(img1, 0, 1)
        img2 = np.clip(img2, 0, 1)
        
        blend_mode = self.params["blend_mode"]
        
        if blend_mode == "normal":
            result = img1 * (1 - alpha) + img2 * alpha
        elif blend_mode == "multiply":
            result = img1 * img2
        elif blend_mode == "screen":
            result = 1 - (1 - img1) * (1 - img2)
        elif blend_mode == "overlay":
            mask = img1 > 0.5
            result = np.where(mask,
                            1 - 2 * (1 - img1) * (1 - img2),
                            2 * img1 * img2)
        elif blend_mode == "hard_light":
            mask = img2 > 0.5
            result = np.where(mask,
                            1 - 2 * (1 - img1) * (1 - img2),
                            2 * img1 * img2)
        elif blend_mode == "soft_light":
            result = np.where(img2 <= 0.5,
                            img1 - (1 - 2 * img2) * img1 * (1 - img1),
                            img1 + (2 * img2 - 1) * (np.sqrt(img1) - img1))
        elif blend_mode == "difference":
            result = np.abs(img1 - img2)
        elif blend_mode == "exclusion":
            result = img1 + img2 - 2 * img1 * img2
        elif blend_mode == "add":
            result = img1 + img2
        elif blend_mode == "subtract":
            result = img1 - img2
        
        # Apply alpha blending to the result
        if blend_mode != "normal":
            result = img1 * (1 - alpha) + result * alpha
        
        # Apply bloom effect if enabled
        if self.params["bloom_effect"] > 0:
            # Create a blurred version of the result
            blurred = cv2.GaussianBlur(result, (5, 5), 0)
            # Blend with original using bloom strength
            result = result * (1 - self.params["bloom_effect"]) + blurred * self.params["bloom_effect"]
        
        # Convert back to uint8
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Increment frame counter
        self.frame_count += 1
        
        # Determine frame type (simplified - in real implementation, use proper codec analysis)
        is_i_frame = self.frame_count % self.params["i_frame_interval"] == 0
        is_p_frame = not is_i_frame
        
        if self.params["effect_type"] == "i_frame_removal" or self.params["effect_type"] == "both":
            if is_i_frame:
                # Store the I-frame but don't use it
                self.last_i_frame = frame.copy()
                return frame
            elif is_p_frame and self.last_i_frame is not None:
                # Apply P-frame motion to the wrong I-frame
                # Calculate motion between current frame and last P-frame
                if self.last_p_frame is not None:
                    # Convert to grayscale for motion estimation
                    current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    last_gray = cv2.cvtColor(self.last_p_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate optical flow
                    flow = cv2.calcOpticalFlowFarneback(
                        last_gray, current_gray, None,
                        0.5, 3, 15, 3, 5, 1.2, 0
                    )
                    
                    # Scale motion vectors
                    flow = flow * self.params["motion_scale"]
                    
                    # Start with the current frame instead of zeros
                    result = frame.copy()
                    
                    # Apply motion to the stored I-frame
                    h, w = frame.shape[:2]
                    block_size = self.params["block_size"]
                    
                    # Create a grid of points
                    y, x = np.mgrid[0:h:block_size, 0:w:block_size].reshape(2, -1).astype(int)
                    
                    for i, j in zip(y, x):
                        # Get motion vector
                        dy, dx = flow[i, j]
                        
                        # Calculate block boundaries
                        y_start = i
                        y_end = min(i + block_size, h)
                        x_start = j
                        x_end = min(j + block_size, w)
                        
                        # Calculate new position with bounds checking
                        new_y_start = min(max(y_start + int(dy), 0), h - (y_end - y_start))
                        new_x_start = min(max(x_start + int(dx), 0), w - (x_end - x_start))
                        
                        # Get the actual block size
                        block_height = y_end - y_start
                        block_width = x_end - x_start
                        
                        # Get the source and destination blocks
                        src_block = self.last_i_frame[y_start:y_end, x_start:x_end]
                        dst_block = result[new_y_start:new_y_start+block_height, new_x_start:new_x_start+block_width]
                        
                        # Blend the blocks using the selected blend mode
                        alpha = 0.5  # Blend factor
                        blended = self._blend_images(src_block, dst_block, alpha)
                        result[new_y_start:new_y_start+block_height, new_x_start:new_x_start+block_width] = blended
                    
                    # Store current frame as last P-frame
                    self.last_p_frame = frame.copy()
                    return result
        
        if self.params["effect_type"] == "p_frame_repeat" or self.params["effect_type"] == "both":
            if is_p_frame:
                self.p_frame_repeat_count += 1
                if self.p_frame_repeat_count < self.params["p_frame_repeat"]:
                    # Repeat the last P-frame with blending
                    if self.last_p_frame is not None:
                        # Blend the repeated frame with the current frame
                        alpha = 0.7  # Higher alpha means more of the repeated frame shows through
                        return self._blend_images(frame, self.last_p_frame, alpha)
                else:
                    self.p_frame_repeat_count = 0
                    self.last_p_frame = frame.copy()
        
        # Store current frame as last P-frame
        self.last_p_frame = frame.copy()
        return frame 