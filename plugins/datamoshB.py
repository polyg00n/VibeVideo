"""
Datamosh B - A different approach to datamoshing by manipulating frame data directly
"""
import numpy as np
import cv2
from __main__ import GlitchEffect
from typing import Dict, Any, Optional

class DatamoshBPlugin(GlitchEffect):
    """Creates a different style of datamoshing where moving areas retain structure while static areas get distorted"""
    
    name = "Datamosh B"
    description = "Creates a different style of datamoshing where moving areas retain structure while static areas get distorted"
    
    def __init__(self):
        super().__init__()
        self.author = "VibeVideo"
        self.version = "1.0.0"
        
        # Initialize parameters first
        self.parameters = {
            "motion_threshold": {
                "type": int,
                "min": 1,
                "max": 50,
                "default": 10,
                "label": "Motion Threshold"
            },
            "static_distortion": {
                "type": int,
                "min": 1,
                "max": 20,
                "default": 5,
                "label": "Static Distortion"
            },
            "blend_strength": {
                "type": float,
                "min": 0.1,
                "max": 1.0,
                "default": 0.7,
                "label": "Blend Strength"
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
            }
        }
        
        # Initialize with default parameter values
        self.params = {name: details["default"] for name, details in self.parameters.items()}
        
        # Store reference frames
        self.last_frame = None
        self.motion_mask = None
        self.distorted_frame = None

    def _calculate_motion_mask(self, current_frame: np.ndarray, last_frame: np.ndarray) -> np.ndarray:
        """Calculate a mask of moving areas in the frame"""
        # Convert to grayscale
        current_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        last_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate absolute difference
        diff = cv2.absdiff(current_gray, last_gray)
        
        # Apply threshold to get motion mask
        _, motion_mask = cv2.threshold(diff, self.params["motion_threshold"], 255, cv2.THRESH_BINARY)
        
        # Apply some blur to smooth the mask
        motion_mask = cv2.GaussianBlur(motion_mask, (5, 5), 0)
        
        # Normalize to 0-1 range
        motion_mask = motion_mask.astype(np.float32) / 255.0
        
        return motion_mask
    
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
        
        # Convert back to uint8
        return (np.clip(result, 0, 1) * 255).astype(np.uint8)

    def _distort_static_areas(self, frame: np.ndarray, motion_mask: np.ndarray) -> np.ndarray:
        """Apply distortion to static areas of the frame"""
        # Create a distorted version of the frame
        distorted = frame.copy()
        
        # Get image dimensions
        h, w = frame.shape[:2]
        
        # Create a grid of points
        y, x = np.mgrid[0:h:8, 0:w:8].reshape(2, -1).astype(int)
        
        for i, j in zip(y, x):
            # Calculate block boundaries
            y_start = i
            y_end = min(i + 8, h)
            x_start = j
            x_end = min(j + 8, w)
            
            # Get the average motion in this block
            block_motion = np.mean(motion_mask[y_start:y_end, x_start:x_end])
            
            # Only distort if the block is mostly static
            if block_motion < 0.3:  # Threshold for static areas
                # Random offset for distortion
                offset_y = np.random.randint(-self.params["static_distortion"], self.params["static_distortion"] + 1)
                offset_x = np.random.randint(-self.params["static_distortion"], self.params["static_distortion"] + 1)
                
                # Calculate new position with bounds checking
                new_y_start = min(max(y_start + offset_y, 0), h - (y_end - y_start))
                new_x_start = min(max(x_start + offset_x, 0), w - (x_end - x_start))
                
                # Get the actual block size
                block_height = y_end - y_start
                block_width = x_end - x_start
                
                # Copy and blend the block
                src_block = frame[y_start:y_end, x_start:x_end]
                dst_block = distorted[new_y_start:new_y_start+block_height, new_x_start:new_x_start+block_width]
                
                # Blend based on motion (less motion = more distortion)
                alpha = float(1.0 - block_motion)
                blended = self._blend_images(src_block, dst_block, alpha)
                distorted[new_y_start:new_y_start+block_height, new_x_start:new_x_start+block_width] = blended
        
        return distorted
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.last_frame is None:
            self.last_frame = frame.copy()
            return frame
            
        # Calculate motion mask
        motion_mask = self._calculate_motion_mask(frame, self.last_frame)
        
        # Create distorted version of the frame
        distorted = self._distort_static_areas(frame, motion_mask)
        
        # Blend the original and distorted frames based on motion
        # Moving areas (high motion) get more of the original frame
        # Static areas (low motion) get more of the distorted frame
        motion_mask_3d = np.stack([motion_mask] * 3, axis=2)
        blend_strength = float(self.params["blend_strength"])
        result = self._blend_images(frame, distorted, 1 - blend_strength)
        
        # Store current frame for next iteration
        self.last_frame = frame.copy()
        
        return result 