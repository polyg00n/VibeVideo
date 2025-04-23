import numpy as np
from typing import Dict, Any, Optional
from __main__ import GlitchEffect
import cv2

class DatamoshCPlugin(GlitchEffect):
    """Enhanced version of Datamosh B with color channel separation and motion trails"""
    
    name = "Datamosh C"
    description = "Enhanced version with color channel separation and motion trails"
    
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
            "color_offset": {
                "type": int,
                "min": 0,
                "max": 20,
                "default": 5,
                "label": "Color Offset"
            },
            "trail_length": {
                "type": int,
                "min": 0,
                "max": 10,
                "default": 3,
                "label": "Trail Length"
            },
            "trail_strength": {
                "type": float,
                "min": 0.0,
                "max": 1.0,
                "default": 0.3,
                "label": "Trail Strength"
            }
        }
        
        # Initialize with default parameter values
        self.params = {name: details["default"] for name, details in self.parameters.items()}
        
        # Store reference frames and motion history
        self.last_frame = None
        self.motion_mask = None
        self.distorted_frame = None
        self.motion_history = []
        self.max_history = 10  # Maximum number of frames to keep in history
        
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
    
    def _apply_color_separation(self, frame: np.ndarray, motion_mask: np.ndarray) -> np.ndarray:
        """Apply color channel separation based on motion"""
        # Split into color channels
        b, g, r = cv2.split(frame)
        
        # Calculate offsets based on motion
        offset = self.params["color_offset"]
        h, w = frame.shape[:2]
        
        # Create offset versions of each channel
        b_offset = np.roll(b, offset, axis=1)
        g_offset = np.roll(g, -offset, axis=1)
        r_offset = np.roll(r, offset//2, axis=0)
        
        # Blend channels based on motion mask
        b = (b * (1 - motion_mask) + b_offset * motion_mask).astype(np.uint8)
        g = (g * (1 - motion_mask) + g_offset * motion_mask).astype(np.uint8)
        r = (r * (1 - motion_mask) + r_offset * motion_mask).astype(np.uint8)
        
        # Merge channels back together
        return cv2.merge([b, g, r])
    
    def _apply_motion_trail(self, frame: np.ndarray, motion_mask: np.ndarray) -> np.ndarray:
        """Apply motion trail effect"""
        if not self.motion_history:
            return frame
            
        result = frame.copy()
        trail_strength = self.params["trail_strength"]
        
        # Blend previous frames based on their age
        for i, (hist_frame, hist_mask) in enumerate(self.motion_history):
            # Calculate weight based on age (older frames have less influence)
            weight = trail_strength * (1.0 - (i / len(self.motion_history)))
            
            # Only apply trail to areas that were moving
            mask = hist_mask * weight
            
            # Create a 3D mask for each color channel
            mask_3d = np.stack([mask] * 3, axis=2)
            
            # Create a temporary result for this blend
            temp_result = result.copy()
            
            # Blend with current frame using the mask
            result = np.where(mask_3d > 0,
                            (temp_result * (1 - mask_3d) + hist_frame * mask_3d).astype(np.uint8),
                            temp_result)
        
        return result
    
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
                alpha = 1.0 - block_motion
                distorted[new_y_start:new_y_start+block_height, new_x_start:new_x_start+block_width] = \
                    cv2.addWeighted(dst_block, 1 - alpha, src_block, alpha, 0)
        
        return distorted
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        if self.last_frame is None:
            self.last_frame = frame.copy()
            return frame
            
        # Calculate motion mask
        motion_mask = self._calculate_motion_mask(frame, self.last_frame)
        
        # Update motion history
        self.motion_history.insert(0, (frame.copy(), motion_mask.copy()))
        if len(self.motion_history) > self.params["trail_length"]:
            self.motion_history.pop()
        
        # Apply color separation
        frame_with_color = self._apply_color_separation(frame, motion_mask)
        
        # Create distorted version of the frame
        distorted = self._distort_static_areas(frame_with_color, motion_mask)
        
        # Apply motion trail
        result = self._apply_motion_trail(distorted, motion_mask)
        
        # Final blend
        result = cv2.addWeighted(
            frame_with_color, 
            self.params["blend_strength"],
            result,
            1.0 - self.params["blend_strength"],
            0
        )
        
        # Store current frame for next iteration
        self.last_frame = frame.copy()
        
        return result 