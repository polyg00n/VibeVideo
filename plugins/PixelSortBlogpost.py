"""
Pixel Sort Effect - Sort pixels based on various criteria
Based on the ASDF Pixel Sort algorithm by Kim Asendorf
"""
import numpy as np
import cv2
from base_effect import GlitchEffect
from typing import Any, Tuple, Optional
import logging

class PixelSortEffect(GlitchEffect):
    """Sort pixels based on various criteria like brightness, black/white thresholds"""
    
    name = "Pixel Sort"
    description = "Sort pixels based on brightness, black/white thresholds"
    parameters = {
        "mode": {
            "type": int,
            "min": 0,
            "max": 2,
            "default": 1,
            "description": "0=black, 1=brightness, 2=white"
        },
        "black_threshold": {
            "type": int,
            "min": -255,
            "max": 0,
            "default": -160,
            "description": "Threshold for black mode"
        },
        "brightness_threshold": {
            "type": int,
            "min": 0,
            "max": 255,
            "default": 60,
            "description": "Threshold for brightness mode"
        },
        "white_threshold": {
            "type": int,
            "min": 0,
            "max": 255,
            "default": 130,
            "description": "Threshold for white mode"
        },
        "sort_direction": {
            "type": int,
            "min": 0,
            "max": 1,
            "default": 0,
            "description": "0=horizontal, 1=vertical"
        },
        "intensity": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "default": 1.0,
            "description": "Effect intensity"
        }
    }
    
    def __init__(self):
        super().__init__()
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
    
    def _get_pixel_value(self, pixel: np.ndarray) -> float:
        """Get pixel value based on current mode"""
        mode = self.params["mode"]
        
        if mode == 0:  # Black mode
            return np.mean(pixel)  # Average of RGB channels
        elif mode == 1:  # Brightness mode
            return np.mean(pixel)  # Average of RGB channels
        else:  # White mode
            return np.mean(pixel)  # Average of RGB channels
    
    def _should_start_sort(self, pixel: np.ndarray) -> bool:
        """Determine if sorting should start based on current mode"""
        mode = self.params["mode"]
        value = self._get_pixel_value(pixel)
        
        if mode == 0:  # Black mode
            return value > self.params["black_threshold"]
        elif mode == 1:  # Brightness mode
            return value > self.params["brightness_threshold"]
        else:  # White mode
            return value < self.params["white_threshold"]
    
    def _should_stop_sort(self, pixel: np.ndarray) -> bool:
        """Determine if sorting should stop based on current mode"""
        mode = self.params["mode"]
        value = self._get_pixel_value(pixel)
        
        if mode == 0:  # Black mode
            return value <= self.params["black_threshold"]
        elif mode == 1:  # Brightness mode
            return value <= self.params["brightness_threshold"]
        else:  # White mode
            return value >= self.params["white_threshold"]
    
    def _sort_segment(self, segment: np.ndarray) -> np.ndarray:
        """Sort a segment of pixels based on their values"""
        # Get values for each pixel
        values = np.array([self._get_pixel_value(p) for p in segment])
        
        # Get sorted indices
        sorted_indices = np.argsort(values)
        
        # Sort pixels
        return segment[sorted_indices]
    
    def _sort_line(self, line: np.ndarray) -> np.ndarray:
        """Sort a line of pixels"""
        result = line.copy()
        i = 0
        
        while i < len(line):
            # Find start of segment
            while i < len(line) and not self._should_start_sort(line[i]):
                i += 1
            
            if i >= len(line):
                break
            
            start = i
            
            # Find end of segment
            while i < len(line) and not self._should_stop_sort(line[i]):
                i += 1
            
            end = i
            
            # Sort segment
            if end > start:
                result[start:end] = self._sort_segment(line[start:end])
            
            i += 1
        
        return result
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame with pixel sorting"""
        if not isinstance(frame, np.ndarray):
            self._logger.error("Input frame must be a numpy array")
            return frame
            
        if frame.dtype != np.uint8:
            self._logger.error("Input frame must be uint8 type")
            return frame
            
        if len(frame.shape) not in (2, 3):
            self._logger.error("Input frame must be 2D (grayscale) or 3D (color)")
            return frame
        
        try:
            result = frame.copy()
            height, width = frame.shape[:2]
            
            # Convert to grayscale for thresholding if needed
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Sort horizontally or vertically based on direction
            if self.params["sort_direction"] == 0:  # Horizontal
                for y in range(height):
                    if len(frame.shape) == 3:  # Color
                        result[y] = self._sort_line(frame[y])
                    else:  # Grayscale
                        result[y] = self._sort_line(frame[y])
            else:  # Vertical
                for x in range(width):
                    if len(frame.shape) == 3:  # Color
                        result[:, x] = self._sort_line(frame[:, x])
                    else:  # Grayscale
                        result[:, x] = self._sort_line(frame[:, x])
            
            # Apply intensity
            intensity = self.params["intensity"]
            if intensity < 1.0:
                result = cv2.addWeighted(frame, 1 - intensity, result, intensity, 0)
            
            return result
            
        except Exception as e:
            self._logger.error(f"Error processing frame: {e}")
            return frame 