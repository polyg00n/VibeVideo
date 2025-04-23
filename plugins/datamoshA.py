import numpy as np
from typing import Dict, Any, Optional
from .base_effect import GlitchEffect
import cv2

class DatamoshAPlugin(GlitchEffect):
    """Creates glitch effects by copying P-frames to adjacent frames"""
    
    name = "Datamosh A"
    description = "Creates glitch effects by copying P-frames to adjacent frames"
    parameters = {
        "frame_range": {
            "type": int,
            "min": 1,
            "max": 20,
            "default": 5,
            "label": "Frame Range"
        },
        "direction": {
            "type": "choice",
            "options": ["forward", "backward", "both"],
            "default": "forward",
            "label": "Direction"
        },
        "intensity": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "default": 0.5,
            "label": "Intensity"
        }
    }
    
    def __init__(self):
        super().__init__()
        self.author = "VibeVideo"
        self.version = "1.0.0"
        
        # Store the last P-frame for reference
        self.last_p_frame = None
        self.frame_count = 0

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Increment frame counter
        self.frame_count += 1
        
        # Check if this is a P-frame (simplified detection)
        # In a real implementation, you would use proper video codec analysis
        is_p_frame = self.frame_count % 10 == 0  # Simplified P-frame detection
        
        if is_p_frame and np.random.random() < self.params["intensity"]:
            # Store the P-frame
            self.last_p_frame = frame.copy()
            
            # Apply the effect based on direction
            if self.params["direction"] == "forward":
                return frame  # Original frame, will affect future frames
            elif self.params["direction"] == "backward" and self.last_p_frame is not None:
                return self.last_p_frame  # Use previous P-frame
            elif self.params["direction"] == "both" and self.last_p_frame is not None:
                # Blend current frame with previous P-frame
                return cv2.addWeighted(frame, 0.5, self.last_p_frame, 0.5, 0)
        elif self.last_p_frame is not None and self.frame_count % self.params["frame_range"] == 0:
            # Apply the stored P-frame to regular frames
            if self.params["direction"] == "forward":
                return self.last_p_frame
            elif self.params["direction"] == "backward":
                return frame  # Original frame
            else:  # both
                return cv2.addWeighted(frame, 0.5, self.last_p_frame, 0.5, 0)
        
        return frame 