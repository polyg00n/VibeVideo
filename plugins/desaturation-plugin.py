"""
Desaturation plugin with brightness and contrast controls for the Video Glitch Player
"""
import cv2
import numpy as np
from __main__ import GlitchEffect

class DesaturationEffect(GlitchEffect):
    """Desaturate video with brightness and contrast controls"""
    
    name = "Desaturation"
    description = "Convert video to grayscale with adjustable brightness and contrast"
    parameters = {
        "brightness": {"type": float, "min": -100.0, "max": 100.0, "default": 0.0},
        "contrast": {"type": float, "min": 0.0, "max": 3.0, "default": 1.0}
    }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Convert to grayscale while preserving 3 channels
        result = cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
        
        # Convert to float32 for arithmetic operations
        result = result.astype(np.float32)
        
        # Apply contrast adjustment first
        contrast = self.params["contrast"]
        if contrast != 1.0:
            result = cv2.multiply(result, contrast)
        
        # Apply brightness adjustment
        brightness = self.params["brightness"]
        if brightness != 0:
            result = cv2.add(result, brightness)
        
        # Ensure pixel values stay within valid range [0, 255] and convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result 