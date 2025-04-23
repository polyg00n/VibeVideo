"""
Datamosh B - A different approach to datamoshing by manipulating frame data directly
"""
import numpy as np
import cv2
from .base_effect import GlitchEffect

class DatamoshBPlugin(GlitchEffect):
    """Creates glitch effects by directly manipulating frame data"""
    
    name = "Datamosh B"
    description = "Creates glitch effects by directly manipulating frame data"
    
    def __init__(self):
        super().__init__()
        self.author = "VibeVideo"
        self.version = "1.0.0"
        
        self.parameters = {
            "block_size": {
                "type": int,
                "min": 4,
                "max": 64,
                "default": 16
            },
            "corruption_strength": {
                "type": float,
                "min": 0.0,
                "max": 1.0,
                "default": 0.3
            },
            "mode": {
                "type": "choice",
                "options": ["block", "scanline", "random"],
                "default": "block"
            },
            "frame_skip": {
                "type": int,
                "min": 0,
                "max": 10,
                "default": 0
            }
        }
        
        # State tracking
        self._frame_count = 0
        self._last_frame = None

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with the datamosh effect"""
        self._frame_count += 1
        
        # Skip frames if needed
        if self._frame_count % (self.params["frame_skip"] + 1) != 0:
            self._last_frame = frame.copy()
            return frame
        
        # Get parameters
        block_size = self.params["block_size"]
        corruption_strength = self.params["corruption_strength"]
        mode = self.params["mode"]
        
        # Create a copy of the frame to work with
        result = frame.copy()
        height, width = result.shape[:2]
        
        if mode == "block":
            # Block-based corruption
            for y in range(0, height, block_size):
                for x in range(0, width, block_size):
                    if np.random.random() < corruption_strength:
                        # Randomly decide whether to use current or previous block
                        if self._last_frame is not None and np.random.random() < 0.5:
                            # Use block from previous frame
                            y_end = min(y + block_size, height)
                            x_end = min(x + block_size, width)
                            result[y:y_end, x:x_end] = self._last_frame[y:y_end, x:x_end]
                        else:
                            # Corrupt the block
                            y_end = min(y + block_size, height)
                            x_end = min(x + block_size, width)
                            block = result[y:y_end, x:x_end]
                            if len(block.shape) == 3:  # Color image
                                for c in range(3):
                                    block[:,:,c] = np.roll(block[:,:,c], 
                                                         np.random.randint(-block_size//2, block_size//2),
                                                         axis=np.random.randint(0, 2))
                            else:  # Grayscale
                                block = np.roll(block,
                                              np.random.randint(-block_size//2, block_size//2),
                                              axis=np.random.randint(0, 2))
                            result[y:y_end, x:x_end] = block
        
        elif mode == "scanline":
            # Scanline-based corruption
            for y in range(0, height, block_size):
                if np.random.random() < corruption_strength:
                    # Randomly decide whether to use current or previous scanline
                    if self._last_frame is not None and np.random.random() < 0.5:
                        # Use scanline from previous frame
                        y_end = min(y + block_size, height)
                        result[y:y_end, :] = self._last_frame[y:y_end, :]
                    else:
                        # Corrupt the scanline
                        y_end = min(y + block_size, height)
                        scanline = result[y:y_end, :]
                        if len(scanline.shape) == 3:  # Color image
                            for c in range(3):
                                scanline[:,:,c] = np.roll(scanline[:,:,c],
                                                         np.random.randint(-width//4, width//4),
                                                         axis=1)
                        else:  # Grayscale
                            scanline = np.roll(scanline,
                                             np.random.randint(-width//4, width//4),
                                             axis=1)
                        result[y:y_end, :] = scanline
        
        else:  # random mode
            # Random pixel corruption
            if np.random.random() < corruption_strength:
                # Select random region
                y_start = np.random.randint(0, height - block_size)
                x_start = np.random.randint(0, width - block_size)
                y_end = min(y_start + block_size, height)
                x_end = min(x_start + block_size, width)
                
                if self._last_frame is not None and np.random.random() < 0.5:
                    # Use region from previous frame
                    result[y_start:y_end, x_start:x_end] = self._last_frame[y_start:y_end, x_start:x_end]
                else:
                    # Corrupt the region
                    region = result[y_start:y_end, x_start:x_end]
                    if len(region.shape) == 3:  # Color image
                        for c in range(3):
                            region[:,:,c] = np.roll(region[:,:,c],
                                                  np.random.randint(-block_size//2, block_size//2),
                                                  axis=np.random.randint(0, 2))
                    else:  # Grayscale
                        region = np.roll(region,
                                       np.random.randint(-block_size//2, block_size//2),
                                       axis=np.random.randint(0, 2))
                    result[y_start:y_end, x_start:x_end] = region
        
        # Store current frame for next iteration
        self._last_frame = frame.copy()
        
        return result 