"""
LUT Loader Effect - Apply color lookup tables to video with color controls
"""
import numpy as np
import cv2
from __main__ import GlitchEffect
from typing import Any, Tuple, Optional, Dict
import os
import logging
from tkinter import filedialog
import re

class LUTLoaderEffect(GlitchEffect):
    """Apply color lookup tables (LUTs) to video with color controls"""
    
    name = "LUT Loader"
    description = "Load and apply color lookup tables (LUTs) with color controls"
    parameters: Dict[str, Any] = {
        "lut_file": {
            "type": str,
            "default": "",
            "description": "Path to LUT file (.cube, .3dl)"
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
        },
        "saturation": {
            "type": float,
            "min": 0.0,
            "max": 2.0,
            "default": 1.0,
            "description": "Color saturation"
        }
    }
    
    def __init__(self):
        super().__init__()
        self._lut: Optional[np.ndarray] = None
        self._lut_size: int = 0
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
        
        # Initialize LUT if file is specified
        if self.params["lut_file"]:
            self._load_lut(self.params["lut_file"])
    
    def _load_lut(self, file_path: str) -> bool:
        """Load a LUT file in .cube or .3dl format"""
        try:
            if not os.path.exists(file_path):
                self._logger.error(f"LUT file not found: {file_path}")
                return False
                
            ext = os.path.splitext(file_path)[1].lower()
            if ext == '.cube':
                return self._load_cube_lut(file_path)
            elif ext == '.3dl':
                return self._load_3dl_lut(file_path)
            else:
                self._logger.error(f"Unsupported LUT format: {ext}")
                return False
                
        except Exception as e:
            self._logger.error(f"Error loading LUT file: {e}")
            return False
    
    def _load_cube_lut(self, file_path: str) -> bool:
        """Load a .cube format LUT file"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse LUT size
            size_line = next(line for line in lines if line.startswith('LUT_3D_SIZE'))
            self._lut_size = int(size_line.split()[-1])
            
            # Create LUT array
            self._lut = np.zeros((self._lut_size, self._lut_size, self._lut_size, 3), dtype=np.float32)
            
            # Parse LUT data
            data_lines = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('LUT_3D_SIZE') or line.startswith('LUT_3D_INPUT_RANGE'):
                    continue
                try:
                    # Try to parse as float values
                    r, g, b = map(float, line.split())
                    data_lines.append((r, g, b))
                except ValueError:
                    # Skip lines that can't be parsed as floats
                    continue
            
            # Fill LUT array
            for i, (r, g, b) in enumerate(data_lines):
                z = i // (self._lut_size * self._lut_size)
                y = (i // self._lut_size) % self._lut_size
                x = i % self._lut_size
                self._lut[x, y, z] = [r, g, b]
            
            # Log LUT statistics for debugging
            self._logger.info(f"Loaded CUBE LUT with size {self._lut_size}")
            self._logger.info(f"LUT min values: {np.min(self._lut, axis=(0,1,2))}")
            self._logger.info(f"LUT max values: {np.max(self._lut, axis=(0,1,2))}")
            self._logger.info(f"LUT mean values: {np.mean(self._lut, axis=(0,1,2))}")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error parsing CUBE LUT: {e}")
            return False
    
    def _load_3dl_lut(self, file_path: str) -> bool:
        """Load a .3dl format LUT file"""
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Parse LUT size
            size_line = next(line for line in lines if 'LUT_3D_SIZE' in line)
            self._lut_size = int(re.search(r'\d+', size_line).group())
            
            # Create LUT array
            self._lut = np.zeros((self._lut_size, self._lut_size, self._lut_size, 3), dtype=np.float32)
            
            # Parse LUT data
            data_lines = []
            for line in lines:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if 'LUT_3D_SIZE' in line:
                    continue
                try:
                    # Try to parse as float values
                    r, g, b = map(float, line.split())
                    data_lines.append((r, g, b))
                except ValueError:
                    # Skip lines that can't be parsed as floats
                    continue
            
            # Fill LUT array
            for i, (r, g, b) in enumerate(data_lines):
                z = i // (self._lut_size * self._lut_size)
                y = (i // self._lut_size) % self._lut_size
                x = i % self._lut_size
                self._lut[x, y, z] = [r, g, b]
            
            # Log LUT statistics for debugging
            self._logger.info(f"Loaded 3DL LUT with size {self._lut_size}")
            self._logger.info(f"LUT min values: {np.min(self._lut, axis=(0,1,2))}")
            self._logger.info(f"LUT max values: {np.max(self._lut, axis=(0,1,2))}")
            self._logger.info(f"LUT mean values: {np.mean(self._lut, axis=(0,1,2))}")
            
            return True
            
        except Exception as e:
            self._logger.error(f"Error parsing 3DL LUT: {e}")
            return False
    
    def _apply_color_shifts(self, frame: np.ndarray) -> np.ndarray:
        """Apply color channel shifts and saturation"""
        # Convert to float for calculations
        frame_float = frame.astype(np.float32) / 255.0
        
        # Apply color shifts
        frame_float[:, :, 0] = np.clip(frame_float[:, :, 0] + self.params["red_shift"], 0, 1)
        frame_float[:, :, 1] = np.clip(frame_float[:, :, 1] + self.params["green_shift"], 0, 1)
        frame_float[:, :, 2] = np.clip(frame_float[:, :, 2] + self.params["blue_shift"], 0, 1)
        
        # Apply saturation
        if self.params["saturation"] != 1.0:
            hsv = cv2.cvtColor(frame_float, cv2.COLOR_BGR2HSV)
            hsv[:, :, 1] = np.clip(hsv[:, :, 1] * self.params["saturation"], 0, 1)
            frame_float = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return (frame_float * 255).astype(np.uint8)
    
    def _apply_lut(self, frame: np.ndarray) -> np.ndarray:
        """Apply the loaded LUT to the frame using vectorized operations"""
        if self._lut is None:
            return frame
            
        # Convert frame to float and normalize
        frame_float = frame.astype(np.float32) / 255.0
        
        # Calculate indices for LUT lookup
        indices = (frame_float * (self._lut_size - 1)).astype(np.int32)
        
        # Ensure indices are within bounds
        indices = np.clip(indices, 0, self._lut_size - 1)
        
        # Apply LUT using advanced indexing
        result = self._lut[indices[:, :, 0], indices[:, :, 1], indices[:, :, 2]]
        
        # Convert back to uint8 and ensure values are in range
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        
        return result
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame with the LUT and color controls"""
        try:
            # Apply color shifts and saturation
            frame = self._apply_color_shifts(frame)
            
            # Apply LUT if loaded
            if self._lut is not None:
                frame = self._apply_lut(frame)
            
            return frame
            
        except Exception as e:
            self._logger.error(f"Error processing frame: {e}")
            return frame
    
    def set_param(self, name: str, value: Any) -> None:
        """Override set_param to handle LUT file loading"""
        if name == "lut_file":
            # If value is empty string, show file dialog
            if value == "":
                file_path = filedialog.askopenfilename(
                    title="Select LUT File",
                    filetypes=[("LUT files", "*.cube *.3dl"), ("All files", "*.*")]
                )
                if file_path:
                    value = file_path
                    self._load_lut(file_path)
            else:
                self._load_lut(value)
        
        super().set_param(name, value) 