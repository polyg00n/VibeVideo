"""
Data Crunch Effect - Process pixel data using audio-like techniques
"""
import numpy as np
import cv2
from base_effect import GlitchEffect
from typing import Any, Tuple, Optional
import random
import logging

class DataCrunchEffect(GlitchEffect):
    """Process pixel data using audio-like techniques with temporal evolution"""
    
    name = "Data Crunch"
    description = "Process pixel data using audio-like techniques with temporal evolution"
    parameters = {
        "process_mode": {
            "type": int,
            "min": 0,
            "max": len(PROCESS_MODES) - 1,
            "default": 0
        },
        "intensity": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "default": 0.5
        },
        "temporal_speed": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "default": 0.1
        },
        "bit_depth": {
            "type": int,
            "min": 1,
            "max": 8,
            "default": 4
        },
        "sample_rate": {
            "type": float,
            "min": 0.1,
            "max": 1.0,
            "default": 0.5
        },
        "distortion": {
            "type": float,
            "min": 0.0,
            "max": 1.0,
            "default": 0.3
        },
        "feedback": {
            "type": float,
            "min": 0.0,
            "max": 0.95,
            "default": 0.5
        }
    }
    # Define processing modes
    PROCESS_MODES = {
        0: "bit_crush",      # Reduce bit depth
        1: "sample_rate",    # Reduce sample rate
        2: "wave_distort",   # Apply wave distortion
        3: "freq_shift",     # Frequency shift
        4: "echo",           # Echo effect
        5: "reverb",         # Reverb effect
        6: "bit_shift",      # Bit shift operations
        7: "xor_pattern",    # XOR with pattern
        8: "phase_shift",    # Phase shift
        9: "granular"        # Granular synthesis
    }
    
    
    
    def __init__(self):
        super().__init__()
        self._time = 0
        self._last_frame: Optional[np.ndarray] = None
        self._buffer: Optional[np.ndarray] = None
        self._pattern: Optional[np.ndarray] = None
        self._phase_offset = 0
        
        # Initialize temporal evolution parameters
        self._temporal_params = {
            "wave_freq": random.uniform(0.1, 0.5),
            "phase_shift": random.uniform(0, 2 * np.pi),
            "pattern_seed": random.randint(0, 1000)
        }
        
        # Setup logging
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
    
    def set_param(self, name: str, value: Any) -> None:
        """Set a parameter value with type safety and validation"""
        self._logger.debug(f"Setting parameter {name} to {value}")
        self._logger.debug(f"Current params before change: {self.params}")
        
        if name in self.params:
            param_details = self.parameters[name]
            expected_type = param_details["type"]
            
            try:
                # Convert value to expected type
                if expected_type == int:
                    value = int(value)
                    # Ensure value is within bounds
                    if "min" in param_details and value < param_details["min"]:
                        value = param_details["min"]
                    if "max" in param_details and value > param_details["max"]:
                        value = param_details["max"]
                elif expected_type == float:
                    value = float(value)
                    # Ensure value is within bounds
                    if "min" in param_details and value < param_details["min"]:
                        value = param_details["min"]
                    if "max" in param_details and value > param_details["max"]:
                        value = param_details["max"]
                
                self.params[name] = value
                self._logger.debug(f"Successfully set {name} to {value}")
                self._logger.debug(f"Params after change: {self.params}")
                
            except Exception as e:
                self._logger.error(f"Failed to set parameter {name}: {e}")
    
    def _validate_frame(self, frame: np.ndarray) -> bool:
        """Validate input frame"""
        if not isinstance(frame, np.ndarray):
            self._logger.error("Input frame must be a numpy array")
            return False
        if frame.dtype != np.uint8:
            self._logger.error("Input frame must be uint8 type")
            return False
        if len(frame.shape) not in (2, 3):
            self._logger.error("Input frame must be 2D (grayscale) or 3D (color)")
            return False
        return True
    
    def _generate_pattern(self, size: Tuple[int, int]) -> np.ndarray:
        """Generate a random pattern for XOR operations"""
        try:
            if self._pattern is None or self._pattern.shape != size:
                np.random.seed(self._temporal_params["pattern_seed"])
                self._pattern = np.random.randint(0, 256, size, dtype=np.uint8)
            return self._pattern
        except Exception as e:
            self._logger.error(f"Error generating pattern: {e}")
            return np.zeros(size, dtype=np.uint8)
    
    def _bit_crush(self, frame: np.ndarray) -> np.ndarray:
        """Reduce bit depth of the image"""
        if not self._validate_frame(frame):
            return frame
            
        bits = self.params["bit_depth"]
        self._logger.debug(f"Bit crushing with depth {bits}")
        mask = ~((1 << (8 - bits)) - 1)
        
        if len(frame.shape) == 3:  # Color image
            return np.clip(frame & mask, 0, 255).astype(np.uint8)
        else:  # Grayscale image
            return np.clip(frame & mask, 0, 255).astype(np.uint8)
    
    def _sample_rate_reduce(self, frame: np.ndarray) -> np.ndarray:
        """Reduce sample rate of the image"""
        if not self._validate_frame(frame):
            return frame
            
        factor = self.params["sample_rate"]
        self._logger.debug(f"Reducing sample rate with factor {factor}")
        
        h, w = frame.shape[:2]
        new_h = max(1, int(h * factor))
        new_w = max(1, int(w * factor))
        
        try:
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            return cv2.resize(resized, (w, h), interpolation=cv2.INTER_NEAREST)
        except Exception as e:
            self._logger.error(f"Error in sample rate reduction: {e}")
            return frame
    
    def _wave_distort(self, frame: np.ndarray) -> np.ndarray:
        """Apply wave distortion to the image"""
        h, w = frame.shape[:2]
        x = np.linspace(0, 2 * np.pi, w)
        y = np.linspace(0, 2 * np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        # Create wave distortion
        wave = np.sin(X * self._temporal_params["wave_freq"] + 
                     self._temporal_params["phase_shift"])
        wave = (wave + 1) / 2  # Normalize to 0-1
        
        # Apply distortion
        distortion = self.params["distortion"]
        offset = (wave * distortion * 255).astype(np.uint8)
        
        # Apply to each channel with clipping
        result = frame.copy()
        for c in range(3):
            result[:, :, c] = np.clip(result[:, :, c] + offset, 0, 255)
        
        return result.astype(np.uint8)
    
    def _freq_shift(self, frame: np.ndarray) -> np.ndarray:
        """Apply frequency shift to the image"""
        # Convert to grayscale and float32 for DFT
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        gray_float = np.float32(gray)
        
        # Convert to frequency domain
        dft = cv2.dft(gray_float, flags=cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)
        
        # Create frequency shift
        rows, cols = gray.shape
        crow, ccol = rows // 2, cols // 2
        
        # Shift frequencies
        shift = int(self._time * 10) % min(rows, cols)
        dft_shift = np.roll(dft_shift, shift, axis=0)
        dft_shift = np.roll(dft_shift, shift, axis=1)
        
        # Inverse transform
        f_ishift = np.fft.ifftshift(dft_shift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        
        # Normalize and clip
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        img_back = np.clip(img_back, 0, 255).astype(np.uint8)
        
        # Convert back to BGR if input was color
        if len(frame.shape) == 3:
            return cv2.cvtColor(img_back, cv2.COLOR_GRAY2BGR)
        return img_back
    
    def _echo(self, frame: np.ndarray) -> np.ndarray:
        """Apply echo effect to the image"""
        if not self._validate_frame(frame):
            return frame
            
        if self._buffer is None:
            self._buffer = frame.copy()
        
        feedback = self.params["feedback"]
        result = frame.copy()
        
        try:
            # Mix current frame with buffer and clip
            result = cv2.addWeighted(result, 1 - feedback, self._buffer, feedback, 0)
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # Update buffer
            self._buffer = result.copy()
            return result
        except Exception as e:
            self._logger.error(f"Error in echo effect: {e}")
            return frame
    
    def _reverb(self, frame: np.ndarray) -> np.ndarray:
        """Apply reverb effect to the image"""
        if not self._validate_frame(frame):
            return frame
            
        if self._buffer is None:
            self._buffer = frame.copy()
        
        feedback = self.params["feedback"]
        result = frame.copy()
        
        try:
            # Create multiple taps with different delays
            for i in range(3):
                delay = 2 ** i
                if delay < frame.shape[0]:
                    tap = np.roll(self._buffer, delay, axis=0)
                    tap = np.roll(tap, delay, axis=1)
                    result = cv2.addWeighted(result, 1 - feedback/3, tap, feedback/3, 0)
            
            # Clip and update buffer
            result = np.clip(result, 0, 255).astype(np.uint8)
            self._buffer = result.copy()
            return result
        except Exception as e:
            self._logger.error(f"Error in reverb effect: {e}")
            return frame
    
    def _bit_shift(self, frame: np.ndarray) -> np.ndarray:
        """Apply bit shift-like effect using mathematical operations"""
        shift = int(self._time * 10) % 8
        self._logger.debug(f"Applying bit shift with shift {shift}")
        
        # Convert to float for calculations
        frame_float = frame.astype(np.float32) / 255.0
        
        # Create a pattern based on the shift value
        h, w = frame.shape[:2]
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Create a wave pattern that changes with shift
        pattern = np.sin(X * (shift + 1) * np.pi + Y * (shift + 1) * np.pi)
        pattern = (pattern + 1) / 2  # Normalize to 0-1
        
        # Apply pattern to each channel
        result = frame_float.copy()
        for c in range(3):
            result[:, :, c] = np.abs(result[:, :, c] - pattern)
        
        # Convert back to uint8
        return (result * 255).astype(np.uint8)
    
    def _xor_pattern(self, frame: np.ndarray) -> np.ndarray:
        """Apply XOR-like effect using mathematical operations"""
        # Generate a pattern using sine waves
        h, w = frame.shape[:2]
        x = np.linspace(0, 2 * np.pi, w)
        y = np.linspace(0, 2 * np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        # Create a complex pattern
        pattern = np.sin(X * self._temporal_params["wave_freq"] + 
                        Y * self._temporal_params["wave_freq"] + 
                        self._temporal_params["phase_shift"])
        pattern = (pattern + 1) / 2  # Normalize to 0-1
        
        # Convert to float for calculations
        frame_float = frame.astype(np.float32) / 255.0
        
        # Apply pattern to each channel
        result = frame_float.copy()
        for c in range(3):
            result[:, :, c] = np.abs(result[:, :, c] - pattern)
        
        # Convert back to uint8
        return (result * 255).astype(np.uint8)
    
    def _phase_shift(self, frame: np.ndarray) -> np.ndarray:
        """Apply phase shift to the image"""
        self._phase_offset = (self._phase_offset + self._time * 0.1) % (2 * np.pi)
        h, w = frame.shape[:2]
        
        # Create phase shift pattern
        x = np.linspace(0, 2 * np.pi, w)
        y = np.linspace(0, 2 * np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        phase = np.sin(X + Y + self._phase_offset)
        phase = (phase + 1) / 2  # Normalize to 0-1
        
        # Apply phase shift with clipping
        intensity = self.params["intensity"]
        result = frame.copy()
        for c in range(3):
            result[:, :, c] = np.clip(
                result[:, :, c] * (1 + phase * intensity),
                0, 255
            ).astype(np.uint8)
        
        return result
    
    def _granular(self, frame: np.ndarray) -> np.ndarray:
        """Apply granular synthesis to the image"""
        h, w = frame.shape[:2]
        grain_size = max(1, int(min(h, w) * 0.1))  # Ensure at least 1 pixel
        
        # Create grains
        grains = []
        for y in range(0, h, grain_size):
            for x in range(0, w, grain_size):
                grain = frame[y:y+grain_size, x:x+grain_size]
                if grain.size > 0:
                    # Randomly process grain
                    if random.random() < 0.3:
                        grain = cv2.flip(grain, random.randint(-1, 1))
                    if random.random() < 0.3:
                        grain = cv2.rotate(grain, random.randint(0, 3))
                    grains.append((y, x, grain))
        
        # Reconstruct image
        result = frame.copy()
        for y, x, grain in grains:
            h_grain, w_grain = grain.shape[:2]
            result[y:y+h_grain, x:x+w_grain] = grain
        
        return result.astype(np.uint8)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame with the selected effect"""
        if not self._validate_frame(frame):
            return frame
            
        try:
            # Update temporal parameters
            self._time += self.params["temporal_speed"] * 0.1
            
            # Get current processing mode
            mode = self.params["process_mode"]
            mode_name = self.PROCESS_MODES[mode]
            self._logger.debug(f"Processing frame with mode {mode_name} ({mode})")
            
            # Apply selected processing mode
            if mode_name == "bit_crush":
                result = self._bit_crush(frame)
            elif mode_name == "sample_rate":
                result = self._sample_rate_reduce(frame)
            elif mode_name == "wave_distort":
                result = self._wave_distort(frame)
            elif mode_name == "freq_shift":
                result = self._freq_shift(frame)
            elif mode_name == "echo":
                result = self._echo(frame)
            elif mode_name == "reverb":
                result = self._reverb(frame)
            elif mode_name == "bit_shift":
                result = self._bit_shift(frame)
            elif mode_name == "xor_pattern":
                result = self._xor_pattern(frame)
            elif mode_name == "phase_shift":
                result = self._phase_shift(frame)
            elif mode_name == "granular":
                result = self._granular(frame)
            else:
                result = frame
            
            # Apply intensity with clipping
            intensity = self.params["intensity"]
            result = cv2.addWeighted(frame, 1 - intensity, result, intensity, 0)
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # Store last frame for temporal effects
            self._last_frame = frame.copy()
            
            return result
        except Exception as e:
            self._logger.error(f"Error processing frame: {e}")
            return frame 