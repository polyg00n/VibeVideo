"""
Data Crunch Effect - Process pixel data using audio-like techniques
"""
import numpy as np
import cv2
from __main__ import GlitchEffect
from typing import Any, Tuple, Optional, Dict
import random
import logging

class DataCrunchEffect(GlitchEffect):
    """Process pixel data using audio-like techniques with temporal evolution"""
    
    name = "Data Crunch"
    description = "Process pixel data using audio-like techniques with temporal evolution"
    parameters: Dict[str, Any] = {
        "process_mode": {
            "type": int,
            "min": 0,
            "max": 9,
            "default": 0,
            "description": "Processing mode to apply (0: bit_crush, 1: sample_rate, 2: wave_distort, 3: freq_shift, 4: echo, 5: reverb, 6: bit_shift, 7: xor_pattern, 8: phase_shift, 9: granular)"
        },
        "intensity": {
            "type": float,
            "min": 0.0,
            "max": 2.0,  # Increased max for more impact
            "default": 1.0,
            "description": "Effect intensity"
        },
        "temporal_speed": {
            "type": float,
            "min": 0.0,
            "max": 2.0,  # Increased max for faster evolution
            "default": 0.5,
            "description": "Speed of temporal evolution"
        },
        "animate": {
            "type": bool,
            "default": True,
            "description": "Enable/disable effect animation"
        },
        "bit_depth": {
            "type": int,
            "min": 1,
            "max": 7,  # Reduced max to ensure more visible effect
            "default": 4,
            "description": "Bit depth for bit crushing"
        },
        "sample_rate": {
            "type": float,
            "min": 0.1,
            "max": 0.9,  # Reduced max to ensure more visible effect
            "default": 0.5,
            "description": "Sample rate reduction factor"
        },
        "distortion": {
            "type": float,
            "min": 0.0,
            "max": 2.0,  # Increased max for more distortion
            "default": 0.5,
            "description": "Distortion amount"
        },
        "feedback": {
            "type": float,
            "min": 0.0,
            "max": 0.98,  # Increased max for stronger feedback
            "default": 0.7,
            "description": "Feedback amount for echo/reverb"
        }
    }
    
    # Define processing modes as a class constant
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
        self._current_mode = None
        self._frozen_time = 0
        
        # Initialize temporal evolution parameters
        self._temporal_params = {
            "wave_freq": random.uniform(0.1, 0.5),
            "phase_shift": random.uniform(0, 2 * np.pi),
            "pattern_seed": random.randint(0, 1000)
        }
        
        # Setup logging
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.INFO)
    
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
    
    def _remap_to_positive(self, frame: np.ndarray) -> np.ndarray:
        """Remap values to positive range while preserving relative differences"""
        if frame.dtype == np.uint8:
            return frame
            
        # Convert to float for calculations
        frame_float = frame.astype(np.float32)
        
        # Find min and max values
        min_val = np.min(frame_float)
        max_val = np.max(frame_float)
        
        # If all values are already positive, return as is
        if min_val >= 0:
            return self._clamp_values(frame_float)
            
        # Remap values to positive range
        range_val = max_val - min_val
        if range_val == 0:
            return self._clamp_values(frame_float)
            
        # Scale to 0-255 range
        result = ((frame_float - min_val) / range_val) * 255
        return self._clamp_values(result)
    
    def _clamp_values(self, frame: np.ndarray) -> np.ndarray:
        """Clamp values to uint8 range (0-255)"""
        return np.clip(frame, 0, 255).astype(np.uint8)
    
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
            f"Speed: {params['temporal_speed']:.2f}",
            f"Animation: {'ON' if params['animate'] else 'OFF'}"
        ]
        
        # Add mode-specific parameters
        if mode_name == "bit_crush":
            lines.append(f"Bit Depth: {params['bit_depth']}")
        elif mode_name == "sample_rate":
            lines.append(f"Sample Rate: {params['sample_rate']:.2f}")
        elif mode_name == "wave_distort":
            lines.append(f"Distortion: {params['distortion']:.2f}")
        elif mode_name in ["echo", "reverb"]:
            lines.append(f"Feedback: {params['feedback']:.2f}")
        
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
    
    def _bit_crush(self, frame: np.ndarray) -> np.ndarray:
        """Reduce bit depth of the image"""
        if not self._validate_frame(frame):
            return frame
            
        bits = self.params["bit_depth"]
        self._logger.debug(f"Bit crushing with depth {bits}")
        
        # Calculate mask based on bit depth
        mask = ~((1 << (8 - bits)) - 1)
        
        # Apply mask to all channels at once
        result = frame & mask
        
        # Add some noise for more visible effect
        noise = np.random.randint(0, 1 << (8 - bits), frame.shape, dtype=np.uint8)
        result = self._clamp_values(result + noise)
        
        return result
    
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
            # Use vectorized operations for resizing
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            result = cv2.resize(resized, (w, h), interpolation=cv2.INTER_NEAREST)
            
            # Add some pixelation effect
            block_size = max(1, int(8 * (1 - factor)))
            result = cv2.resize(
                cv2.resize(result, (w // block_size, h // block_size), 
                         interpolation=cv2.INTER_NEAREST),
                (w, h), interpolation=cv2.INTER_NEAREST
            )
            
            return self._clamp_values(result)
        except Exception as e:
            self._logger.error(f"Error in sample rate reduction: {e}")
            return frame
    
    def _wave_distort(self, frame: np.ndarray) -> np.ndarray:
        """Apply wave distortion to the image"""
        h, w = frame.shape[:2]
        
        # Create wave distortion using vectorized operations
        x = np.linspace(0, 2 * np.pi, w)
        y = np.linspace(0, 2 * np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        # Create more complex wave pattern
        wave = (np.sin(X * self._temporal_params["wave_freq"] + 
                      self._temporal_params["phase_shift"]) +
                np.sin(Y * self._temporal_params["wave_freq"] * 0.5 +
                      self._temporal_params["phase_shift"] * 0.5))
        wave = (wave + 2) / 4  # Normalize to 0-1
        
        # Apply distortion to all channels at once
        distortion = self.params["distortion"]
        offset = (wave * distortion * 255).astype(np.uint8)
        
        # Apply to all channels simultaneously with more impact
        result = frame.copy()
        result = self._clamp_values(result + offset[:, :, np.newaxis] * 2)
        
        return result
    
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
        
        # Shift frequencies using vectorized operations
        shift = int(self._time * 20) % min(rows, cols)  # Increased shift speed
        dft_shift = np.roll(dft_shift, shift, axis=(0, 1))
        
        # Add some frequency domain manipulation
        mask = np.ones_like(dft_shift)
        mask[crow-30:crow+30, ccol-30:ccol+30] = 0.5
        dft_shift = dft_shift * mask
        
        # Inverse transform
        f_ishift = np.fft.ifftshift(dft_shift)
        img_back = cv2.idft(f_ishift)
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
        
        # Normalize and clip
        img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
        img_back = self._clamp_values(img_back)
        
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
        
        try:
            # Mix current frame with buffer using vectorized operations
            result = cv2.addWeighted(frame, 1 - feedback, self._buffer, feedback, 0)
            
            # Add some motion blur for echo effect
            kernel_size = int(5 * feedback)
            if kernel_size > 0:
                kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
                result = cv2.filter2D(result, -1, kernel)
            
            result = self._clamp_values(result)
            
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
            # Create multiple taps with different delays using vectorized operations
            for i in range(3):
                delay = 2 ** i
                if delay < frame.shape[0]:
                    tap = np.roll(self._buffer, delay, axis=(0, 1))
                    # Add some color shift to each tap
                    tap = cv2.cvtColor(tap, cv2.COLOR_BGR2HSV)
                    tap[:, :, 0] = (tap[:, :, 0] + i * 30) % 180
                    tap = cv2.cvtColor(tap, cv2.COLOR_HSV2BGR)
                    result = cv2.addWeighted(result, 1 - feedback/3, tap, feedback/3, 0)
            
            # Add some blur for reverb effect
            kernel_size = int(3 * feedback)
            if kernel_size > 0:
                kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
                result = cv2.filter2D(result, -1, kernel)
            
            # Clip and update buffer
            result = self._clamp_values(result)
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
        
        # Create a pattern based on the shift value using vectorized operations
        h, w = frame.shape[:2]
        x = np.linspace(0, 1, w)
        y = np.linspace(0, 1, h)
        X, Y = np.meshgrid(x, y)
        
        # Create a more complex wave pattern
        pattern = (np.sin(X * (shift + 1) * np.pi + Y * (shift + 1) * np.pi) +
                  np.sin(X * (shift + 2) * np.pi * 0.5 + Y * (shift + 2) * np.pi * 0.5))
        pattern = (pattern + 2) / 4  # Normalize to 0-1
        
        # Apply pattern to all channels simultaneously
        result = np.abs(frame_float - pattern[:, :, np.newaxis])
        
        # Add some color shift
        result = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        result[:, :, 0] = (result[:, :, 0] + shift * 30) % 180
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        
        return self._clamp_values(result)
    
    def _xor_pattern(self, frame: np.ndarray) -> np.ndarray:
        """Apply XOR-like effect using mathematical operations"""
        # Generate a pattern using sine waves
        h, w = frame.shape[:2]
        x = np.linspace(0, 2 * np.pi, w)
        y = np.linspace(0, 2 * np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        # Create a more complex pattern using vectorized operations
        pattern = (np.sin(X * self._temporal_params["wave_freq"] + 
                         Y * self._temporal_params["wave_freq"] + 
                         self._temporal_params["phase_shift"]) +
                  np.sin(X * self._temporal_params["wave_freq"] * 0.5 + 
                         Y * self._temporal_params["wave_freq"] * 0.5 + 
                         self._temporal_params["phase_shift"] * 0.5))
        pattern = (pattern + 2) / 4  # Normalize to 0-1
        
        # Convert to float for calculations
        frame_float = frame.astype(np.float32) / 255.0
        
        # Apply pattern to all channels simultaneously
        result = np.abs(frame_float - pattern[:, :, np.newaxis])
        
        # Add some color shift
        result = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_BGR2HSV)
        result[:, :, 0] = (result[:, :, 0] + int(self._time * 10) % 180) % 180
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        
        return self._clamp_values(result)
    
    def _phase_shift(self, frame: np.ndarray) -> np.ndarray:
        """Apply phase shift to the image"""
        self._phase_offset = (self._phase_offset + self._time * 0.2) % (2 * np.pi)  # Increased speed
        h, w = frame.shape[:2]
        
        # Create phase shift pattern using vectorized operations
        x = np.linspace(0, 2 * np.pi, w)
        y = np.linspace(0, 2 * np.pi, h)
        X, Y = np.meshgrid(x, y)
        
        # Create more complex phase pattern
        phase = (np.sin(X + Y + self._phase_offset) +
                np.sin(X * 0.5 + Y * 0.5 + self._phase_offset * 0.5))
        phase = (phase + 2) / 4  # Normalize to 0-1
        
        # Apply phase shift with clipping to all channels simultaneously
        intensity = self.params["intensity"]
        result = frame.copy()
        result = self._clamp_values(result * (1 + phase[:, :, np.newaxis] * intensity * 2))
        
        # Add some color shift
        result = cv2.cvtColor(result, cv2.COLOR_BGR2HSV)
        result[:, :, 0] = (result[:, :, 0] + int(self._phase_offset * 30) % 180) % 180
        result = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
        
        return result
    
    def _granular(self, frame: np.ndarray) -> np.ndarray:
        """Apply granular synthesis to the image"""
        h, w = frame.shape[:2]
        grain_size = max(1, int(min(h, w) * 0.1))  # Ensure at least 1 pixel
        
        # Create grains using vectorized operations
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
                    if random.random() < 0.3:
                        # Add some color shift
                        grain = cv2.cvtColor(grain, cv2.COLOR_BGR2HSV)
                        grain[:, :, 0] = (grain[:, :, 0] + random.randint(0, 180)) % 180
                        grain = cv2.cvtColor(grain, cv2.COLOR_HSV2BGR)
                    grains.append((y, x, grain))
        
        # Reconstruct image
        result = frame.copy()
        for y, x, grain in grains:
            h_grain, w_grain = grain.shape[:2]
            # Ensure we don't exceed the frame boundaries
            y_end = min(y + h_grain, h)
            x_end = min(x + w_grain, w)
            h_grain = y_end - y
            w_grain = x_end - x
            if h_grain > 0 and w_grain > 0:
                result[y:y_end, x:x_end] = self._clamp_values(grain[:h_grain, :w_grain])
        
        return result
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame with the selected effect"""
        if not self._validate_frame(frame):
            return frame
            
        try:
            # Update temporal parameters only if animation is enabled
            if self.params["animate"]:
                self._time += self.params["temporal_speed"] * 0.1
            else:
                # Use frozen time for consistent effect
                if self._frozen_time == 0:
                    self._frozen_time = self._time
            
            # Get current processing mode
            mode = self.params["process_mode"]
            mode_name = self.PROCESS_MODES.get(mode, "bit_crush")  # Default to bit_crush if invalid mode
            
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
            result = self._clamp_values(result)
            
            # Add text feedback
            result = self._add_text_feedback(result, mode_name, self.params)
            
            # Store last frame for temporal effects
            self._last_frame = frame.copy()
            
            return result
        except Exception as e:
            self._logger.error(f"Error processing frame: {e}")
            return frame 