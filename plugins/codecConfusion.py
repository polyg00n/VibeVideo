"""
Codec Confusion Plugin for Video Glitch Player
Simulates the effects of interpreting video with incorrect codecs

This plugin creates various codec-like artifacts by manipulating the video frames
in ways that mimic how different codecs would interpret the data incorrectly.
The effects range from subtle compression artifacts to more extreme glitch-like
effects depending on the intensity setting.

Key Features:
1. Codec Selection: Choose from 7 different codec simulations
2. Adjustable Parameters: Fine-tune the effect intensity and characteristics
3. Codec-Specific Effects: Each codec has unique artifact patterns
4. Additional Features: Color channel shifting and block size control

The plugin works by applying different types of block-based compression,
chroma subsampling, and color channel manipulation to simulate how each codec
would incorrectly interpret the video data.
"""
import cv2
import numpy as np
from __main__ import GlitchEffect

class CodecConfusionEffect(GlitchEffect):
    """Simulates the effects of interpreting video with incorrect codecs"""
    
    name = "Codec Confusion"
    description = "Simulates interpreting video with incorrect codecs"
    
    # ===== Parameters =====
    # codec_type: Select from different codec simulations
    #   - H.264: Modern compression with macroblock artifacts
    #   - MPEG-2: Classic compression with ringing artifacts
    #   - VP9: Variable block size compression
    #   - ProRes: Professional format with chroma subsampling
    #   - DV: Strong block artifacts with noise
    #   - MJPEG: Frame-by-frame compression with blocking
    #   - H.265: Advanced compression with weighted blocks
    #
    # intensity: Controls the strength of all effects (0.0 to 1.0)
    #   - Lower values create subtle artifacts
    #   - Higher values create more extreme glitch effects
    #
    # block_size: Controls the size of compression blocks (4 to 32 pixels)
    #   - Smaller blocks create finer artifacts
    #   - Larger blocks create more visible compression
    #
    # color_shift: Toggles color channel misalignment
    #   - Simulates incorrect color space interpretation
    #   - Creates RGB channel separation effects
    parameters = {
        "codec_type": {
            "type": int,
            "min": 0,
            "max": 6,
            "default": 0,
            "options": [
                "H.264 (MPEG-4 AVC)",
                "MPEG-2",
                "VP9",
                "ProRes",
                "DV (Digital Video)",
                "MJPEG",
                "H.265 (HEVC)"
            ]
        },
        "intensity": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5},
        "block_size": {"type": int, "min": 4, "max": 32, "default": 8},
        "color_shift": {"type": bool, "default": True}
    }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Main processing function that applies the selected codec effect"""
        h, w = frame.shape[:2]
        result = frame.copy()
        
        # Get parameters
        codec_type = self.params["codec_type"]
        intensity = self.params["intensity"]
        block_size = self.params["block_size"]
        color_shift = self.params["color_shift"]
        
        # Apply codec-specific effects
        if codec_type == 0:  # H.264
            result = self._apply_h264_effect(result, intensity, block_size)
        elif codec_type == 1:  # MPEG-2
            result = self._apply_mpeg2_effect(result, intensity, block_size)
        elif codec_type == 2:  # VP9
            result = self._apply_vp9_effect(result, intensity, block_size)
        elif codec_type == 3:  # ProRes
            result = self._apply_prores_effect(result, intensity)
        elif codec_type == 4:  # DV
            result = self._apply_dv_effect(result, intensity, block_size)
        elif codec_type == 5:  # MJPEG
            result = self._apply_mjpeg_effect(result, intensity)
        elif codec_type == 6:  # H.265
            result = self._apply_h265_effect(result, intensity, block_size)
        
        # Apply color shift if enabled
        if color_shift:
            result = self._apply_color_shift(result, intensity)
            
        return result
    
    def _apply_h264_effect(self, frame: np.ndarray, intensity: float, block_size: int) -> np.ndarray:
        """
        Simulate H.264 compression artifacts
        - Applies block-based compression with macroblock artifacts
        - Creates characteristic H.264 compression patterns
        - Adds random macroblock corruption at higher intensities
        """
        result = frame.copy()
        
        # Apply block-based compression
        for y in range(0, frame.shape[0], block_size):
            for x in range(0, frame.shape[1], block_size):
                block = frame[y:y+block_size, x:x+block_size]
                if block.size > 0:
                    # Average the block
                    avg_color = np.mean(block, axis=(0,1))
                    # Apply intensity-based variation
                    variation = np.random.normal(0, intensity * 20, 3)
                    avg_color = np.clip(avg_color + variation, 0, 255)
                    # Fill block with averaged color
                    result[y:y+block_size, x:x+block_size] = avg_color
        
        # Add some macroblock artifacts
        if intensity > 0.3:
            for y in range(0, frame.shape[0], block_size*2):
                for x in range(0, frame.shape[1], block_size*2):
                    if np.random.random() < intensity:
                        result[y:y+block_size, x:x+block_size] = np.random.randint(0, 255, 3)
        
        return result
    
    def _apply_mpeg2_effect(self, frame: np.ndarray, intensity: float, block_size: int) -> np.ndarray:
        """
        Simulate MPEG-2 compression artifacts
        - Applies stronger block artifacts than H.264
        - Creates characteristic ringing artifacts
        - Uses more aggressive color averaging
        """
        result = frame.copy()
        
        # Apply stronger block artifacts
        for y in range(0, frame.shape[0], block_size):
            for x in range(0, frame.shape[1], block_size):
                block = frame[y:y+block_size, x:x+block_size]
                if block.size > 0:
                    # More aggressive averaging
                    avg_color = np.mean(block, axis=(0,1))
                    # Add more variation
                    variation = np.random.normal(0, intensity * 30, 3)
                    avg_color = np.clip(avg_color + variation, 0, 255)
                    result[y:y+block_size, x:x+block_size] = avg_color
        
        # Add ringing artifacts
        if intensity > 0.4:
            kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]]) * intensity
            result = cv2.filter2D(result, -1, kernel)
        
        return result
    
    def _apply_vp9_effect(self, frame: np.ndarray, intensity: float, block_size: int) -> np.ndarray:
        """
        Simulate VP9 compression artifacts
        - Uses variable block sizes for compression
        - Creates more subtle artifacts than H.264
        - Implements VP9's characteristic block size variation
        """
        result = frame.copy()
        
        # Apply variable block size compression
        sizes = [block_size, block_size*2, block_size//2]
        for y in range(0, frame.shape[0], block_size):
            for x in range(0, frame.shape[1], block_size):
                size = np.random.choice(sizes)
                block = frame[y:y+size, x:x+size]
                if block.size > 0:
                    # More subtle averaging
                    avg_color = np.mean(block, axis=(0,1))
                    variation = np.random.normal(0, intensity * 15, 3)
                    avg_color = np.clip(avg_color + variation, 0, 255)
                    result[y:y+size, x:x+size] = avg_color
        
        return result
    
    def _apply_prores_effect(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """
        Simulate ProRes compression artifacts
        - Implements chroma subsampling in 2x2 blocks
        - Creates banding artifacts in luminance
        - Simulates ProRes' characteristic color space handling
        """
        result = frame.copy()
        
        # Convert to YUV
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        # Apply chroma subsampling
        if intensity > 0.3:
            # Average UV channels in 2x2 blocks
            for y in range(0, yuv.shape[0], 2):
                for x in range(0, yuv.shape[1], 2):
                    uv_block = yuv[y:y+2, x:x+2, 1:3]
                    if uv_block.size > 0:
                        avg_uv = np.mean(uv_block, axis=(0,1))
                        yuv[y:y+2, x:x+2, 1:3] = avg_uv
        
        # Add some banding artifacts
        if intensity > 0.5:
            yuv[:,:,0] = np.round(yuv[:,:,0] / (intensity * 20)) * (intensity * 20)
        
        result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return result
    
    def _apply_dv_effect(self, frame: np.ndarray, intensity: float, block_size: int) -> np.ndarray:
        """
        Simulate DV compression artifacts
        - Applies very aggressive block averaging
        - Adds significant noise to simulate DV tape artifacts
        - Creates characteristic DV compression patterns
        """
        result = frame.copy()
        
        # Apply strong block artifacts
        for y in range(0, frame.shape[0], block_size):
            for x in range(0, frame.shape[1], block_size):
                block = frame[y:y+block_size, x:x+block_size]
                if block.size > 0:
                    # Very aggressive averaging
                    avg_color = np.mean(block, axis=(0,1))
                    # Add significant variation
                    variation = np.random.normal(0, intensity * 40, 3)
                    avg_color = np.clip(avg_color + variation, 0, 255)
                    result[y:y+block_size, x:x+block_size] = avg_color
        
        # Add some noise
        if intensity > 0.4:
            noise = np.random.normal(0, intensity * 25, frame.shape)
            result = np.clip(result + noise, 0, 255).astype(np.uint8)
        
        return result
    
    def _apply_mjpeg_effect(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """
        Simulate MJPEG compression artifacts
        - Implements strong chroma subsampling in 4x4 blocks
        - Creates characteristic JPEG blocking artifacts
        - Simulates frame-by-frame compression
        """
        result = frame.copy()
        
        # Convert to YUV
        yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
        
        # Apply strong chroma subsampling
        if intensity > 0.3:
            # Average UV channels in 4x4 blocks
            for y in range(0, yuv.shape[0], 4):
                for x in range(0, yuv.shape[1], 4):
                    uv_block = yuv[y:y+4, x:x+4, 1:3]
                    if uv_block.size > 0:
                        avg_uv = np.mean(uv_block, axis=(0,1))
                        yuv[y:y+4, x:x+4, 1:3] = avg_uv
        
        # Add blocking artifacts
        if intensity > 0.5:
            for y in range(0, yuv.shape[0], 8):
                for x in range(0, yuv.shape[1], 8):
                    y_block = yuv[y:y+8, x:x+8, 0]
                    if y_block.size > 0:
                        avg_y = np.mean(y_block)
                        yuv[y:y+8, x:x+8, 0] = np.round(avg_y / (intensity * 10)) * (intensity * 10)
        
        result = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        return result
    
    def _apply_h265_effect(self, frame: np.ndarray, intensity: float, block_size: int) -> np.ndarray:
        """
        Simulate H.265 compression artifacts
        - Uses sophisticated block-based compression
        - Implements weighted averaging for smoother transitions
        - Creates more subtle artifacts than H.264
        """
        result = frame.copy()
        
        # Apply more sophisticated block-based compression
        sizes = [block_size, block_size*2, block_size//2]
        for y in range(0, frame.shape[0], block_size):
            for x in range(0, frame.shape[1], block_size):
                size = np.random.choice(sizes)
                block = frame[y:y+size, x:x+size]
                if block.size > 0:
                    # Use weighted average
                    weights = np.random.random(block.shape[:2])
                    weights = weights / np.sum(weights)
                    avg_color = np.sum(block * weights[..., np.newaxis], axis=(0,1))
                    # Add subtle variation
                    variation = np.random.normal(0, intensity * 10, 3)
                    avg_color = np.clip(avg_color + variation, 0, 255)
                    result[y:y+size, x:x+size] = avg_color
        
        return result
    
    def _apply_color_shift(self, frame: np.ndarray, intensity: float) -> np.ndarray:
        """
        Apply color channel shifting
        - Shifts RGB channels in different directions
        - Creates color separation effects
        - Simulates incorrect color space interpretation
        """
        result = frame.copy()
        
        # Shift color channels
        shift = int(intensity * 5)
        if shift > 0:
            # Shift red channel
            result[:, shift:, 2] = frame[:, :-shift, 2]
            # Shift green channel
            result[shift:, :, 1] = frame[:-shift, :, 1]
            # Shift blue channel
            result[:, :-shift, 0] = frame[:, shift:, 0]
        
        return result 