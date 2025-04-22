"""
Binary Manipulation Plugin - For more advanced glitch effects
This plugin works with the raw binary data of video files
"""
import os
import random
import numpy as np
import cv2
import tempfile
from typing import Optional, Tuple

# Import the base class
from __main__ import GlitchEffect


class ChunkManipulationEffect(GlitchEffect):
    """Manipulate video file chunks directly"""
    
    name = "Chunk Manipulation"
    description = "Drastically manipulate raw binary chunks of the video file"
    parameters = {
        "chunk_size": {"type": int, "min": 16, "max": 256, "default": 64},
        "manipulation_type": {"type": "choice", 
                            "options": ["Swap", "Corrupt", "Shift"],
                            "default": "Swap"},
        "corruption_strength": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5},
        "shift_amount": {"type": int, "min": 1, "max": 7, "default": 4},
        "frame_skip": {"type": int, "min": 0, "max": 5, "default": 0},
        "frame_randomness": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5}
    }
    
    def __init__(self):
        super().__init__()
        self._frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process individual frames with chunk manipulation"""
        self._frame_count += 1
        
        # Calculate random frame skip
        base_skip = self.params["frame_skip"]
        randomness = self.params["frame_randomness"]
        random_skip = int(base_skip * (1 + random.uniform(-randomness, randomness)))
        random_skip = max(0, random_skip)  # Ensure we don't get negative skips
        
        # Skip frames if needed
        if self._frame_count % (random_skip + 1) != 0:
            return frame
        
        # Convert frame to bytes
        frame_bytes = frame.tobytes()
        data = bytearray(frame_bytes)
        
        # Get parameters
        chunk_size = self.params["chunk_size"]
        manip_type = self.params["manipulation_type"]
        corruption_strength = self.params["corruption_strength"]
        shift_amount = self.params["shift_amount"]
        
        # Calculate number of chunks
        num_chunks = len(data) // chunk_size
        if num_chunks == 0:
            return frame
        
        # Calculate how many chunks to manipulate
        num_to_manipulate = int(num_chunks * 0.5)  # Fixed at 50% of chunks
        if num_to_manipulate == 0:
            return frame
        
        # Select random chunks to manipulate
        chunks_to_manipulate = random.sample(range(num_chunks), min(num_to_manipulate, num_chunks))
        
        if manip_type == "Swap":
            # Simple swap of adjacent chunks
            for i in range(0, len(chunks_to_manipulate) - 1, 2):
                if i + 1 < len(chunks_to_manipulate):
                    chunk1_idx = chunks_to_manipulate[i]
                    chunk2_idx = chunks_to_manipulate[i + 1]
                    
                    chunk1_start = chunk1_idx * chunk_size
                    chunk1_end = (chunk1_idx + 1) * chunk_size
                    chunk2_start = chunk2_idx * chunk_size
                    chunk2_end = (chunk2_idx + 1) * chunk_size
                    
                    # Swap chunks
                    temp = data[chunk1_start:chunk1_end]
                    data[chunk1_start:chunk1_end] = data[chunk2_start:chunk2_end]
                    data[chunk2_start:chunk2_end] = temp
        
        elif manip_type == "Corrupt":
            # Simple corruption
            for chunk_idx in chunks_to_manipulate:
                chunk_start = chunk_idx * chunk_size
                chunk_end = (chunk_idx + 1) * chunk_size
                
                # Corrupt entire chunks
                if random.random() < corruption_strength:
                    data[chunk_start:chunk_end] = bytes([random.randint(0, 255) for _ in range(chunk_size)])
        
        elif manip_type == "Shift":
            # Simple bit shifting
            for chunk_idx in chunks_to_manipulate:
                chunk_start = chunk_idx * chunk_size
                chunk_end = (chunk_idx + 1) * chunk_size
                
                # Shift bits in chunk
                chunk = bytearray(data[chunk_start:chunk_end])
                for i in range(len(chunk)):
                    chunk[i] = ((chunk[i] << shift_amount) | (chunk[i] >> (8 - shift_amount))) & 0xFF
                data[chunk_start:chunk_end] = chunk
        
        # Convert back to numpy array
        try:
            result = np.frombuffer(data, dtype=frame.dtype).reshape(frame.shape)
            return result
        except:
            return frame


class IDCTCoefficientEffect(GlitchEffect):
    """Manipulate DCT coefficients of the image"""
    
    name = "DCT Coefficient Manipulation"
    description = "Modify discrete cosine transform coefficients"
    parameters = {
        "dc_shift": {"type": int, "min": -50, "max": 50, "default": 0},
        "high_freq_boost": {"type": float, "min": 0.0, "max": 2.0, "default": 1.0},
        "low_freq_suppress": {"type": float, "min": 0.0, "max": 1.0, "default": 0.0},
        "random_coeffs": {"type": float, "min": 0.0, "max": 1.0, "default": 0.0}
    }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        
        # Process each channel separately
        for c in range(3):
            channel = result[:,:,c].astype(np.float32)
            
            # Apply DCT
            dct = cv2.dct(channel)
            
            # Modify DCT coefficients
            # DC coefficient (0,0)
            dct[0,0] += self.params["dc_shift"]
            
            # High frequency boost (higher indices)
            if self.params["high_freq_boost"] != 1.0:
                h, w = dct.shape
                for i in range(h):
                    for j in range(w):
                        # Higher indices = higher frequencies
                        freq_factor = (i + j) / (h + w)
                        if freq_factor > 0.5:  # Only boost high frequencies
                            boost_factor = 1.0 + (self.params["high_freq_boost"] - 1.0) * (freq_factor - 0.5) * 2
                            dct[i,j] *= boost_factor
            
            # Low frequency suppression
            if self.params["low_freq_suppress"] > 0.0:
                h, w = dct.shape
                for i in range(h):
                    for j in range(w):
                        freq_factor = (i + j) / (h + w)
                        if freq_factor < 0.5:  # Only suppress low frequencies
                            suppress_factor = 1.0 - self.params["low_freq_suppress"] * (0.5 - freq_factor) * 2
                            dct[i,j] *= suppress_factor
            
            # Random coefficient modification
            if self.params["random_coeffs"] > 0.0:
                noise = np.random.normal(0, 20 * self.params["random_coeffs"], dct.shape)
                dct += noise
            
            # Apply inverse DCT
            idct = cv2.idct(dct)
            result[:,:,c] = np.clip(idct, 0, 255).astype(np.uint8)
        
        return result


class BitCorruptionEffect(GlitchEffect):
    """Corrupt specific bits in the frame"""
    
    name = "Bit Corruption"
    description = "Drastically reduce bit depth and corrupt image data"
    parameters = {
        "bit_depth": {"type": int, "min": 1, "max": 8, "default": 4},  # Reduce to N bits per channel
        "corruption_strength": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5},
        "channel_mask": {"type": int, "min": 0, "max": 7, "default": 7},  # Bitmask: 1=B, 2=G, 4=R
        "noise_type": {"type": "choice", 
                      "options": ["Random", "Pattern", "Block", "Scanline"],
                      "default": "Random"},
        "invert_colors": {"type": bool, "default": False}
    }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        h, w = result.shape[:2]
        
        bit_depth = self.params["bit_depth"]
        corruption_strength = self.params["corruption_strength"]
        channel_mask = self.params["channel_mask"]
        noise_type = self.params["noise_type"]
        invert_colors = self.params["invert_colors"]
        
        # Calculate bit mask for depth reduction
        bit_mask = (1 << bit_depth) - 1
        shift = 8 - bit_depth
        
        # Process each channel if selected
        for c in range(3):
            if (channel_mask & (1 << c)) == 0:
                continue  # Skip this channel
            
            channel = result[:,:,c]
            
            # Reduce bit depth
            channel = (channel >> shift) << shift
            
            # Add corruption based on noise type
            if noise_type == "Random":
                # Random corruption throughout the image
                noise = np.random.random((h, w)) < corruption_strength
                channel[noise] = np.random.randint(0, 256, size=np.sum(noise))
                
            elif noise_type == "Pattern":
                # Create a repeating pattern
                pattern_size = max(4, int(32 * (1 - corruption_strength)))
                pattern = np.random.randint(0, 256, size=(pattern_size, pattern_size))
                pattern = np.tile(pattern, (h//pattern_size + 1, w//pattern_size + 1))[:h, :w]
                mask = np.random.random((h, w)) < corruption_strength
                channel[mask] = pattern[mask]
                
            elif noise_type == "Block":
                # Create blocky corruption
                block_size = max(4, int(32 * (1 - corruption_strength)))
                for y in range(0, h, block_size):
                    for x in range(0, w, block_size):
                        if np.random.random() < corruption_strength:
                            block_value = np.random.randint(0, 256)
                            channel[y:y+block_size, x:x+block_size] = block_value
                            
            elif noise_type == "Scanline":
                # Create scanline corruption
                for y in range(h):
                    if np.random.random() < corruption_strength:
                        channel[y,:] = np.random.randint(0, 256)
            
            # Apply channel back to result
            result[:,:,c] = channel
            
            # Invert colors if enabled
            if invert_colors:
                result[:,:,c] = 255 - result[:,:,c]
        
        return result


class JpegArtifactEffect(GlitchEffect):
    """Introduce JPEG compression artifacts"""
    
    name = "JPEG Artifacts"
    description = "Add JPEG compression artifacts to the image"
    parameters = {
        "quality": {"type": int, "min": 0, "max": 100, "default": 10},
        "double_compress": {"type": bool, "default": False},
        "chroma_subsampling": {"type": bool, "default": True}
    }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        quality = self.params["quality"]
        double_compress = self.params["double_compress"]
        chroma_subsampling = self.params["chroma_subsampling"]
        
        # Create encode parameters
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        
        # First compression
        _, encoded = cv2.imencode('.jpg', frame, encode_params)
        result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # Second compression if enabled
        if double_compress:
            # Use even lower quality for second pass
            second_quality = max(1, quality // 2)
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, second_quality]
            _, encoded = cv2.imencode('.jpg', result, encode_params)
            result = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        # Emulate chroma subsampling artifacts if enabled
        if chroma_subsampling:
            # Convert to YCrCb
            ycrcb = cv2.cvtColor(result, cv2.COLOR_BGR2YCrCb)
            
            # Downscale chroma channels
            cr = ycrcb[:,:,1]
            cb = ycrcb[:,:,2]
            
            # Downscale by 2x
            cr_small = cv2.resize(cr, (cr.shape[1]//2, cr.shape[0]//2))
            cb_small = cv2.resize(cb, (cb.shape[1]//2, cb.shape[0]//2))
            
            # Upscale back with artifacts
            cr_upscaled = cv2.resize(cr_small, (cr.shape[1], cr.shape[0]))
            cb_upscaled = cv2.resize(cb_small, (cb.shape[1], cb.shape[0]))
            
            # Replace chroma channels
            ycrcb[:,:,1] = cr_upscaled
            ycrcb[:,:,2] = cb_upscaled
            
            # Convert back to BGR
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        
        return result
