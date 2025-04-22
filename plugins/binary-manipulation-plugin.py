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
    description = "Manipulate raw binary chunks of the video file"
    parameters = {
        "chunk_size": {"type": int, "min": 512, "max": 8192, "default": 2048},
        "manipulation_type": {"type": int, "min": 0, "max": 3, "default": 0},  # 0: swap, 1: repeat, 2: remove, 3: reverse
        "intensity": {"type": float, "min": 0.0, "max": 1.0, "default": 0.2},
        "preserve_header": {"type": bool, "default": True}
    }
    
    def __init__(self):
        super().__init__()
        self.temp_file = None
        self.temp_filename = None
        self.cap = None
    
    def _manipulate_file(self, original_file: str) -> Optional[str]:
        """Create a manipulated copy of the original file"""
        # Create a temporary file
        temp_fd, temp_path = tempfile.mkstemp(suffix='.mp4')
        os.close(temp_fd)
        
        try:
            # Read the original file in binary mode
            with open(original_file, 'rb') as f:
                data = bytearray(f.read())
            
            chunk_size = self.params["chunk_size"]
            manip_type = self.params["manipulation_type"]
            intensity = self.params["intensity"]
            preserve_header = self.params["preserve_header"]
            
            # Determine how many chunks to process
            num_chunks = len(data) // chunk_size
            
            # Skip header if preserve_header is True (usually first chunk)
            start_chunk = 1 if preserve_header else 0
            
            # Calculate how many chunks to manipulate based on intensity
            num_to_manipulate = int(num_chunks * intensity)
            
            # Select random chunks to manipulate
            chunks_to_manipulate = random.sample(range(start_chunk, num_chunks), min(num_to_manipulate, num_chunks - start_chunk))
            
            # Perform manipulation
            if manip_type == 0:  # Swap chunks
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
            
            elif manip_type == 1:  # Repeat chunks
                for chunk_idx in chunks_to_manipulate:
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = (chunk_idx + 1) * chunk_size
                    
                    # Repeat the chunk (replace next chunk with this one)
                    if chunk_idx < num_chunks - 1:
                        next_start = (chunk_idx + 1) * chunk_size
                        next_end = (chunk_idx + 2) * chunk_size
                        data[next_start:next_end] = data[chunk_start:chunk_end]
            
            elif manip_type == 2:  # Remove chunks
                # We'll create a new bytearray excluding chunks to remove
                new_data = bytearray()
                chunks_to_remove = set(chunks_to_manipulate)
                
                for i in range(num_chunks):
                    if i not in chunks_to_remove:
                        chunk_start = i * chunk_size
                        chunk_end = (i + 1) * chunk_size
                        new_data.extend(data[chunk_start:chunk_end])
                
                # Add any remaining data
                new_data.extend(data[num_chunks * chunk_size:])
                data = new_data
            
            elif manip_type == 3:  # Reverse chunks
                for chunk_idx in chunks_to_manipulate:
                    chunk_start = chunk_idx * chunk_size
                    chunk_end = (chunk_idx + 1) * chunk_size
                    
                    # Reverse the bytes in the chunk
                    chunk = data[chunk_start:chunk_end]
                    data[chunk_start:chunk_end] = chunk[::-1]
            
            # Write to temporary file
            with open(temp_path, 'wb') as f:
                f.write(data)
            
            return temp_path
        
        except Exception as e:
            print(f"Error manipulating file: {e}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # This effect doesn't process individual frames directly
        # Instead, it processes the entire file
        # For demonstration, we'll just return the frame
        return frame
    
    def pre_process_video(self, original_file: str) -> Optional[Tuple[cv2.VideoCapture, str]]:
        """Pre-process the video file by manipulating its binary data"""
        # Clean up previous temp files if they exist
        if self.temp_filename and os.path.exists(self.temp_filename):
            try:
                os.remove(self.temp_filename)
            except:
                pass
        
        # Create new manipulated file
        self.temp_filename = self._manipulate_file(original_file)
        
        if self.temp_filename:
            # Open the manipulated file with OpenCV
            self.cap = cv2.VideoCapture(self.temp_filename)
            if self.cap.isOpened():
                return self.cap, self.temp_filename
        
        return None
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
            self.cap = None
        
        if self.temp_filename and os.path.exists(self.temp_filename):
            try:
                os.remove(self.temp_filename)
                self.temp_filename = None
            except:
                pass


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
    description = "Flip specific bits in the image data"
    parameters = {
        "bit_position": {"type": int, "min": 0, "max": 7, "default": 4},  # Which bit to corrupt (0-7)
        "probability": {"type": float, "min": 0.0, "max": 1.0, "default": 0.1},
        "channel_mask": {"type": int, "min": 0, "max": 7, "default": 7},  # Bitmask: 1=B, 2=G, 4=R
        "region_specific": {"type": bool, "default": False}
    }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        h, w = result.shape[:2]
        
        bit_pos = self.params["bit_position"]
        prob = self.params["probability"]
        channel_mask = self.params["channel_mask"]
        region_specific = self.params["region_specific"]
        
        # Create bit mask
        bit_mask = 1 << bit_pos
        
        # Process each channel if selected
        for c in range(3):
            if (channel_mask & (1 << c)) == 0:
                continue  # Skip this channel
            
            if region_specific:
                # Corrupt specific regions
                x_center = w // 2
                y_center = h // 2
                radius = min(w, h) // 4
                
                for i in range(h):
                    for j in range(w):
                        # Distance from center
                        dist = np.sqrt((i - y_center)**2 + (j - x_center)**2)
                        
                        # Only corrupt within circle
                        if dist < radius and random.random() < prob:
                            # Flip the bit
                            result[i,j,c] ^= bit_mask
            else:
                # Random corruption throughout the image
                random_mask = np.random.random((h, w)) < prob
                
                # Apply the bit flip to selected pixels
                result[:,:,c][random_mask] ^= bit_mask
        
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
