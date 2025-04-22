"""
Example plugin for the Video Glitch Player
Save this file to a 'plugins' directory where the main script is located
"""
import cv2
import numpy as np
from random import randint
import math

# Import the base class from parent directory
# Note: In a real implementation, you'd need to ensure this import works correctly
# For demo purposes, we're assuming the main module is properly installed or in the PYTHONPATH
from __main__ import GlitchEffect


class PixelSortEffect(GlitchEffect):
    """Sort pixels based on brightness"""
    
    name = "Pixel Sort"
    description = "Sort pixels in rows based on brightness"
    parameters = {
        "threshold": {"type": int, "min": 0, "max": 255, "default": 128},
        "sort_direction": {"type": int, "min": 0, "max": 3, "default": 0},  # 0: horizontal, 1: vertical, 2: both, 3: diagonal
        "invert": {"type": bool, "default": False}
    }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        
        threshold = self.params["threshold"]
        direction = self.params["sort_direction"]
        invert = self.params["invert"]
        
        # Horizontal sorting (rows)
        if direction == 0 or direction == 2:
            for i in range(result.shape[0]):
                # Find pixels above threshold
                if invert:
                    indices = np.where(gray[i,:] < threshold)[0]
                else:
                    indices = np.where(gray[i,:] > threshold)[0]
                
                if len(indices) > 0:
                    # Sort these pixels by intensity
                    result[i,indices] = result[i,indices][np.argsort(gray[i,indices])]
        
        # Vertical sorting (columns)
        if direction == 1 or direction == 2:
            for j in range(result.shape[1]):
                if invert:
                    indices = np.where(gray[:,j] < threshold)[0]
                else:
                    indices = np.where(gray[:,j] > threshold)[0]
                
                if len(indices) > 0:
                    result[indices,j] = result[indices,j][np.argsort(gray[indices,j])]
        
        # Diagonal sorting (experimental)
        if direction == 3:
            h, w = result.shape[:2]
            # Process diagonals from top-left to bottom-right
            for offset in range(-h + 1, w):
                # Get diagonal indices
                indices = np.array([(i, i + offset) for i in range(max(0, -offset), min(h, w - offset))])
                if len(indices) > 0:
                    i_indices, j_indices = zip(*indices)
                    i_indices, j_indices = np.array(i_indices), np.array(j_indices)
                    
                    # Get mask for current diagonal
                    if invert:
                        mask = gray[i_indices, j_indices] < threshold
                    else:
                        mask = gray[i_indices, j_indices] > threshold
                    
                    # Get sortable indices
                    sort_i = i_indices[mask]
                    sort_j = j_indices[mask]
                    
                    if len(sort_i) > 0:
                        # Sort by intensity
                        sort_order = np.argsort(gray[sort_i, sort_j])
                        result[sort_i, sort_j] = result[sort_i, sort_j][sort_order]
        
        return result


class BlockShuffleEffect(GlitchEffect):
    """Shuffle blocks of the image"""
    
    name = "Block Shuffle"
    description = "Divide image into blocks and shuffle them"
    parameters = {
        "block_size": {"type": int, "min": 8, "max": 128, "default": 16},
        "shuffle_percentage": {"type": int, "min": 0, "max": 100, "default": 50},
        "preserve_regions": {"type": bool, "default": False}
    }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        h, w = result.shape[:2]
        block_size = self.params["block_size"]
        shuffle_pct = self.params["shuffle_percentage"] / 100.0
        preserve = self.params["preserve_regions"]
        
        # Calculate number of blocks
        n_blocks_h = h // block_size
        n_blocks_w = w // block_size
        
        # Create block positions
        blocks = []
        for i in range(n_blocks_h):
            for j in range(n_blocks_w):
                blocks.append((i, j))
        
        # Shuffle a percentage of blocks
        n_shuffle = int(len(blocks) * shuffle_pct)
        shuffle_indices = np.random.choice(len(blocks), n_shuffle, replace=False)
        shuffle_blocks = [blocks[i] for i in shuffle_indices]
        
        # Create a copy of the shuffled blocks and shuffle them
        target_blocks = shuffle_blocks.copy()
        np.random.shuffle(target_blocks)
        
        # Perform the shuffle
        temp_result = result.copy()
        
        for (i1, j1), (i2, j2) in zip(shuffle_blocks, target_blocks):
            # Block coordinates
            y1, x1 = i1 * block_size, j1 * block_size
            y2, x2 = i2 * block_size, j2 * block_size
            
            # Extract blocks
            block1 = result[y1:y1+block_size, x1:x1+block_size].copy()
            block2 = result[y2:y2+block_size, x2:x2+block_size].copy()
            
            # Swap blocks
            temp_result[y1:y1+block_size, x1:x1+block_size] = block2
            
            if not preserve:
                temp_result[y2:y2+block_size, x2:x2+block_size] = block1
        
        return temp_result


class FrameEchoEffect(GlitchEffect):
    """Create trailing/echo effects with frames"""
    
    name = "Frame Echo"
    description = "Blend current frame with previous frames"
    parameters = {
        "echo_strength": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5},
        "echo_duration": {"type": int, "min": 1, "max": 10, "default": 3},
        "direction": {"type": int, "min": 0, "max": 1, "default": 0}  # 0: past echo, 1: future echo (requires frame buffer)
    }
    
    def __init__(self):
        super().__init__()
        self.frame_buffer = []
        self.max_buffer_size = 10  # Maximum buffer size
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        
        # Add current frame to buffer
        self.frame_buffer.append(frame.copy())
        # Limit buffer size
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        # Not enough frames in buffer yet
        if len(self.frame_buffer) < 2:
            return result
        
        echo_strength = self.params["echo_strength"]
        echo_duration = min(self.params["echo_duration"], len(self.frame_buffer) - 1)
        direction = self.params["direction"]
        
        # Calculate echo effect
        if direction == 0:  # Past echo
            for i in range(1, echo_duration + 1):
                if i < len(self.frame_buffer):
                    # Get frame from buffer (most recent is at the end)
                    past_frame = self.frame_buffer[-i - 1]
                    # Decrease strength with distance
                    current_strength = echo_strength * (echo_duration - i + 1) / echo_duration
                    # Blend with result
                    result = cv2.addWeighted(result, 1.0, past_frame, current_strength, 0)
        
        return result


class StaticNoiseEffect(GlitchEffect):
    """Add static noise to the image"""
    
    name = "Static Noise"
    description = "Add random noise to the image"
    parameters = {
        "noise_amount": {"type": int, "min": 0, "max": 255, "default": 50},
        "noise_type": {"type": int, "min": 0, "max": 2, "default": 0},  # 0: uniform, 1: salt & pepper, 2: gaussian
        "colored": {"type": bool, "default": True}
    }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        h, w = result.shape[:2]
        
        noise_amount = self.params["noise_amount"]
        noise_type = self.params["noise_type"]
        colored = self.params["colored"]
        
        if noise_type == 0:  # Uniform noise
            if colored:
                noise = np.random.randint(-noise_amount, noise_amount + 1, result.shape, dtype=np.int16)
                result = np.clip(result.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            else:
                noise = np.random.randint(-noise_amount, noise_amount + 1, (h, w), dtype=np.int16)
                for c in range(3):
                    result[:,:,c] = np.clip(result[:,:,c].astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        elif noise_type == 1:  # Salt & pepper
            # Salt (white) pixels
            salt_pct = noise_amount / 255.0
            salt_mask = np.random.random((h, w)) < salt_pct/2
            if colored:
                result[salt_mask] = 255
            else:
                for c in range(3):
                    result[:,:,c][salt_mask] = 255
            
            # Pepper (black) pixels
            pepper_mask = np.random.random((h, w)) < salt_pct/2
            if colored:
                result[pepper_mask] = 0
            else:
                for c in range(3):
                    result[:,:,c][pepper_mask] = 0
        
        elif noise_type == 2:  # Gaussian noise
            sigma = noise_amount / 255.0 * 50  # Scale to reasonable range
            if colored:
                noise = np.random.normal(0, sigma, result.shape)
                result = np.clip(result.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            else:
                noise = np.random.normal(0, sigma, (h, w))
                for c in range(3):
                    result[:,:,c] = np.clip(result[:,:,c].astype(np.float32) + noise, 0, 255).astype(np.uint8)
        
        return result


class ScanlineEffect(GlitchEffect):
    """Add horizontal scanlines to the image"""
    
    name = "Scanlines"
    description = "Add horizontal scanlines like CRT displays"
    parameters = {
        "line_thickness": {"type": int, "min": 1, "max": 10, "default": 2},
        "line_spacing": {"type": int, "min": 1, "max": 20, "default": 4},
        "line_opacity": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5},
        "rolling": {"type": bool, "default": False}
    }
    
    def __init__(self):
        super().__init__()
        self._frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        result = frame.copy()
        h, w = result.shape[:2]
        
        thickness = self.params["line_thickness"]
        spacing = self.params["line_spacing"]
        opacity = self.params["line_opacity"]
        rolling = self.params["rolling"]
        
        # Create a mask for scanlines
        mask = np.ones((h, w), dtype=np.float32)
        
        if rolling:
            # Rolling scanlines (move with each frame)
            self._frame_count = (self._frame_count + 1) % (thickness + spacing)
            offset = self._frame_count
        else:
            offset = 0
        
        # Draw scanlines on the mask
        for y in range(offset, h, thickness + spacing):
            if y + thickness <= h:
                mask[y:y+thickness, :] = 1.0 - opacity
        
        # Apply mask to each channel
        for c in range(3):
            result[:,:,c] = (result[:,:,c] * mask).astype(np.uint8)
        
        return result
