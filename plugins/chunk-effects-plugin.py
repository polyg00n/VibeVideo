"""
Chunk Effects Plugin - Individual chunk manipulation effects
Each effect manipulates video data at the binary level
"""
import random
import numpy as np
import cv2

# Import the base class
from __main__ import GlitchEffect


class ChunkSwapEffect(GlitchEffect):
    """Swap chunks of video data"""
    
    name = "Chunk Swap"
    description = "Swap chunks of video data to create glitch effects"
    parameters = {
        "chunk_size": {"type": int, "min": 16, "max": 256, "default": 64},
        "frame_skip": {"type": int, "min": 0, "max": 5, "default": 0},
        "frame_randomness": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5}
    }
    
    def __init__(self):
        super().__init__()
        self._frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self._frame_count += 1
        
        # Calculate random frame skip
        base_skip = self.params["frame_skip"]
        randomness = self.params["frame_randomness"]
        random_skip = int(base_skip * (1 + random.uniform(-randomness, randomness)))
        random_skip = max(0, random_skip)  # Ensure we don't get negative skips
        
        # Skip frames if needed
        if self._frame_count % (random_skip + 1) != 0:
            return frame
        
        # Get parameters
        chunk_size = self.params["chunk_size"]
        
        # Calculate chunk dimensions based on frame size
        height, width = frame.shape[:2]
        chunk_width = min(chunk_size, width)
        chunk_height = min(chunk_size, height)
        
        # Calculate number of chunks in each dimension
        num_chunks_x = width // chunk_width
        num_chunks_y = height // chunk_height
        num_chunks = num_chunks_x * num_chunks_y
        
        if num_chunks == 0:
            return frame
        
        # Calculate how many chunks to manipulate
        min_chunks = max(1, int(num_chunks * 0.1))  # At least 10% of chunks
        num_to_manipulate = max(min_chunks, int(num_chunks * 0.5))  # Fixed at 50% of chunks
        
        # Create a list of all possible chunk positions
        chunk_positions = [(x, y) for x in range(num_chunks_x) for y in range(num_chunks_y)]
        chunks_to_manipulate = random.sample(chunk_positions, min(num_to_manipulate, len(chunk_positions)))
        
        # Swap chunks
        for i in range(0, len(chunks_to_manipulate) - 1, 2):
            if i + 1 < len(chunks_to_manipulate):
                x1, y1 = chunks_to_manipulate[i]
                x2, y2 = chunks_to_manipulate[i + 1]
                
                # Calculate chunk boundaries
                x1_start = x1 * chunk_width
                x1_end = min((x1 + 1) * chunk_width, width)
                y1_start = y1 * chunk_height
                y1_end = min((y1 + 1) * chunk_height, height)
                
                x2_start = x2 * chunk_width
                x2_end = min((x2 + 1) * chunk_width, width)
                y2_start = y2 * chunk_height
                y2_end = min((y2 + 1) * chunk_height, height)
                
                # Store original chunks
                chunk1_orig = frame[y1_start:y1_end, x1_start:x1_end].copy()
                chunk2_orig = frame[y2_start:y2_end, x2_start:x2_end].copy()
                
                # Ensure chunks have the same shape
                if chunk1_orig.shape != chunk2_orig.shape:
                    continue
                
                # Swap chunks
                frame[y1_start:y1_end, x1_start:x1_end] = chunk2_orig
                frame[y2_start:y2_end, x2_start:x2_end] = chunk1_orig
        
        return frame


class ChunkCorruptEffect(GlitchEffect):
    """Corrupt chunks of video data"""
    
    name = "Chunk Corrupt"
    description = "Corrupt chunks of video data with random values"
    parameters = {
        "chunk_size": {"type": int, "min": 16, "max": 256, "default": 64},
        "frame_skip": {"type": int, "min": 0, "max": 5, "default": 0},
        "frame_randomness": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5}
    }
    
    def __init__(self):
        super().__init__()
        self._frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self._frame_count += 1
        
        # Calculate random frame skip
        base_skip = self.params["frame_skip"]
        randomness = self.params["frame_randomness"]
        random_skip = int(base_skip * (1 + random.uniform(-randomness, randomness)))
        random_skip = max(0, random_skip)  # Ensure we don't get negative skips
        
        # Skip frames if needed
        if self._frame_count % (random_skip + 1) != 0:
            return frame
        
        # Get parameters
        chunk_size = self.params["chunk_size"]
        
        # Calculate chunk dimensions based on frame size
        height, width = frame.shape[:2]
        chunk_width = min(chunk_size, width)
        chunk_height = min(chunk_size, height)
        
        # Calculate number of chunks in each dimension
        num_chunks_x = width // chunk_width
        num_chunks_y = height // chunk_height
        num_chunks = num_chunks_x * num_chunks_y
        
        if num_chunks == 0:
            return frame
        
        # Calculate how many chunks to manipulate
        min_chunks = max(1, int(num_chunks * 0.1))  # At least 10% of chunks
        num_to_manipulate = max(min_chunks, int(num_chunks * 0.5))  # Fixed at 50% of chunks
        
        # Create a list of all possible chunk positions
        chunk_positions = [(x, y) for x in range(num_chunks_x) for y in range(num_chunks_y)]
        chunks_to_manipulate = random.sample(chunk_positions, min(num_to_manipulate, len(chunk_positions)))
        
        # Corrupt chunks
        for x, y in chunks_to_manipulate:
            # Calculate chunk boundaries
            x_start = x * chunk_width
            x_end = min((x + 1) * chunk_width, width)
            y_start = y * chunk_height
            y_end = min((y + 1) * chunk_height, height)
            
            # Create corrupted chunk
            corrupted_chunk = np.random.randint(0, 256, frame[y_start:y_end, x_start:x_end].shape, dtype=frame.dtype)
            
            # Apply corrupted values
            frame[y_start:y_end, x_start:x_end] = corrupted_chunk
        
        return frame


class ChunkShiftEffect(GlitchEffect):
    """Bit shift chunks of video data"""
    
    name = "Chunk Shift"
    description = "Bit shift chunks of video data"
    parameters = {
        "chunk_size": {"type": int, "min": 16, "max": 256, "default": 64},
        "shift_amount": {"type": int, "min": 1, "max": 7, "default": 4},
        "frame_skip": {"type": int, "min": 0, "max": 5, "default": 0},
        "frame_randomness": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5}
    }
    
    def __init__(self):
        super().__init__()
        self._frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self._frame_count += 1
        
        # Calculate random frame skip
        base_skip = self.params["frame_skip"]
        randomness = self.params["frame_randomness"]
        random_skip = int(base_skip * (1 + random.uniform(-randomness, randomness)))
        random_skip = max(0, random_skip)  # Ensure we don't get negative skips
        
        # Skip frames if needed
        if self._frame_count % (random_skip + 1) != 0:
            return frame
        
        # Get parameters
        chunk_size = self.params["chunk_size"]
        shift_amount = self.params["shift_amount"]
        
        # Calculate chunk dimensions based on frame size
        height, width = frame.shape[:2]
        chunk_width = min(chunk_size, width)
        chunk_height = min(chunk_size, height)
        
        # Calculate number of chunks in each dimension
        num_chunks_x = width // chunk_width
        num_chunks_y = height // chunk_height
        num_chunks = num_chunks_x * num_chunks_y
        
        if num_chunks == 0:
            return frame
        
        # Calculate how many chunks to manipulate
        min_chunks = max(1, int(num_chunks * 0.1))  # At least 10% of chunks
        num_to_manipulate = max(min_chunks, int(num_chunks * 0.5))  # Fixed at 50% of chunks
        
        # Create a list of all possible chunk positions
        chunk_positions = [(x, y) for x in range(num_chunks_x) for y in range(num_chunks_y)]
        chunks_to_manipulate = random.sample(chunk_positions, min(num_to_manipulate, len(chunk_positions)))
        
        # Shift bits in chunks
        for x, y in chunks_to_manipulate:
            # Calculate chunk boundaries
            x_start = x * chunk_width
            x_end = min((x + 1) * chunk_width, width)
            y_start = y * chunk_height
            y_end = min((y + 1) * chunk_height, height)
            
            # Store original chunk
            original_chunk = frame[y_start:y_end, x_start:x_end].copy()
            
            # Create shifted chunk
            shifted_chunk = original_chunk.copy()
            
            # Apply bit shift to each color channel if present
            if len(shifted_chunk.shape) == 3:  # RGB image
                for c in range(shifted_chunk.shape[2]):
                    shifted_chunk[:, :, c] = ((shifted_chunk[:, :, c] << shift_amount) | 
                                            (shifted_chunk[:, :, c] >> (8 - shift_amount))) & 0xFF
            else:  # Grayscale image
                shifted_chunk = ((shifted_chunk << shift_amount) | 
                               (shifted_chunk >> (8 - shift_amount))) & 0xFF
            
            # Apply shifted values
            frame[y_start:y_end, x_start:x_end] = shifted_chunk
        
        return frame


class ChunkScrambleEffect(GlitchEffect):
    """Scramble chunks of video data"""
    
    name = "Chunk Scramble"
    description = "Randomly scramble bytes within chunks"
    parameters = {
        "chunk_size": {"type": int, "min": 16, "max": 256, "default": 64},
        "frame_skip": {"type": int, "min": 0, "max": 5, "default": 0},
        "frame_randomness": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5}
    }
    
    def __init__(self):
        super().__init__()
        self._frame_count = 0
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self._frame_count += 1
        
        # Calculate random frame skip
        base_skip = self.params["frame_skip"]
        randomness = self.params["frame_randomness"]
        random_skip = int(base_skip * (1 + random.uniform(-randomness, randomness)))
        random_skip = max(0, random_skip)  # Ensure we don't get negative skips
        
        # Skip frames if needed
        if self._frame_count % (random_skip + 1) != 0:
            return frame
        
        # Get parameters
        chunk_size = self.params["chunk_size"]
        
        # Calculate chunk dimensions based on frame size
        height, width = frame.shape[:2]
        chunk_width = min(chunk_size, width)
        chunk_height = min(chunk_size, height)
        
        # Calculate number of chunks in each dimension
        num_chunks_x = width // chunk_width
        num_chunks_y = height // chunk_height
        num_chunks = num_chunks_x * num_chunks_y
        
        if num_chunks == 0:
            return frame
        
        # Calculate how many chunks to manipulate
        min_chunks = max(1, int(num_chunks * 0.1))  # At least 10% of chunks
        num_to_manipulate = max(min_chunks, int(num_chunks * 0.5))  # Fixed at 50% of chunks
        
        # Create a list of all possible chunk positions
        chunk_positions = [(x, y) for x in range(num_chunks_x) for y in range(num_chunks_y)]
        chunks_to_manipulate = random.sample(chunk_positions, min(num_to_manipulate, len(chunk_positions)))
        
        # Scramble chunks
        for x, y in chunks_to_manipulate:
            # Calculate chunk boundaries
            x_start = x * chunk_width
            x_end = min((x + 1) * chunk_width, width)
            y_start = y * chunk_height
            y_end = min((y + 1) * chunk_height, height)
            
            # Store original chunk
            original_chunk = frame[y_start:y_end, x_start:x_end].copy()
            
            # Create scrambled chunk
            scrambled_chunk = original_chunk.copy()
            shape = scrambled_chunk.shape
            if len(shape) == 3:  # RGB image
                # Scramble each color channel separately
                for c in range(shape[2]):
                    channel = scrambled_chunk[:, :, c]
                    channel_flat = channel.reshape(-1)
                    np.random.shuffle(channel_flat)
                    scrambled_chunk[:, :, c] = channel_flat.reshape(channel.shape)
            else:  # Grayscale image
                scrambled_flat = scrambled_chunk.reshape(-1)
                np.random.shuffle(scrambled_flat)
                scrambled_chunk = scrambled_flat.reshape(shape)
            
            # Apply scrambled values
            frame[y_start:y_end, x_start:x_end] = scrambled_chunk
        
        return frame


class ChunkPatternEffect(GlitchEffect):
    """Apply patterns to chunks of video data"""
    
    name = "Chunk Pattern"
    description = "Apply repeating patterns to chunks of video data"
    parameters = {
        "chunk_size": {"type": int, "min": 16, "max": 256, "default": 64},
        "pattern_size": {"type": int, "min": 1, "max": 8, "default": 4},
        "pattern_scale": {"type": int, "min": 1, "max": 8, "default": 1},
        "frame_skip": {"type": int, "min": 0, "max": 5, "default": 0},
        "frame_randomness": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5}
    }
    
    def __init__(self):
        super().__init__()
        self._frame_count = 0
        self._pattern_buffer = None
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        self._frame_count += 1
        
        # Calculate random frame skip
        base_skip = self.params["frame_skip"]
        randomness = self.params["frame_randomness"]
        random_skip = int(base_skip * (1 + random.uniform(-randomness, randomness)))
        random_skip = max(0, random_skip)  # Ensure we don't get negative skips
        
        # Skip frames if needed
        if self._frame_count % (random_skip + 1) != 0:
            return frame
        
        # Get parameters
        chunk_size = self.params["chunk_size"]
        pattern_size = self.params["pattern_size"]
        pattern_scale = self.params["pattern_scale"]
        
        # Calculate chunk dimensions based on frame size
        height, width = frame.shape[:2]
        chunk_width = min(chunk_size, width)
        chunk_height = min(chunk_size, height)
        
        # Calculate number of chunks in each dimension
        num_chunks_x = width // chunk_width
        num_chunks_y = height // chunk_height
        num_chunks = num_chunks_x * num_chunks_y
        
        if num_chunks == 0:
            return frame
        
        # Create pattern buffer if needed or if size changed
        if (self._pattern_buffer is None or 
            self._pattern_buffer.shape[0] != chunk_height or 
            self._pattern_buffer.shape[1] != chunk_width):
            # Create a smaller base pattern
            base_pattern = np.random.randint(0, 256, 
                (chunk_height // pattern_scale, chunk_width // pattern_scale, 
                 3 if len(frame.shape) == 3 else 1), 
                dtype=frame.dtype)
            
            # Scale up the pattern
            self._pattern_buffer = cv2.resize(base_pattern, 
                (chunk_width, chunk_height), 
                interpolation=cv2.INTER_NEAREST)
        
        # Calculate how many chunks to manipulate
        min_chunks = max(1, int(num_chunks * 0.1))  # At least 10% of chunks
        num_to_manipulate = max(min_chunks, int(num_chunks * 0.5))  # Fixed at 50% of chunks
        
        # Create a list of all possible chunk positions
        chunk_positions = [(x, y) for x in range(num_chunks_x) for y in range(num_chunks_y)]
        chunks_to_manipulate = random.sample(chunk_positions, min(num_to_manipulate, len(chunk_positions)))
        
        # Apply patterns to chunks
        for x, y in chunks_to_manipulate:
            # Calculate chunk boundaries
            x_start = x * chunk_width
            x_end = min((x + 1) * chunk_width, width)
            y_start = y * chunk_height
            y_end = min((y + 1) * chunk_height, height)
            
            # Get the actual chunk size
            actual_width = x_end - x_start
            actual_height = y_end - y_start
            
            # Create pattern chunk of the correct size
            pattern_chunk = self._pattern_buffer[:actual_height, :actual_width].copy()
            
            # Apply pattern
            frame[y_start:y_end, x_start:x_end] = pattern_chunk
        
        return frame 