"""
P-Frame Insert Plugin for Video Glitch Player
Inserts P-frames from a reference video into the main video

This plugin creates glitch effects by:
1. Loading a reference video
2. Extracting its P-frames (predicted frames)
3. Inserting these P-frames into the main video
4. Optionally removing I-frames from the main video

The effect creates interesting motion artifacts and temporal glitches by
mixing motion prediction data between different videos.
"""
import cv2
import numpy as np
from __main__ import GlitchEffect
import os

class PFrameInsertEffect(GlitchEffect):
    """Inserts P-frames from a reference video into the main video"""
    
    name = "P-Frame Insert"
    description = "Inserts P-frames from a reference video into the main video"
    
    # ===== Parameters =====
    # reference_video: Path to the video whose P-frames will be used
    #   - The video must be in a format that OpenCV can read
    #   - P-frames will be extracted and inserted into the main video
    #
    # p_frame_interval: How often to insert P-frames (in frames)
    #   - Lower values create more frequent glitches
    #   - Higher values create more subtle effects
    #
    # remove_i_frames: Whether to remove I-frames from the main video
    #   - When enabled, removes keyframes to create more dramatic glitches
    #   - When disabled, maintains some original video structure
    #
    # blend_amount: How much to blend the inserted P-frames (0.0 to 1.0)
    #   - 0.0: Complete replacement with P-frames
    #   - 1.0: Full blending with original frames
    parameters = {
        "reference_video": {"type": str, "default": ""},
        "p_frame_interval": {"type": int, "min": 1, "max": 30, "default": 5},
        "remove_i_frames": {"type": bool, "default": True},
        "blend_amount": {"type": float, "min": 0.0, "max": 10.0, "default": 1}
    }
    
    def __init__(self):
        super().__init__()
        self.reference_frames = []
        self.current_frame_index = 0
        self.reference_cap = None
        self.frame_buffer = []  # Buffer for frame history
        self.max_buffer_size = 10
    
    def initialize(self):
        """Initialize the effect by loading the reference video"""
        video_path = self.params["reference_video"]
        if not video_path or not os.path.exists(video_path):
            raise ValueError("Reference video path is invalid or not provided")
        
        # Open the reference video
        self.reference_cap = cv2.VideoCapture(video_path)
        if not self.reference_cap.isOpened():
            raise ValueError("Could not open reference video")
        
        # Extract P-frames from the reference video
        self._extract_p_frames()
    
    def _extract_p_frames(self):
        """Extract P-frames from the reference video"""
        while True:
            ret, frame = self.reference_cap.read()
            if not ret:
                break
            
            # Convert frame to grayscale for motion detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect motion between frames (simple P-frame detection)
            if len(self.reference_frames) > 0:
                prev_gray = cv2.cvtColor(self.reference_frames[-1], cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(gray, prev_gray)
                motion = np.mean(diff)
                
                # If significant motion detected, consider it a P-frame
                if motion > 10:  # Threshold for motion detection
                    self.reference_frames.append(frame)
            else:
                self.reference_frames.append(frame)
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process the current frame by inserting P-frames from reference video"""
        if not self.reference_frames:
            return frame
        
        # Add current frame to buffer
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.max_buffer_size:
            self.frame_buffer.pop(0)
        
        # Get parameters
        interval = self.params["p_frame_interval"]
        remove_i_frames = self.params["remove_i_frames"]
        blend_amount = self.params["blend_amount"]
        
        # Get reference frames
        ref_frame = self.reference_frames[self.current_frame_index % len(self.reference_frames)]
        next_ref_frame = self.reference_frames[(self.current_frame_index + 1) % len(self.reference_frames)]
        
        # Create motion vectors
        gray_current = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_ref = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate optical flow
        flow = cv2.calcOpticalFlowFarneback(
            gray_ref, gray_current,
            None, 0.8, 5, 25, 7, 7, 1.5, 0
        )
        
        # Apply motion vectors
        h, w = flow.shape[:2]
        map_x = np.tile(np.arange(w), (h, 1))
        map_y = np.tile(np.arange(h), (w, 1)).T
        
        map_x = map_x + flow[..., 0] * 2.0
        map_y = map_y + flow[..., 1] * 2.0
        
        # Create warped reference frame
        warped_ref = cv2.remap(ref_frame, map_x.astype(np.float32), map_y.astype(np.float32),
                              cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        # Start with reference frame as base
        result = warped_ref.copy()
        
        # Blend with original frame
        result = cv2.addWeighted(result, blend_amount, frame, 1 - blend_amount, 0)
        
        # Add block artifacts
        block_size = 32
        for y in range(0, h, block_size):
            for x in range(0, w, block_size):
                if np.random.random() < 0.5:
                    result[y:y+block_size, x:x+block_size] = \
                        ref_frame[y:y+block_size, x:x+block_size]
        
        # Remove I-frames if enabled
        if remove_i_frames and self.current_frame_index % (interval * 2) == 0:
            # Use previous frame from buffer
            if len(self.frame_buffer) > 1:
                prev_frame = self.frame_buffer[-2]
                result = cv2.addWeighted(result, 0.7, prev_frame, 0.3, 0)
            
            # Add more block artifacts
            block_size = 64
            for y in range(0, h, block_size):
                for x in range(0, w, block_size):
                    if np.random.random() < 0.7:
                        result[y:y+block_size, x:x+block_size] = \
                            ref_frame[y:y+block_size, x:x+block_size]
        
        # Add noise
        noise = np.random.normal(0, 15, result.shape).astype(np.uint8)
        result = cv2.addWeighted(result, 0.9, noise, 0.1, 0)
        
        self.current_frame_index += 1
        return result
    
    def cleanup(self):
        """Clean up resources when effect is removed"""
        if self.reference_cap is not None:
            self.reference_cap.release()
        self.reference_frames = []
        self.current_frame_index = 0
        self.frame_buffer = [] 