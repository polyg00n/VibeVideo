"""
Video processing core with multiprocessing support.
"""
import cv2
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import threading
from queue import Queue
import time
from .effects import EffectChain

class VideoProcessor:
    """Core video processing engine with multiprocessing support"""
    
    def __init__(self, source_path: Optional[str] = None):
        self.source_path: Optional[str] = source_path
        self.cap: Optional[cv2.VideoCapture] = None
        self.width: int = 0
        self.height: int = 0
        self.fps: float = 0
        self.frame_count: int = 0
        self.effect_chain = EffectChain()
        
        # Threading and multiprocessing
        self._processing_lock = threading.Lock()
        self._frame_queue: Queue = Queue(maxsize=100)  # Buffer for processed frames
        self._stop_event = threading.Event()
        self._processing_thread: Optional[threading.Thread] = None
        self._num_workers = max(1, mp.cpu_count() - 1)  # Leave one core for UI
        
        if source_path:
            self.load_video(source_path)
    
    def load_video(self, path: str) -> bool:
        """Load a video file"""
        with self._processing_lock:
            if self.cap:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(path)
            if not self.cap.isOpened():
                return False
                
            self.source_path = path
            self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Clear any existing processing
            self._stop_processing()
            self.effect_chain.clear_cache()
            
            return True
    
    def get_frame(self, position: int) -> Tuple[bool, Optional[np.ndarray]]:
        """Get a specific frame from the video"""
        with self._processing_lock:
            if not self.cap or not self.cap.isOpened():
                return False, None
                
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
            ret, frame = self.cap.read()
            return ret, frame
    
    def process_frame(self, frame: np.ndarray, frame_index: int = -1) -> np.ndarray:
        """Process a frame with the current effect chain"""
        return self.effect_chain.process_frame(frame, frame_index)
    
    def _process_frame_worker(self, frame_data: Tuple[int, np.ndarray]) -> Tuple[int, np.ndarray]:
        """Worker function for processing frames"""
        frame_index, frame = frame_data
        processed = self.process_frame(frame, frame_index)
        return frame_index, processed
    
    def _processing_loop(self):
        """Background thread for processing frames"""
        with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
            while not self._stop_event.is_set():
                try:
                    # Get next frame to process
                    frame_data = self._frame_queue.get(timeout=0.1)
                    if frame_data is None:
                        break
                        
                    # Process frame
                    frame_index, processed = self._process_frame_worker(frame_data)
                    
                    # Store in cache
                    self.effect_chain._frame_cache[frame_index] = processed
                    
                except Exception as e:
                    print(f"Error processing frame: {e}")
                    continue
    
    def start_processing(self):
        """Start background processing of frames"""
        if self._processing_thread is not None and self._processing_thread.is_alive():
            return
            
        self._stop_event.clear()
        self._processing_thread = threading.Thread(target=self._processing_loop)
        self._processing_thread.daemon = True
        self._processing_thread.start()
    
    def _stop_processing(self):
        """Stop background processing"""
        self._stop_event.set()
        if self._processing_thread is not None:
            self._processing_thread.join(timeout=1.0)
            self._processing_thread = None
    
    def preload_frames(self, start_frame: int = 0, num_frames: int = 10):
        """Preload frames into the processing queue"""
        with self._processing_lock:
            if not self.cap or not self.cap.isOpened():
                return
                
            # Clear existing queue
            while not self._frame_queue.empty():
                self._frame_queue.get_nowait()
                
            # Queue frames for processing
            for i in range(start_frame, min(start_frame + num_frames, self.frame_count)):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = self.cap.read()
                if ret:
                    self._frame_queue.put((i, frame))
    
    def process_video(self, output_path: str, progress_callback=None) -> bool:
        """Process the entire video and save to output path"""
        with self._processing_lock:
            if not self.cap or not self.cap.isOpened():
                return False
                
            # Reset to beginning
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
            
            # Process frames in parallel
            with ThreadPoolExecutor(max_workers=self._num_workers) as executor:
                futures = []
                frame_index = 0
                
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                        
                    # Submit frame for processing
                    future = executor.submit(self._process_frame_worker, (frame_index, frame))
                    futures.append((frame_index, future))
                    
                    # Write processed frames in order
                    while futures and futures[0][1].done():
                        idx, future = futures.pop(0)
                        _, processed = future.result()
                        out.write(processed)
                        
                        # Update progress
                        if progress_callback:
                            progress_callback(idx / self.frame_count)
                    
                    frame_index += 1
                
                # Wait for remaining frames
                for idx, future in futures:
                    _, processed = future.result()
                    out.write(processed)
                    if progress_callback:
                        progress_callback(idx / self.frame_count)
            
            out.release()
            return True
    
    def release(self):
        """Release resources"""
        with self._processing_lock:
            self._stop_processing()
            if self.cap:
                self.cap.release()
            self.cap = None
            self.effect_chain.clear_cache() 