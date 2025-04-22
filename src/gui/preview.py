"""
Video preview window with optimized frame display.
"""
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image, ImageTk
from typing import Optional, Tuple, Callable
import threading
from queue import Queue
import time

class VideoPreview:
    """Video preview widget with optimized frame display"""
    
    def __init__(self, parent: tk.Widget):
        self.parent = parent
        self.canvas = tk.Canvas(parent, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Frame display
        self._current_frame: Optional[np.ndarray] = None
        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._display_size: Tuple[int, int] = (0, 0)
        
        # Frame queue for async updates
        self._frame_queue: Queue = Queue(maxsize=1)
        self._update_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Performance tracking
        self._last_update_time = 0
        self._update_interval = 1/30  # Target 30 FPS
        
        # Bind resize event
        self.canvas.bind("<Configure>", self._on_resize)
    
    def update_frame(self, frame: np.ndarray) -> None:
        """Update the preview with a new frame"""
        if frame is None:
            return
            
        # Put frame in queue (replacing any existing frame)
        try:
            self._frame_queue.get_nowait()  # Clear old frame
        except:
            pass
        self._frame_queue.put(frame)
        
        # Start update thread if not running
        if self._update_thread is None or not self._update_thread.is_alive():
            self._stop_event.clear()
            self._update_thread = threading.Thread(target=self._update_loop)
            self._update_thread.daemon = True
            self._update_thread.start()
    
    def _update_loop(self):
        """Background thread for updating the display"""
        while not self._stop_event.is_set():
            try:
                # Get next frame from queue
                frame = self._frame_queue.get(timeout=0.1)
                
                # Check if enough time has passed since last update
                current_time = time.time()
                if current_time - self._last_update_time < self._update_interval:
                    continue
                
                # Convert frame to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image
                img = Image.fromarray(rgb_frame)
                
                # Resize to fit canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    img_ratio = img.width / img.height
                    canvas_ratio = canvas_width / canvas_height
                    
                    if img_ratio > canvas_ratio:
                        new_width = canvas_width
                        new_height = int(canvas_width / img_ratio)
                    else:
                        new_height = canvas_height
                        new_width = int(canvas_height * img_ratio)
                    
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Convert to PhotoImage
                photo = ImageTk.PhotoImage(img)
                
                # Update display in main thread
                self.parent.after(0, lambda: self._update_display(photo))
                
                self._last_update_time = current_time
                
            except Exception as e:
                print(f"Preview update error: {e}")
                continue
    
    def _update_display(self, photo: ImageTk.PhotoImage):
        """Update the canvas display"""
        # Keep reference to avoid garbage collection
        self._photo_image = photo
        
        # Clear canvas
        self.canvas.delete("all")
        
        # Draw image
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        self.canvas.create_image(
            canvas_width // 2,
            canvas_height // 2,
            anchor=tk.CENTER,
            image=self._photo_image
        )
    
    def _on_resize(self, event):
        """Handle canvas resize"""
        if self._current_frame is not None:
            self.update_frame(self._current_frame)
    
    def clear(self):
        """Clear the preview"""
        self._stop_event.set()
        if self._update_thread is not None:
            self._update_thread.join(timeout=1.0)
            self._update_thread = None
            
        self.canvas.delete("all")
        self._current_frame = None
        self._photo_image = None
    
    def destroy(self):
        """Clean up resources"""
        self.clear()
        self.canvas.destroy() 