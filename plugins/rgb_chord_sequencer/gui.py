"""
GUI components for RGB Chord Sequencer
"""
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from typing import Tuple, Callable

class ProgressDialog:
    def __init__(self, parent: tk.Tk, title: str, maximum: int):
        self.dialog = tk.Toplevel(parent)
        self.dialog.title(title)
        self.dialog.transient(parent)
        self.dialog.grab_set()
        
        self.progress = ttk.Progressbar(
            self.dialog, 
            length=300, 
            mode='determinate',
            maximum=maximum
        )
        self.progress.pack(padx=10, pady=10)
        
        self.label = ttk.Label(self.dialog, text="0%")
        self.label.pack(pady=5)
        
    def update(self, value: int, text: str = None):
        self.progress['value'] = value
        if text:
            self.label['text'] = text
        else:
            self.label['text'] = f"{int(value/self.progress['maximum']*100)}%"
        self.dialog.update()
        
    def close(self):
        self.dialog.destroy()

def draw_grid(frame: np.ndarray, 
              h_div: int, 
              v_div: int, 
              tempo: int, 
              note_duration: str) -> np.ndarray:
    """Draw grid overlay on frame"""
    height, width = frame.shape[:2]
    overlay = frame.copy()
    
    # Draw vertical lines
    for i in range(h_div + 1):
        x = (width * i) // h_div
        cv2.line(overlay, (x, 0), (x, height), (255, 255, 255), 1)
    
    # Draw horizontal lines
    for i in range(v_div + 1):
        y = (height * i) // v_div
        cv2.line(overlay, (0, y), (width, y), (255, 255, 255), 1)
    
    # Draw block numbers
    font_scale = min(width, height) / (1000 * max(h_div, v_div))
    for y in range(v_div):
        for x in range(h_div):
            x1 = (x * width) // h_div
            y1 = (y * height) // v_div
            block_num = y * h_div + x
            cv2.putText(overlay, 
                       str(block_num), 
                       (x1 + 5, y1 + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 
                       font_scale, (255, 255, 255), 1)
    
    # Draw info
    cv2.putText(overlay, 
               f"Tempo: {tempo} BPM", 
               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay, 
               f"Divisions: {h_div}x{v_div}", 
               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(overlay,
               f"Note: {note_duration}", 
               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Blend overlay
    alpha = 0.7
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    return frame 