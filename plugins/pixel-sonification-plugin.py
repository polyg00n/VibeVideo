"""
Pixel Sonification plugin - Converts pixel data into audio
"""
import cv2
import numpy as np
from __main__ import GlitchEffect
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog
import wave
import os

class PixelSonificationEffect(GlitchEffect):
    """Convert pixel data into audio signals"""
    
    name = "Pixel Sonification"
    description = "Generate audio from pixel data"
    parameters = {
        "frequency_scale": {"type": float, "min": 0.1, "max": 2.0, "default": 1.0},
        "volume": {"type": float, "min": 0.0, "max": 1.0, "default": 0.3},
        "scan_mode": {
            "type": "choice",
            "options": ["Horizontal", "Vertical", "Overall Brightness"],
            "default": "Horizontal"
        },
        "sample_rate": {"type": int, "min": 22050, "max": 44100, "default": 22050},
        "save_audio": {"type": bool, "default": False}
    }

    def __init__(self):
        super().__init__()
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_thread, daemon=True)
        self.save_thread.start()
        self.current_audio_data = None
        self.last_save_state = False
        self.root = None
        self.video_processor = None

    def _get_scan_mode_index(self):
        """Convert scan mode string to index"""
        mode_map = {
            "Horizontal": 0,
            "Vertical": 1,
            "Overall Brightness": 2
        }
        return mode_map.get(self.params["scan_mode"], 0)

    def _save_thread(self):
        while True:
            try:
                save_signal = self.save_queue.get(timeout=1.0)
                if save_signal and self.video_processor:
                    if self.root is None:
                        self.root = tk.Tk()
                        self.root.withdraw()
                    
                    file_path = filedialog.asksaveasfilename(
                        parent=self.root,
                        defaultextension=".wav",
                        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
                        title="Save Audio As"
                    )
                    
                    if file_path:
                        print("Processing video frames for audio...")
                        accumulated_audio = []
                        total_frames = self.video_processor.frame_count
                        
                        cap = cv2.VideoCapture(self.video_processor.source_path)
                        
                        for frame_idx in range(total_frames):
                            ret, frame = cap.read()
                            if ret:
                                audio_data = self._process_frame_to_audio(frame)
                                accumulated_audio.append(audio_data)
                                
                                if frame_idx % 10 == 0:
                                    print(f"Processing frame {frame_idx + 1}/{total_frames}")
                        
                        cap.release()
                        
                        if accumulated_audio:
                            combined_audio = np.concatenate(accumulated_audio)
                            with wave.open(file_path, 'w') as wav_file:
                                wav_file.setnchannels(1)
                                wav_file.setsampwidth(2)
                                wav_file.setframerate(self.params["sample_rate"])
                                wav_file.writeframes(combined_audio.tobytes())
                            
                            print(f"Successfully saved audio to {file_path}")
                        else:
                            print("No audio data generated")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Save error: {e}")

    def _generate_audio(self, pixel_data):
        sample_rate = self.params["sample_rate"]
        duration = 0.1  # 100ms of audio per frame
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        
        # Normalize pixel data to frequency range (20Hz - 2000Hz)
        frequencies = np.interp(pixel_data, (0, 255), (20, 2000))
        frequencies *= self.params["frequency_scale"]
        
        # Generate audio signal
        audio_signal = np.zeros_like(t)
        for freq in frequencies:
            audio_signal += np.sin(2 * np.pi * freq * t)
        
        # Normalize and adjust volume
        if np.max(np.abs(audio_signal)) > 0:
            audio_signal = audio_signal / np.max(np.abs(audio_signal))
        audio_signal *= self.params["volume"]
        
        # Convert to 16-bit integer format
        audio_signal = (audio_signal * 32767).astype(np.int16)
        
        return audio_signal

    def _process_frame_to_audio(self, frame):
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Get scan mode index
        scan_mode = self._get_scan_mode_index()
        
        # Extract pixel data based on scan mode
        if scan_mode == 0:  # Horizontal
            pixel_data = np.mean(gray, axis=0)
        elif scan_mode == 1:  # Vertical
            pixel_data = np.mean(gray, axis=1)
        else:  # Overall brightness
            pixel_data = np.array([np.mean(gray)])
        
        return self._generate_audio(pixel_data)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Generate audio data for current frame
        self.current_audio_data = self._process_frame_to_audio(frame)
        
        # Handle save toggle
        save_state = self.params["save_audio"]
        if save_state and not self.last_save_state:
            try:
                self.save_queue.put(True, block=False)
            except queue.Full:
                pass
            self.params["save_audio"] = False
        self.last_save_state = save_state
        
        # Visualize the audio waveform on the frame
        self._draw_waveform(frame, self.current_audio_data)
        
        return frame

    def _draw_waveform(self, frame, audio_data):
        h, w = frame.shape[:2]
        waveform_height = h // 4
        baseline = h - 50
        
        # Downsample audio data to match frame width
        samples = np.linspace(0, len(audio_data) - 1, w, dtype=int)
        waveform = audio_data[samples]
        waveform = np.interp(waveform, (-32768, 32767), (-waveform_height, waveform_height))
        
        # Draw waveform
        for x in range(w - 1):
            y1 = int(baseline + waveform[x])
            y2 = int(baseline + waveform[x + 1])
            cv2.line(frame, (x, y1), (x + 1, y2), (0, 255, 0), 1)
        
        # Draw baseline
        cv2.line(frame, (0, baseline), (w, baseline), (0, 255, 0), 1)
        
        # Draw current scan mode
        cv2.putText(frame, f"Scan Mode: {self.params['scan_mode']}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def __del__(self):
        """Cleanup when the effect is destroyed"""
        if self.root:
            self.root.destroy() 