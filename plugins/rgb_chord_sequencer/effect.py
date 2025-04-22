"""
Main RGB Chord Sequencer effect implementation
"""
import cv2
import numpy as np
from ..base_effect import GlitchEffect
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog
import wave
import os
import logging
from pathlib import Path
from typing import List, Optional

from .audio_generator import AudioGenerator, AudioConfig
from .midi_generator import MIDIGenerator
from .gui import ProgressDialog, draw_grid

class RGBChordSequencerEffect(GlitchEffect):
    """Convert RGB blocks into musical chord sequences"""
    
    name = "RGB Chord Sequencer"
    description = "Generate musical chords from RGB block data"
    parameters = {
        "horizontal_divisions": {
            "type": "choice",
            "options": ["4", "8", "16", "32"],
            "default": "8"
        },
        "vertical_divisions": {
            "type": "choice",
            "options": ["4", "8", "16", "32"],
            "default": "8"
        },
        "tempo": {"type": int, "min": 60, "max": 240, "default": 120},
        "volume": {"type": float, "min": 0.0, "max": 1.0, "default": 0.3},
        "note_duration": {
            "type": "choice", 
            "options": ["1/4", "1/8", "1/16", "1/32"],
            "default": "1/8"
        },
        "save_audio": {"type": bool, "default": False}
    }

    def __init__(self):
        super().__init__()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        
        # Setup components
        self.audio_generator = AudioGenerator()
        self.midi_generator = MIDIGenerator()
        
        # Setup threading
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_thread, daemon=True)
        self.save_thread.start()
        
        # State tracking
        self.last_save_state = False
        self.root = None
        self.current_frame = None
        
        # Load saved config
        self.config_path = Path.home() / '.rgb_sequencer_config.json'
        self.load_config()

    def _get_block_sequence(self, frame: np.ndarray) -> List[np.ndarray]:
        """Convert frame into sequence of RGB blocks"""
        height, width = frame.shape[:2]
        
        h_div = int(self.params["horizontal_divisions"])
        v_div = int(self.params["vertical_divisions"])
        
        blocks = []
        for y in range(v_div):
            for x in range(h_div):
                x1 = (x * width) // h_div
                y1 = (y * height) // v_div
                x2 = ((x + 1) * width) // h_div
                y2 = ((y + 1) * height) // v_div
                
                block = frame[y1:y2, x1:x2]
                avg_color = np.mean(block, axis=(0, 1))
                blocks.append(avg_color)
        
        return blocks

    def _save_thread(self):
        while True:
            try:
                save_signal = self.save_queue.get(timeout=1.0)
                if save_signal and self.current_frame is not None:
                    self._handle_save()
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Save error: {str(e)}", exc_info=True)

    def _handle_save(self):
        """Handle saving audio and MIDI files"""
        if self.root is None:
            self.root = tk.Tk()
            self.root.withdraw()
        
        file_path = filedialog.asksaveasfilename(
            parent=self.root,
            defaultextension=".wav",
            filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
            title="Save Audio Sequence"
        )
        
        if file_path and self.current_frame is not None:
            base_path = os.path.splitext(file_path)[0]
            wav_path = base_path + ".wav"
            midi_path = base_path + ".mid"
            
            try:
                blocks = self._get_block_sequence(self.current_frame)
                progress = ProgressDialog(self.root, "Generating Audio", len(blocks))
                
                # Calculate timing
                seconds_per_beat = 60.0 / self.params["tempo"]
                note_duration = self._get_note_duration_value() * seconds_per_beat
                
                # Generate and save WAV
                self._save_wav(blocks, note_duration, wav_path, progress)
                
                # Generate and save MIDI
                self._save_midi(blocks, note_duration, midi_path)
                
                self.logger.info(f"Successfully saved:\nWAV: {wav_path}\nMIDI: {midi_path}")
                progress.close()
                
            except Exception as e:
                self.logger.error(f"Error generating audio: {str(e)}", exc_info=True)
                if progress:
                    progress.close()

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame and update the display"""
        self.current_frame = frame.copy()
        
        # Draw grid
        frame = draw_grid(
            frame,
            int(self.params["horizontal_divisions"]),
            int(self.params["vertical_divisions"]),
            self.params["tempo"],
            self.params["note_duration"]
        )
        
        # Handle save toggle
        save_state = self.params["save_audio"]
        if save_state and not self.last_save_state:
            try:
                self.save_queue.put(True, block=False)
            except queue.Full:
                pass
            self.params["save_audio"] = False
        self.last_save_state = save_state
        
        return frame

    def _get_note_duration_value(self) -> float:
        """Convert note duration fraction to float"""
        duration_str = self.params["note_duration"]
        numerator, denominator = map(int, duration_str.split('/'))
        return numerator / denominator

    def _save_wav(self, blocks, note_duration, wav_path, progress):
        """Save audio data to WAV file"""
        samples_per_note = int(self.audio_generator.config.sample_rate * note_duration)
        total_samples = samples_per_note * len(blocks)
        final_audio = np.zeros(total_samples, dtype=np.float32)
        
        for i, block_color in enumerate(blocks):
            chord = self.audio_generator.generate_chord(
                block_color, 
                note_duration,
                self.params["volume"]
            )
            start_idx = i * samples_per_note
            end_idx = start_idx + samples_per_note
            final_audio[start_idx:end_idx] = chord[:samples_per_note]
            
            progress.update(i + 1, f"Processing block {i+1}/{len(blocks)}")
        
        # Normalize and save
        if np.max(np.abs(final_audio)) > 0:
            final_audio = final_audio / np.max(np.abs(final_audio))
        final_audio = np.int16(final_audio * 32767)
        
        with wave.open(wav_path, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.audio_generator.config.sample_rate)
            wav_file.writeframes(final_audio.tobytes())

    def _save_midi(self, blocks, note_duration, midi_path):
        """Save MIDI file"""
        midi = self.midi_generator.create_midi_file(
            blocks,
            self.params["tempo"],
            self._get_note_duration_value(),
            self.params["volume"]
        )
        
        with open(midi_path, "wb") as midi_file:
            midi.writeFile(midi_file) 