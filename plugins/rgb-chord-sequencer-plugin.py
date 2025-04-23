"""
RGB Chord Sequencer plugin - Converts RGB blocks into musical chord sequences
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
import math
from midiutil import MIDIFile
from typing import Any

class RGBChordSequencerEffect(GlitchEffect):
    """Convert RGB blocks into musical chord sequences"""
    
    name = "RGB Chord Sequencer"
    description = "Generate musical chords from RGB block data"
    
    # Define note duration mapping
    NOTE_DURATIONS = {
        0: "1/4",  # Quarter note
        1: "1/8",  # Eighth note
        2: "1/16", # Sixteenth note
        3: "1/32"  # Thirty-second note
    }
    
    parameters = {
        "horizontal_divisions": {
            "type": int,
            "min": 4,
            "max": 32,
            "default": 4
        },
        "vertical_divisions": {
            "type": int,
            "min": 4,
            "max": 32,
            "default": 4
        },
        "tempo": {"type": int, "min": 60, "max": 240, "default": 120},
        "volume": {"type": float, "min": 0.0, "max": 1.0, "default": 0.3},
        "note_duration": {
            "type": int,
            "min": 0,
            "max": len(NOTE_DURATIONS) - 1,
            "default": 0,  # Default to quarter note
            "label": "Note Duration",
            "description": "Select the duration of each note"
        },
        "save_audio": {"type": bool, "default": False}
    }

    def __init__(self):
        super().__init__()
        self.save_queue = queue.Queue()
        self.save_thread = threading.Thread(target=self._save_thread, daemon=True)
        self.save_thread.start()
        self.last_save_state = False
        self.root = None
        self.current_frame = None
        
        # Track last division settings to detect changes
        self._last_h_div = self.params["horizontal_divisions"]
        self._last_v_div = self.params["vertical_divisions"]
        
        # Audio settings
        self.sample_rate = 44100
        
        # Musical notes (C3 to C5, two octaves)
        self.base_freq = 130.81  # C3
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.notes = [self.base_freq * (2 ** (i/12)) for i in range(25)]  # Two octaves
        
        # MIDI note numbers (C3 = MIDI note 48)
        self.base_midi_note = 48

    def _regenerate_audio(self):
        """Regenerate audio when parameters change"""
        if self.current_frame is not None and self.params["save_audio"]:
            try:
                self.save_queue.put(True, block=False)
            except queue.Full:
                pass

    def set_param(self, name: str, value: Any) -> None:
        """Override set_param to handle parameter changes"""
        print(f"\n[RGBChord] set_param called with name={name}, value={value}")
        print(f"[RGBChord] Current params before change: {self.params}")
        
        if name in self.params:
            expected_type = self.parameters[name]["type"]

            try:
                # Auto-cast value to expected type
                if expected_type == int:
                    value = int(value)
                    # For note duration, ensure it's within valid range
                    if name == "note_duration":
                        value = max(0, min(value, len(self.NOTE_DURATIONS) - 1))
                        print(f"[RGBChord] Note duration index changed to: {value} ({self.NOTE_DURATIONS[value]})")
                    self.params[name] = value
                elif expected_type == float:
                    self.params[name] = float(value)
                elif expected_type == bool:
                    self.params[name] = bool(value)
                else:
                    self.params[name] = value  # fallback
                    
                print(f"[RGBChord] Params after change: {self.params}")
                
                # Store current frame if we need to redraw
                if name in ["horizontal_divisions", "vertical_divisions"] and self.current_frame is not None:
                    # Force redraw by processing current frame again
                    self.process_frame(self.current_frame.copy())
                    # Regenerate audio if save_audio is enabled
                    self._regenerate_audio()
            except Exception as e:
                print(f"[WARNING] Failed to cast {name} to {expected_type}: {e}")

    def _get_midi_note(self, rgb_value, channel):
        """Convert RGB value to MIDI note number"""
        # Map RGB value (0-255) to note index (0-24)
        note_idx = int(np.interp(rgb_value, [0, 255], [0, 24]))
        return self.base_midi_note + note_idx

    def _get_note_duration_value(self):
        """Convert note duration index to float"""
        duration_str = self.NOTE_DURATIONS[self.params["note_duration"]]
        numerator, denominator = map(int, duration_str.split('/'))
        return numerator / denominator

    def _generate_note(self, frequency, duration, volume=0.3):
        """Generate a single note with exact duration"""
        # Calculate exact number of samples needed
        num_samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        note = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope
        attack_time = min(0.05, duration * 0.2)  # 20% of duration or 50ms, whichever is shorter
        release_time = min(0.05, duration * 0.2)  # 20% of duration or 50ms, whichever is shorter
        
        attack_samples = int(attack_time * self.sample_rate)
        release_samples = int(release_time * self.sample_rate)
        
        envelope = np.ones(num_samples)
        
        # Apply attack
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Apply release
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        return (note * envelope * volume).astype(np.float32)

    def _generate_chord(self, rgb_values, duration):
        """Generate a chord from RGB values with exact duration"""
        # Map each color channel to a note index (0-24)
        r_note_idx = int(np.interp(rgb_values[0], [0, 255], [0, 24]))
        g_note_idx = int(np.interp(rgb_values[1], [0, 255], [0, 24]))
        b_note_idx = int(np.interp(rgb_values[2], [0, 255], [0, 24]))
        
        # Calculate exact number of samples
        num_samples = int(self.sample_rate * duration)
        
        # Generate notes with exact duration
        r_note = self._generate_note(self.notes[r_note_idx], duration, self.params["volume"])
        g_note = self._generate_note(self.notes[g_note_idx], duration, self.params["volume"])
        b_note = self._generate_note(self.notes[b_note_idx], duration, self.params["volume"])
        
        # Mix the notes and ensure exact length
        chord = (r_note[:num_samples] + g_note[:num_samples] + b_note[:num_samples]) / 3.0
        return chord

    def _get_block_sequence(self, frame):
        """Convert frame into sequence of RGB blocks"""
        height, width = frame.shape[:2]
        
        # Get division values directly as integers
        h_div = self.params["horizontal_divisions"]
        v_div = self.params["vertical_divisions"]
        
        blocks = []
        # Iterate through blocks in row-major order (left to right, top to bottom)
        for y in range(v_div):
            for x in range(h_div):
                # Calculate exact block boundaries
                x1 = (x * width) // h_div
                y1 = (y * height) // v_div
                x2 = ((x + 1) * width) // h_div
                y2 = ((y + 1) * height) // v_div
                
                # Extract block and compute average color
                block = frame[y1:y2, x1:x2]
                avg_color = np.mean(block, axis=(0, 1))
                blocks.append(avg_color)
                
        return blocks

    def _save_thread(self):
        while True:
            try:
                save_signal = self.save_queue.get(timeout=1.0)
                if save_signal and self.current_frame is not None:
                    if self.root is None:
                        self.root = tk.Tk()
                        self.root.withdraw()
                    
                    file_path = filedialog.asksaveasfilename(
                        parent=self.root,
                        defaultextension=".wav",
                        filetypes=[("WAV files", "*.wav"), ("All files", "*.*")],
                        title="Save Audio Sequence"
                    )
                    
                    if file_path:
                        base_path = os.path.splitext(file_path)[0]
                        wav_path = base_path + ".wav"
                        midi_path = base_path + ".mid"
                        
                        print("Generating audio sequence...")
                        try:
                            # Get block sequence
                            blocks = self._get_block_sequence(self.current_frame)
                            
                            # Print debug information
                            h_div = int(self.params["horizontal_divisions"])
                            v_div = int(self.params["vertical_divisions"])
                            print(f"Divisions: {h_div}x{v_div}")
                            print(f"Number of blocks: {len(blocks)} (should be {h_div * v_div})")
                            
                            # Calculate exact note duration
                            seconds_per_beat = 60.0 / self.params["tempo"]
                            note_duration = self._get_note_duration_value() * seconds_per_beat
                            
                            # Pre-calculate total samples needed
                            samples_per_note = int(self.sample_rate * note_duration)
                            total_samples = samples_per_note * len(blocks)
                            
                            # Generate audio
                            final_audio = np.zeros(total_samples, dtype=np.float32)
                            
                            for i, block_color in enumerate(blocks):
                                try:
                                    chord = self._generate_chord(block_color, note_duration)
                                    start_idx = i * samples_per_note
                                    end_idx = start_idx + samples_per_note
                                    final_audio[start_idx:end_idx] = chord[:samples_per_note]
                                    
                                    if i % 10 == 0:
                                        print(f"Processing block {i+1}/{len(blocks)}")
                                except Exception as e:
                                    print(f"Error processing block {i}: {e}")
                                    continue
                            
                            # Normalize audio
                            if np.max(np.abs(final_audio)) > 0:
                                final_audio = final_audio / np.max(np.abs(final_audio))
                            final_audio = np.int16(final_audio * 32767)
                            
                            # Save WAV file
                            with wave.open(wav_path, 'w') as wav_file:
                                wav_file.setnchannels(1)
                                wav_file.setsampwidth(2)
                                wav_file.setframerate(self.sample_rate)
                                wav_file.writeframes(final_audio.tobytes())
                            
                            # Save MIDI file
                            midi = MIDIFile(3)  # 3 tracks for R, G, B
                            
                            for i in range(3):
                                midi.addTrackName(i, 0, f"Channel {i}")
                                midi.addTempo(i, 0, self.params["tempo"])
                            
                            note_duration_beats = self._get_note_duration_value()
                            
                            for i, block_color in enumerate(blocks):
                                time = i * note_duration_beats
                                volume = int(self.params["volume"] * 127)
                                
                                for channel, color_value in enumerate(block_color):
                                    midi_note = self._get_midi_note(color_value, channel)
                                    midi.addNote(channel, channel, midi_note, time, 
                                               note_duration_beats, volume)
                            
                            with open(midi_path, "wb") as midi_file:
                                midi.writeFile(midi_file)
                            
                            print(f"Successfully saved:\nWAV: {wav_path}\nMIDI: {midi_path}")
                        except Exception as e:
                            print(f"Error generating audio sequence: {e}")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Save error: {e}")

    def _draw_grid(self, frame):
        """Draw block grid overlay"""
        height, width = frame.shape[:2]
        
        # Convert division strings to integers
        h_div = int(self.params["horizontal_divisions"])
        v_div = int(self.params["vertical_divisions"])
        
        # Create a copy of the frame for drawing
        overlay = frame.copy()
        
        # Draw vertical lines
        for i in range(h_div + 1):
            x = (width * i) // h_div
            cv2.line(overlay, (x, 0), (x, height), (255, 255, 255), 1)
        
        # Draw horizontal lines
        for i in range(v_div + 1):
            y = (height * i) // v_div
            cv2.line(overlay, (0, y), (width, y), (255, 255, 255), 1)
        
        # Draw block numbers for debugging
        font_scale = min(width, height) / (1000 * max(h_div, v_div))  # Adjust font size based on divisions
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
        
        # Draw info - Use current parameter values directly
        cv2.putText(overlay, 
                   f"Tempo: {self.params['tempo']} BPM", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay, 
                   f"Divisions: {h_div}x{v_div}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(overlay,
                   f"Note: {self.NOTE_DURATIONS[self.params['note_duration']]}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Blend the overlay with the original frame
        alpha = 0.7
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame and update the display"""
        # Store current frame for audio generation and potential redraw
        self.current_frame = frame.copy()
        
        # Draw block grid
        self._draw_grid(frame)
        
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