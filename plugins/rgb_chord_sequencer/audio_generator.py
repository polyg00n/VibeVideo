"""
Audio generation utilities for RGB Chord Sequencer
"""
import numpy as np
from typing import List, Tuple
import numpy.typing as npt
from dataclasses import dataclass

@dataclass
class AudioConfig:
    sample_rate: int = 44100
    base_freq: float = 130.81  # C3
    volume: float = 0.3

class AudioGenerator:
    def __init__(self, config: AudioConfig = AudioConfig()):
        self.config = config
        self.note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        self.notes = [self.config.base_freq * (2 ** (i/12)) for i in range(25)]  # Two octaves
        self._frequency_cache = {}
        
    def generate_note(self, frequency: float, duration: float, volume: float) -> npt.NDArray[np.float32]:
        """Generate a single note with exact duration"""
        cache_key = (frequency, duration, volume)
        if cache_key in self._frequency_cache:
            return self._frequency_cache[cache_key].copy()

        num_samples = int(self.config.sample_rate * duration)
        t = np.linspace(0, duration, num_samples, False)
        note = np.sin(2 * np.pi * frequency * t)
        
        # Apply envelope
        attack_time = min(0.05, duration * 0.2)
        release_time = min(0.05, duration * 0.2)
        
        attack_samples = int(attack_time * self.config.sample_rate)
        release_samples = int(release_time * self.config.sample_rate)
        
        envelope = np.ones(num_samples)
        
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        if release_samples > 0:
            envelope[-release_samples:] = np.linspace(1, 0, release_samples)
        
        result = (note * envelope * volume).astype(np.float32)
        self._frequency_cache[cache_key] = result
        return result.copy()

    def generate_chord(self, rgb_values: npt.NDArray[np.float32], duration: float, volume: float) -> npt.NDArray[np.float32]:
        """Generate a chord from RGB values"""
        # Map each color channel to a note index (0-24)
        note_indices = [int(np.interp(value, [0, 255], [0, 24])) for value in rgb_values]
        
        # Generate each note
        num_samples = int(self.config.sample_rate * duration)
        notes = [
            self.generate_note(self.notes[idx], duration, volume)[:num_samples]
            for idx in note_indices
        ]
        
        # Mix the notes
        return sum(notes) / len(notes)

    def cleanup_cache(self, max_size: int = 1000):
        """Clean up the frequency cache if it gets too large"""
        if len(self._frequency_cache) > max_size:
            remove_count = len(self._frequency_cache) - max_size
            for _ in range(remove_count):
                self._frequency_cache.popitem(last=False) 