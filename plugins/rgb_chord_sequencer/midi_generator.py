"""
MIDI generation utilities for RGB Chord Sequencer
"""
from midiutil import MIDIFile
import numpy as np
from typing import List, Tuple
import numpy.typing as npt

class MIDIGenerator:
    def __init__(self, base_midi_note: int = 48):  # C3 = 48
        self.base_midi_note = base_midi_note
        
    def get_midi_note(self, rgb_value: float) -> int:
        """Convert RGB value to MIDI note number"""
        note_idx = int(np.interp(rgb_value, [0, 255], [0, 24]))
        return self.base_midi_note + note_idx
        
    def create_midi_file(self, 
                        blocks: List[npt.NDArray[np.float32]], 
                        tempo: int,
                        note_duration: float,
                        volume: float) -> MIDIFile:
        """Create a MIDI file from block data"""
        midi = MIDIFile(3)  # 3 tracks for R, G, B
        
        # Setup tracks
        for i in range(3):
            midi.addTrackName(i, 0, f"Channel {i}")
            midi.addTempo(i, 0, tempo)
        
        # Add notes for each block
        for i, block_color in enumerate(blocks):
            time = i * note_duration
            midi_volume = int(volume * 127)
            
            # Add notes for each color channel
            for channel, color_value in enumerate(block_color):
                midi_note = self.get_midi_note(color_value)
                midi.addNote(channel, channel, midi_note, time, note_duration, midi_volume)
        
        return midi 