"""
Unit tests for RGB Chord Sequencer
"""
import unittest
import numpy as np
from plugins.rgb_chord_sequencer import RGBChordSequencerEffect
from plugins.rgb_chord_sequencer.audio_generator import AudioGenerator, AudioConfig

class TestRGBChordSequencer(unittest.TestCase):
    def setUp(self):
        self.effect = RGBChordSequencerEffect()
        self.audio_gen = AudioGenerator()
        
    def test_note_generation(self):
        frequency = 440  # A4
        duration = 1.0
        volume = 0.5
        note = self.audio_gen.generate_note(frequency, duration, volume)
        
        self.assertEqual(len(note), int(self.audio_gen.config.sample_rate * duration))
        self.assertTrue(np.all(np.abs(note) <= volume))
        
    def test_chord_generation(self):
        rgb_values = np.array([255, 128, 0])
        duration = 0.5
        volume = 0.3
        chord = self.audio_gen.generate_chord(rgb_values, duration, volume)
        
        self.assertEqual(len(chord), int(self.audio_gen.config.sample_rate * duration))
        
    def test_block_sequence(self):
        # Create test frame
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[:50, :50] = [255, 0, 0]  # Red square
        frame[50:, :50] = [0, 255, 0]  # Green square
        frame[:50, 50:] = [0, 0, 255]  # Blue square
        frame[50:, 50:] = [255, 255, 255]  # White square
        
        # Set divisions to 2x2
        self.effect.params["horizontal_divisions"] = "2"
        self.effect.params["vertical_divisions"] = "2"
        
        blocks = self.effect._get_block_sequence(frame)
        
        self.assertEqual(len(blocks), 4)  # Should have 4 blocks
        self.assertTrue(np.allclose(blocks[0], [255, 0, 0], atol=1))  # Red
        self.assertTrue(np.allclose(blocks[1], [0, 0, 255], atol=1))  # Blue
        self.assertTrue(np.allclose(blocks[2], [0, 255, 0], atol=1))  # Green
        self.assertTrue(np.allclose(blocks[3], [255, 255, 255], atol=1))  # White

if __name__ == '__main__':
    unittest.main() 