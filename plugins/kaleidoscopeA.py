"""
Kaleidoscope Effect Plugin for Video Glitch Player
Simulates various kaleidoscope configurations with adjustable parameters
"""
import cv2
import numpy as np
import math
from __main__ import GlitchEffect

class KaleidoscopeEffect(GlitchEffect):
    """Simulates a kaleidoscope effect with various mirror configurations"""
    
    name = "Kaleidoscope"
    description = "Simulates various kaleidoscope mirror configurations"
    parameters = {
        "config_type": {
            "type": int,
            "min": 0,
            "max": 4,
            "default": 0,
            "options": ["2 Mirrors 45°", "2 Mirrors 30°", "3 Mirrors 60°", "3 Mirrors 90-45-45°", "3 Mirrors 90-60-30°"]
        },
        "rotation": {"type": float, "min": 0.0, "max": 360.0, "default": 0.0},
        "center_x": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5},
        "center_y": {"type": float, "min": 0.0, "max": 1.0, "default": 0.5},
        "scale": {"type": float, "min": 0.1, "max": 2.0, "default": 1.0},
        "angle_adjust": {"type": float, "min": 0.8, "max": 5, "default": 1.5}  # Added angle adjustment control
    }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        result = np.zeros_like(frame)
        
        # Get parameters
        config_type = self.params["config_type"]
        rotation = math.radians(self.params["rotation"])
        center_x = int(self.params["center_x"] * w)
        center_y = int(self.params["center_y"] * h)
        scale = self.params["scale"]
        angle_adjust = self.params["angle_adjust"]  # Get angle adjustment
        
        # Create transformation matrix for rotation and scaling
        M = cv2.getRotationMatrix2D((center_x, center_y), math.degrees(rotation), scale)
        
        # Apply transformation
        transformed = cv2.warpAffine(frame, M, (w, h))
        
        # Process based on configuration type
        if config_type == 0:  # 2 Mirrors 45°
            result = self._process_two_mirrors(transformed, 45, angle_adjust)
        elif config_type == 1:  # 2 Mirrors 30°
            result = self._process_two_mirrors(transformed, 30, angle_adjust)
        elif config_type == 2:  # 3 Mirrors 60°
            result = self._process_three_mirrors(transformed, 60, angle_adjust)
        elif config_type == 3:  # 3 Mirrors 90-45-45°
            result = self._process_three_mirrors_asymmetric(transformed, [90, 45, 45], angle_adjust)
        elif config_type == 4:  # 3 Mirrors 90-60-30°
            result = self._process_three_mirrors_asymmetric(transformed, [90, 60, 30], angle_adjust)
            
        return result
    
    def _process_two_mirrors(self, frame: np.ndarray, angle: float, angle_adjust: float) -> np.ndarray:
        """Process frame with two mirrors at specified angle"""
        h, w = frame.shape[:2]
        result = np.zeros_like(frame)
        
        # Calculate number of reflections
        num_reflections = int(360 / angle)
        
        # Create base wedge with adjusted angle
        wedge = self._create_wedge(frame, angle * angle_adjust)
        
        # Rotate and copy wedge
        for i in range(num_reflections):
            rotation_angle = i * angle
            rotated = self._rotate_image(wedge, rotation_angle)
            result = cv2.add(result, rotated)
            
        return result
    
    def _process_three_mirrors(self, frame: np.ndarray, angle: float, angle_adjust: float) -> np.ndarray:
        """Process frame with three mirrors at specified angle"""
        h, w = frame.shape[:2]
        result = np.zeros_like(frame)
        
        # Create base triangle with adjusted angle
        triangle = self._create_triangle(frame, angle * angle_adjust)
        
        # Rotate and copy triangle
        for i in range(3):
            rotation_angle = i * 120  # 360/3 = 120 degrees
            rotated = self._rotate_image(triangle, rotation_angle)
            result = cv2.add(result, rotated)
            
        return result
    
    def _process_three_mirrors_asymmetric(self, frame: np.ndarray, angles: list, angle_adjust: float) -> np.ndarray:
        """Process frame with three mirrors at different angles"""
        h, w = frame.shape[:2]
        result = np.zeros_like(frame)
        
        # Create base triangle with adjusted angles
        adjusted_angles = [angle * angle_adjust for angle in angles]
        triangle = self._create_asymmetric_triangle(frame, adjusted_angles)
        
        # Calculate rotation angles based on the largest angle
        max_angle = max(angles)
        num_rotations = int(360 / max_angle)
        
        # Rotate and copy triangle
        for i in range(num_rotations):
            rotation_angle = i * max_angle
            rotated = self._rotate_image(triangle, rotation_angle)
            result = cv2.add(result, rotated)
            
        return result
    
    def _create_wedge(self, frame: np.ndarray, angle: float) -> np.ndarray:
        """Create a wedge shape from the frame"""
        h, w = frame.shape[:2]
        wedge = np.zeros_like(frame)
        
        # Create mask for wedge
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w//2, h//2)
        
        # Draw wedge using polygon
        points = [center]
        for i in range(2):
            theta = math.radians(i * angle)
            x = int(center[0] + w * math.cos(theta))
            y = int(center[1] + h * math.sin(theta))
            points.append((x, y))
            
        cv2.fillPoly(mask, [np.array(points)], 255)
        
        # Apply mask to frame
        for c in range(3):
            wedge[:,:,c] = cv2.bitwise_and(frame[:,:,c], mask)
            
        return wedge
    
    def _create_triangle(self, frame: np.ndarray, angle: float) -> np.ndarray:
        """Create a triangle shape from the frame"""
        h, w = frame.shape[:2]
        triangle = np.zeros_like(frame)
        
        # Create mask for triangle
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w//2, h//2)
        
        # Draw triangle using polygon
        points = [center]
        for i in range(3):
            theta = math.radians(i * angle)
            x = int(center[0] + w * math.cos(theta))
            y = int(center[1] + h * math.sin(theta))
            points.append((x, y))
            
        cv2.fillPoly(mask, [np.array(points)], 255)
        
        # Apply mask to frame
        for c in range(3):
            triangle[:,:,c] = cv2.bitwise_and(frame[:,:,c], mask)
            
        return triangle
    
    def _create_asymmetric_triangle(self, frame: np.ndarray, angles: list) -> np.ndarray:
        """Create an asymmetric triangle shape from the frame"""
        h, w = frame.shape[:2]
        triangle = np.zeros_like(frame)
        
        # Create mask for triangle
        mask = np.zeros((h, w), dtype=np.uint8)
        center = (w//2, h//2)
        
        # Draw triangle using polygon
        points = [center]
        current_angle = 0
        for angle in angles:
            theta = math.radians(current_angle)
            x = int(center[0] + w * math.cos(theta))
            y = int(center[1] + h * math.sin(theta))
            points.append((x, y))
            current_angle += angle
            
        cv2.fillPoly(mask, [np.array(points)], 255)
        
        # Apply mask to frame
        for c in range(3):
            triangle[:,:,c] = cv2.bitwise_and(frame[:,:,c], mask)
            
        return triangle
    
    def _rotate_image(self, image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate image around center"""
        h, w = image.shape[:2]
        center = (w//2, h//2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, M, (w, h)) 