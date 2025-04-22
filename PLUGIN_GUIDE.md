# VibeVideo Plugin Development Guide

This guide explains how to create plugins for the VibeVideo application, a video glitch player that allows for extensible video manipulation effects.

## Plugin Structure

A VibeVideo plugin consists of three main components:

1. **Plugin Class**: The main class that implements the glitch effect
2. **Parameter Configuration**: UI and parameter definitions
3. **Effect Implementation**: The actual video processing logic

## Required Classes and Methods

### 1. Plugin Class Structure

```python
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QObject
from .base_plugin import BasePlugin

class YourPluginName(BasePlugin):
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.name = "Your Plugin Name"
        self.description = "Description of what your plugin does"
        self.author = "Your Name"
        self.version = "1.0.0"
        
        # Initialize your parameters here
        self.parameters = {
            "parameter_name": {
                "type": "slider",  # or "checkbox", "dropdown", "color"
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.01,
                "label": "Parameter Label",
                "description": "Parameter description"
            }
        }
```

### 2. Required Methods

Every plugin must implement these methods:

```python
def create_ui(self, parent: QWidget) -> QWidget:
    """
    Create and return the plugin's UI widget.
    This should include all parameter controls.
    """
    pass

def process_frame(self, frame: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    """
    Process a single video frame.
    
    Args:
        frame: The input frame as a numpy array
        params: Dictionary of current parameter values
        
    Returns:
        The processed frame as a numpy array
    """
    pass

def get_default_parameters(self) -> Dict[str, Any]:
    """
    Return the default values for all parameters.
    """
    return {name: param["default"] for name, param in self.parameters.items()}
```

## Parameter Types and UI Guidelines

### 1. Parameter Types

VibeVideo supports several parameter types:

- **Slider**: For continuous numerical values
  ```python
  {
      "type": "slider",
      "default": 0.5,
      "min": 0.0,
      "max": 1.0,
      "step": 0.01,
      "label": "Parameter Label",
      "description": "Parameter description"
  }
  ```

- **Checkbox**: For boolean values
  ```python
  {
      "type": "checkbox",
      "default": False,
      "label": "Enable Effect",
      "description": "Toggle the effect on/off"
  }
  ```

- **Dropdown**: For discrete choices
  ```python
  {
      "type": "dropdown",
      "default": "option1",
      "options": ["option1", "option2", "option3"],
      "label": "Mode",
      "description": "Select processing mode"
  }
  ```

- **Color**: For color selection
  ```python
  {
      "type": "color",
      "default": "#FF0000",
      "label": "Effect Color",
      "description": "Select the color for the effect"
  }
  ```

### 2. UI Guidelines

1. **Layout**:
   - Use a vertical layout for parameters
   - Group related parameters together
   - Add appropriate spacing between groups
   - Use clear, descriptive labels

2. **Responsiveness**:
   - Ensure UI elements scale properly
   - Use appropriate minimum and maximum sizes
   - Consider different screen resolutions

3. **Feedback**:
   - Provide visual feedback for parameter changes
   - Include tooltips for complex parameters
   - Show parameter ranges where applicable

## Example Plugin

Here's a complete example of a simple color shift plugin:

```python
import numpy as np
from typing import Dict, Any, Optional
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider
from PyQt5.QtCore import Qt
from .base_plugin import BasePlugin

class ColorShiftPlugin(BasePlugin):
    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.name = "Color Shift"
        self.description = "Shifts the color channels of the video"
        self.author = "Example Author"
        self.version = "1.0.0"
        
        self.parameters = {
            "red_shift": {
                "type": "slider",
                "default": 0,
                "min": -50,
                "max": 50,
                "step": 1,
                "label": "Red Shift",
                "description": "Shift the red channel"
            },
            "green_shift": {
                "type": "slider",
                "default": 0,
                "min": -50,
                "max": 50,
                "step": 1,
                "label": "Green Shift",
                "description": "Shift the green channel"
            },
            "blue_shift": {
                "type": "slider",
                "default": 0,
                "min": -50,
                "max": 50,
                "step": 1,
                "label": "Blue Shift",
                "description": "Shift the blue channel"
            }
        }

    def create_ui(self, parent: QWidget) -> QWidget:
        widget = QWidget(parent)
        layout = QVBoxLayout(widget)
        
        for param_name, param in self.parameters.items():
            label = QLabel(param["label"])
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(param["min"])
            slider.setMaximum(param["max"])
            slider.setValue(param["default"])
            slider.setSingleStep(param["step"])
            slider.valueChanged.connect(
                lambda value, p=param_name: self.parameter_changed.emit(p, value)
            )
            
            layout.addWidget(label)
            layout.addWidget(slider)
        
        return widget

    def process_frame(self, frame: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        # Create a copy of the frame to avoid modifying the original
        result = frame.copy()
        
        # Apply color shifts
        if params["red_shift"] != 0:
            result[..., 0] = np.roll(result[..., 0], params["red_shift"], axis=1)
        if params["green_shift"] != 0:
            result[..., 1] = np.roll(result[..., 1], params["green_shift"], axis=1)
        if params["blue_shift"] != 0:
            result[..., 2] = np.roll(result[..., 2], params["blue_shift"], axis=1)
        
        return result
```

## Best Practices

1. **Performance**:
   - Optimize frame processing for real-time playback
   - Use numpy operations instead of loops when possible
   - Cache expensive calculations

2. **Error Handling**:
   - Validate input parameters
   - Handle edge cases gracefully
   - Provide meaningful error messages

3. **Documentation**:
   - Document all parameters and their effects
   - Include usage examples
   - Explain any limitations or requirements

4. **Testing**:
   - Test with various video formats and resolutions
   - Verify parameter ranges work as expected
   - Check memory usage and performance

## Plugin Registration

To make your plugin available in VibeVideo:

1. Place your plugin file in the `plugins` directory
2. Import and register your plugin in `plugins/__init__.py`:
   ```python
   from .your_plugin import YourPluginName
   
   def get_plugins():
       return [YourPluginName]
   ```

## Additional Resources

- [NumPy Documentation](https://numpy.org/doc/)
- [PyQt5 Documentation](https://www.riverbankcomputing.com/static/Docs/PyQt5/)
- [OpenCV Documentation](https://docs.opencv.org/) 