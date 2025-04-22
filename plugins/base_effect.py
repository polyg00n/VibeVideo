"""
Base class for all video effects
"""
import abc
from typing import Dict, Any

class GlitchEffect(abc.ABC):
    """Base class for all glitch effects"""
    
    name = "Base Effect"
    description = "Base effect class"
    parameters = {}  # Format: {param_name: {"type": type, "min": min, "max": max, "default": default}}
    
    def __init__(self):
        # Initialize with default parameter values
        self.params = {name: details["default"] for name, details in self.parameters.items()}
    
    def set_param(self, name: str, value: Any) -> None:
        """Set a parameter value with type safety"""
        if name in self.params:
            expected_type = self.parameters[name]["type"]

            try:
                # Auto-cast value to expected type
                if expected_type == int:
                    self.params[name] = int(value)
                elif expected_type == float:
                    self.params[name] = float(value)
                elif expected_type == bool:
                    self.params[name] = bool(value)
                elif expected_type == str:
                    self.params[name] = str(value)
                elif expected_type == "choice":
                    self.params[name] = str(value)  # choices are string values
                else:
                    self.params[name] = value  # fallback
            except Exception as e:
                print(f"[WARNING] Failed to cast {name} to {expected_type}: {e}")
        
    def get_params(self) -> Dict[str, Any]:
        """Get all parameters"""
        return self.params
        
    @abc.abstractmethod
    def process_frame(self, frame):
        """Process a single frame with the effect"""
        pass
    
    @classmethod
    def get_ui_info(cls) -> Dict:
        """Return information needed to build a UI for this effect"""
        return {
            "name": cls.name,
            "description": cls.description,
            "parameters": cls.parameters
        } 