"""
Base classes for video effects and effect chain management.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Type, List
import numpy as np
import cv2
from dataclasses import dataclass
from typing import TypeVar, Generic

T = TypeVar('T')

@dataclass
class EffectParameter:
    """Parameter definition for an effect"""
    name: str
    type: Type[T]
    min: Optional[T] = None
    max: Optional[T] = None
    default: Optional[T] = None
    description: str = ""
    options: Optional[List[T]] = None

class GlitchEffect(ABC):
    """Base class for all glitch effects"""
    
    name: str = "Base Effect"
    description: str = "Base effect class"
    parameters: Dict[str, EffectParameter] = {}
    
    def __init__(self):
        # Initialize with default parameter values
        self._params: Dict[str, Any] = {
            name: param.default for name, param in self.parameters.items()
        }
        self._cache: Dict[str, Any] = {}
    
    def set_param(self, name: str, value: Any) -> None:
        """Set a parameter value with type safety and validation"""
        if name not in self.parameters:
            raise ValueError(f"Unknown parameter: {name}")
            
        param = self.parameters[name]
        
        try:
            # Type conversion
            if param.type == int:
                value = int(value)
            elif param.type == float:
                value = float(value)
            elif param.type == bool:
                value = bool(value)
            elif param.type == str:
                value = str(value)
            
            # Validation
            if param.min is not None and value < param.min:
                value = param.min
            if param.max is not None and value > param.max:
                value = param.max
            if param.options is not None and value not in param.options:
                raise ValueError(f"Invalid value for {name}. Must be one of {param.options}")
            
            self._params[name] = value
            self._cache.clear()  # Clear cache when parameters change
            
        except (ValueError, TypeError) as e:
            raise ValueError(f"Invalid value for parameter {name}: {e}")
    
    def get_param(self, name: str) -> Any:
        """Get a parameter value"""
        return self._params.get(name)
    
    def get_params(self) -> Dict[str, Any]:
        """Get all parameters"""
        return self._params.copy()
    
    @abstractmethod
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame with the effect"""
        pass
    
    @classmethod
    def get_ui_info(cls) -> Dict:
        """Return information needed to build a UI for this effect"""
        return {
            "name": cls.name,
            "description": cls.description,
            "parameters": {
                name: {
                    "type": param.type.__name__,
                    "min": param.min,
                    "max": param.max,
                    "default": param.default,
                    "description": param.description,
                    "options": param.options
                }
                for name, param in cls.parameters.items()
            }
        }

class EffectChain:
    """Chain multiple effects together with caching"""
    
    def __init__(self):
        self.effects: List[GlitchEffect] = []
        self._frame_cache: Dict[int, np.ndarray] = {}
        self._max_cache_size: int = 100  # Maximum number of frames to cache
        
    def add_effect(self, effect: GlitchEffect) -> None:
        """Add an effect to the chain"""
        self.effects.append(effect)
        self._frame_cache.clear()  # Clear cache when chain changes
        
    def remove_effect(self, index: int) -> None:
        """Remove an effect from the chain"""
        if 0 <= index < len(self.effects):
            self.effects.pop(index)
            self._frame_cache.clear()
            
    def move_effect(self, from_idx: int, to_idx: int) -> None:
        """Change the order of effects"""
        if 0 <= from_idx < len(self.effects) and 0 <= to_idx < len(self.effects):
            effect = self.effects.pop(from_idx)
            self.effects.insert(to_idx, effect)
            self._frame_cache.clear()
            
    def process_frame(self, frame: np.ndarray, frame_index: int = -1) -> np.ndarray:
        """Process a frame through all effects in the chain with caching"""
        if frame_index >= 0 and frame_index in self._frame_cache:
            return self._frame_cache[frame_index]
            
        result = frame.copy()
        for effect in self.effects:
            result = effect.process_frame(result)
            
        if frame_index >= 0:
            # Update cache
            if len(self._frame_cache) >= self._max_cache_size:
                # Remove oldest entry
                oldest_key = min(self._frame_cache.keys())
                del self._frame_cache[oldest_key]
            self._frame_cache[frame_index] = result
            
        return result
    
    def clear_cache(self) -> None:
        """Clear the frame cache"""
        self._frame_cache.clear()
    
    def save_config(self) -> Dict:
        """Save the current chain configuration"""
        return [
            {
                "effect_type": effect.__class__.__name__,
                "parameters": effect.get_params()
            }
            for effect in self.effects
        ]
    
    def load_config(self, config: List[Dict], available_effects: Dict[str, Type[GlitchEffect]]) -> None:
        """Load a chain configuration"""
        self.effects = []
        for effect_config in config:
            effect_type = effect_config["effect_type"]
            if effect_type in available_effects:
                effect = available_effects[effect_type]()
                for param_name, param_value in effect_config["parameters"].items():
                    effect.set_param(param_name, param_value)
                self.effects.append(effect)
        self._frame_cache.clear() 