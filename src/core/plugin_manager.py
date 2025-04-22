"""
Plugin management system for video effects.
"""
import os
import importlib
import inspect
from typing import Dict, Type, List, Optional
import logging
from pathlib import Path
from .effects import GlitchEffect

class PluginManager:
    """Manages loading and registering effect plugins"""
    
    def __init__(self, plugin_dirs: Optional[List[str]] = None):
        self.effects: Dict[str, Type[GlitchEffect]] = {}
        self.plugin_dirs = plugin_dirs or ["plugins"]
        self._logger = logging.getLogger(__name__)
        
    def discover_plugins(self) -> None:
        """Discover and load all plugins from plugin directories"""
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                self._logger.warning(f"Plugin directory {plugin_dir} does not exist")
                continue
                
            # Convert to Path object for better handling
            plugin_path = Path(plugin_dir)
            
            # Create __init__.py if it doesn't exist
            init_file = plugin_path / "__init__.py"
            if not init_file.exists():
                init_file.touch()
            
            # Load all Python files in the directory
            for file in plugin_path.glob("*.py"):
                if file.name.startswith("_"):
                    continue
                    
                try:
                    # Import the module
                    module_name = f"{plugin_dir}.{file.stem}"
                    module = importlib.import_module(module_name)
                    
                    # Find all GlitchEffect subclasses
                    for _, obj in inspect.getmembers(module):
                        if (inspect.isclass(obj) and 
                            issubclass(obj, GlitchEffect) and 
                            obj is not GlitchEffect):
                            self.register_effect(obj)
                            
                except Exception as e:
                    self._logger.error(f"Error loading plugin {file.name}: {e}")
    
    def register_effect(self, effect_class: Type[GlitchEffect]) -> None:
        """Register a new effect class"""
        if not issubclass(effect_class, GlitchEffect):
            raise ValueError(f"{effect_class.__name__} is not a subclass of GlitchEffect")
            
        if effect_class.__name__ in self.effects:
            self._logger.warning(f"Effect {effect_class.__name__} already registered")
            return
            
        self.effects[effect_class.__name__] = effect_class
        self._logger.info(f"Registered effect: {effect_class.name}")
    
    def get_effect_classes(self) -> Dict[str, Type[GlitchEffect]]:
        """Get all registered effect classes"""
        return self.effects.copy()
    
    def get_effect_by_name(self, name: str) -> Optional[Type[GlitchEffect]]:
        """Get an effect class by name"""
        return self.effects.get(name)
    
    def create_effect(self, name: str) -> Optional[GlitchEffect]:
        """Create a new effect instance by name"""
        effect_class = self.get_effect_by_name(name)
        if effect_class:
            return effect_class()
        return None
    
    def get_effect_info(self) -> List[Dict]:
        """Get information about all available effects"""
        return [
            {
                "name": effect_class.name,
                "class_name": name,
                "description": effect_class.description,
                "parameters": effect_class.parameters
            }
            for name, effect_class in self.effects.items()
        ]
    
    def reload_plugins(self) -> None:
        """Reload all plugins"""
        self.effects.clear()
        self.discover_plugins() 