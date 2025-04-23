"""
Video Glitch Player - Plugin-based Architecture
"""
import cv2
import numpy as np
import tkinter as tk
from tkinter import filedialog, ttk
import abc
import importlib
import inspect
import os
import json
from typing import Dict, List, Any, Optional, Tuple, Type
import threading
from PIL import Image, ImageTk
from plugins.rgb_chord_sequencer import RGBChordSequencerEffect

# ===== Base Effect Plugin System =====

class GlitchEffect(abc.ABC):
    """Base class for all glitch effects"""
    
    # Class attribute to store metadata
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
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
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


class EffectChain:
    """Chain multiple effects together"""
    
    def __init__(self):
        self.effects = []
        
    def add_effect(self, effect: GlitchEffect) -> None:
        """Add an effect to the chain"""
        self.effects.append(effect)
        
    def remove_effect(self, index: int) -> None:
        """Remove an effect from the chain"""
        if 0 <= index < len(self.effects):
            self.effects.pop(index)
            
    def move_effect(self, from_idx: int, to_idx: int) -> None:
        """Change the order of effects"""
        if 0 <= from_idx < len(self.effects) and 0 <= to_idx < len(self.effects):
            effect = self.effects.pop(from_idx)
            self.effects.insert(to_idx, effect)
            
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame through all effects in the chain"""
        result = frame.copy()
        for effect in self.effects:
            result = effect.process_frame(result)
        return result
    
    def save_config(self) -> Dict:
        """Save the current chain configuration"""
        config = []
        for effect in self.effects:
            effect_config = {
                "effect_type": effect.__class__.__name__,
                "parameters": effect.get_params()
            }
            config.append(effect_config)
        return config
    
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


class PluginManager:
    """Manages loading and registering effect plugins"""
    
    def __init__(self, plugin_dirs=None):
        self.effects = {}  # Maps effect names to effect classes
        self.plugin_dirs = plugin_dirs or ["plugins"]
        
    def discover_plugins(self) -> None:
        """Discover and load all plugins from plugin directories"""
        for plugin_dir in self.plugin_dirs:
            if not os.path.exists(plugin_dir):
                os.makedirs(plugin_dir)
            
            for filename in os.listdir(plugin_dir):
                if filename.endswith(".py") and not filename.startswith("_"):
                    module_name = filename[:-3]  # Remove .py extension
                    try:
                        # Import the module
                        module_path = f"{plugin_dir}.{module_name}"
                        module = importlib.import_module(module_path)
                        
                        # Find all GlitchEffect subclasses in the module
                        for _, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and 
                                issubclass(obj, GlitchEffect) and 
                                obj is not GlitchEffect):
                                self.register_effect(obj)
                    except Exception as e:
                        print(f"Error loading plugin {module_name}: {e}")
    
    def register_effect(self, effect_class: Type[GlitchEffect]) -> None:
        """Register a new effect class"""
        self.effects[effect_class.__name__] = effect_class
        
    def get_effect_classes(self) -> Dict[str, Type[GlitchEffect]]:
        """Get all registered effect classes"""
        return self.effects
    
    def get_effect_by_name(self, name: str) -> Optional[Type[GlitchEffect]]:
        """Get an effect class by name"""
        return self.effects.get(name)


# ===== Video Processing Core =====

class VideoGlitchProcessor:
    """Core video processing engine"""
    
    def __init__(self, source_path=None):
        self.source_path = source_path
        self.cap = None
        self.width = 0
        self.height = 0
        self.fps = 0
        self.frame_count = 0
        self.effect_chain = EffectChain()
        
        if source_path:
            self.load_video(source_path)
            
    def load_video(self, path: str) -> bool:
        """Load a video file"""
        if self.cap:
            self.cap.release()
            
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            return False
            
        self.source_path = path
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return True
        
    def get_frame(self, position: int) -> Tuple[bool, Optional[np.ndarray]]:
        """Get a specific frame from the video"""
        if not self.cap or not self.cap.isOpened():
            return False, None
            
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        ret, frame = self.cap.read()
        return ret, frame
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame with the current effect chain"""
        return self.effect_chain.process_frame(frame)
        
    def process_video(self, output_path: str, progress_callback=None) -> bool:
        """Process the entire video and save to output path"""
        if not self.cap or not self.cap.isOpened():
            return False
            
        # Reset to beginning
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        
        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'H264', etc.
        out = cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
        
        frame_index = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            # Process the frame
            processed = self.process_frame(frame)
            out.write(processed)
            
            # Update progress
            frame_index += 1
            if progress_callback:
                progress_callback(frame_index / self.frame_count)
                
        out.release()
        return True
    
    def release(self):
        """Release resources"""
        if self.cap:
            self.cap.release()


# ===== Sample Effects =====

class ByteShiftEffect(GlitchEffect):
    """Shift bytes in the frame"""
    
    name = "Byte Shift"
    description = "Shift pixel values by a certain amount"
    parameters = {
        "amount": {"type": int, "min": -50, "max": 50, "default": 10},
        "axis": {"type": int, "min": 0, "max": 2, "default": 0}
    }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        return np.roll(frame, self.params["amount"], axis=self.params["axis"])


class ChannelShiftEffect(GlitchEffect):
    """Shift color channels spatially"""
    
    name = "Channel Shift"
    description = "Shift RGB channels independently"
    parameters = {
        "r_shift_x": {"type": int, "min": -50, "max": 50, "default": 10},
        "r_shift_y": {"type": int, "min": -50, "max": 50, "default": 0},
        "g_shift_x": {"type": int, "min": -50, "max": 50, "default": 0},
        "g_shift_y": {"type": int, "min": -50, "max": 50, "default": 10},
        "b_shift_x": {"type": int, "min": -50, "max": 50, "default": -10},
        "b_shift_y": {"type": int, "min": -50, "max": 50, "default": -10}
    }
    
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(frame)
        
        r_shifted = np.roll(r, (self.params["r_shift_y"], self.params["r_shift_x"]), axis=(0, 1))
        g_shifted = np.roll(g, (self.params["g_shift_y"], self.params["g_shift_x"]), axis=(0, 1))
        b_shifted = np.roll(b, (self.params["b_shift_y"], self.params["b_shift_x"]), axis=(0, 1))
        
        return cv2.merge([b_shifted, g_shifted, r_shifted])


# ===== Simple GUI =====

class VideoGlitchGUI:
    """Simple GUI for the video glitch application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Video Glitch Player")
        self.processor = VideoGlitchProcessor()
        self.plugin_manager = PluginManager()
        self.plugin_manager.discover_plugins()
        
        # Register built-in effects
        self.plugin_manager.register_effect(ByteShiftEffect)
        self.plugin_manager.register_effect(ChannelShiftEffect)
        
        # Current frame for preview
        self.current_frame_index = 0
        self.preview_frame = None
        
        # Playback control variables
        self.is_playing = False
        self.play_after_id = None
        
        self._build_ui()
        
    def _build_ui(self):
        # Set window close handler
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
        
        # Main layout using PanedWindow
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        paned = ttk.Panedwindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # === Left side: video controls and preview ===
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)

        controls_frame = ttk.LabelFrame(left_frame, text="Video Controls", padding=5)
        controls_frame.pack(fill=tk.X, pady=5)

        ttk.Button(controls_frame, text="Open Video", command=self._open_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Export Processed Video", command=self._export_video).pack(side=tk.LEFT, padx=5)

        preview_frame = ttk.LabelFrame(left_frame, text="Preview", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.canvas = tk.Canvas(preview_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Add playback controls frame
        playback_frame = ttk.Frame(preview_frame)
        playback_frame.pack(fill=tk.X, pady=5)

        # Add play/pause button
        self.is_playing = False
        self.play_button = ttk.Button(playback_frame, text="▶", width=3, command=self._toggle_playback)
        self.play_button.pack(side=tk.LEFT, padx=5)

        slider_frame = ttk.Frame(preview_frame)
        slider_frame.pack(fill=tk.X, pady=5)

        ttk.Label(slider_frame, text="Frame:").pack(side=tk.LEFT)
        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL, 
                                      command=self._on_frame_slider_change)
        self.frame_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        self.frame_label = ttk.Label(slider_frame, text="0 / 0")
        self.frame_label.pack(side=tk.LEFT)

        # === Right side: effects chain and parameters ===
        right_frame = ttk.Frame(paned)
        paned.add(right_frame, weight=2)

        effects_frame = ttk.LabelFrame(right_frame, text="Effects Chain", padding=5)
        effects_frame.pack(fill=tk.BOTH, pady=5, expand=True)

        effects_list_frame = ttk.Frame(effects_frame)
        effects_list_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.effects_list = ttk.Treeview(effects_list_frame, columns=("name",), show="headings")
        self.effects_list.heading("name", text="Effect")
        self.effects_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        effects_scroll = ttk.Scrollbar(effects_list_frame, command=self.effects_list.yview)
        effects_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.effects_list.configure(yscrollcommand=effects_scroll.set)
        self.effects_list.bind("<<TreeviewSelect>>", self._on_effect_selected)

        effects_buttons_frame = ttk.Frame(effects_frame)
        effects_buttons_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Button(effects_buttons_frame, text="Add", command=self._add_effect).pack(fill=tk.X, pady=2)
        ttk.Button(effects_buttons_frame, text="Remove", command=self._remove_effect).pack(fill=tk.X, pady=2)
        ttk.Button(effects_buttons_frame, text="Move Up", command=self._move_effect_up).pack(fill=tk.X, pady=2)
        ttk.Button(effects_buttons_frame, text="Move Down", command=self._move_effect_down).pack(fill=tk.X, pady=2)

        self.params_frame = ttk.LabelFrame(right_frame, text="Effect Parameters", padding=5)
        self.params_frame.pack(fill=tk.BOTH, expand=False, pady=5)

        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(5, 0))

    def _open_video(self):
        """Open a video file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
            )
            
            if file_path:
                if self.processor.load_video(file_path):
                    self.status_var.set(f"Loaded video: {os.path.basename(file_path)}")
                    self.frame_slider.configure(to=self.processor.frame_count - 1)
                    self.current_frame_index = 0
                    self._update_preview()
                else:
                    self.status_var.set("Error: Could not open video file")
        except Exception as e:
            print(f"Error opening video: {e}")
            self.status_var.set("Error: Could not open video file")
    
    def _export_video(self):
        """Export processed video"""
        if not self.processor.source_path:
            self.status_var.set("Error: No video loaded")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Processed Video",
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("AVI video", "*.avi")]
        )
        
        if file_path:
            # Create a progress dialog
            progress_window = tk.Toplevel(self.root)
            progress_window.title("Exporting Video")
            progress_window.geometry("300x100")
            progress_window.transient(self.root)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="Processing video...")
            progress_label.pack(pady=10)
            
            progress_bar = ttk.Progressbar(progress_window, mode="determinate")
            progress_bar.pack(fill=tk.X, padx=20, pady=10)
            
            def update_progress(progress):
                progress_bar["value"] = progress * 100
                progress_window.update()
            
            # Process in a separate thread to keep UI responsive
            import threading
            
            def process_thread():
                success = self.processor.process_video(file_path, update_progress)
                
                # Update UI in main thread
                self.root.after(0, lambda: self._on_export_complete(progress_window, success, file_path))
            
            thread = threading.Thread(target=process_thread)
            thread.daemon = True
            thread.start()
    
    def _on_export_complete(self, progress_window, success, file_path):
        """Called when export is complete"""
        progress_window.destroy()
        
        if success:
            self.status_var.set(f"Video exported to: {os.path.basename(file_path)}")
        else:
            self.status_var.set("Error exporting video")
    
    def _on_frame_slider_change(self, value):
        """Handle frame slider change"""
        try:
            new_index = int(float(value))
            if new_index != self.current_frame_index:
                self.current_frame_index = new_index
                self._update_preview()
                
                # Stop playback if user manually moves slider
                if self.is_playing:
                    self._stop_playback()
        except Exception as e:
            print(f"Slider error: {e}")
            self._stop_playback()
    
    def _toggle_playback(self):
        """Toggle video playback"""
        if not self.processor.source_path:
            return

        self.is_playing = not self.is_playing
        
        if self.is_playing:
            self.play_button.configure(text="⏸")
            self._play_next_frame()
        else:
            self.play_button.configure(text="▶")
            if self.play_after_id:
                self.root.after_cancel(self.play_after_id)
                self.play_after_id = None

    def _play_next_frame(self):
        """Play the next frame in sequence"""
        if not self.is_playing:
            return

        try:
            next_frame = (self.current_frame_index + 1) % self.processor.frame_count
            self.current_frame_index = next_frame
            self.frame_slider.set(next_frame)
            self._update_preview()

            # Calculate delay based on video FPS
            delay = int(1000 / max(1, self.processor.fps))  # Ensure minimum 1ms delay
            self.play_after_id = self.root.after(delay, self._play_next_frame)
        except Exception as e:
            print(f"Playback error: {e}")
            self._stop_playback()

    def _stop_playback(self):
        """Stop video playback"""
        self.is_playing = False
        self.play_button.configure(text="▶")
        if self.play_after_id:
            self.root.after_cancel(self.play_after_id)
            self.play_after_id = None

    def _update_preview(self):
        """Update the preview with the current frame"""
        if not self.processor.source_path:
            return
            
        try:
            ret, frame = self.processor.get_frame(self.current_frame_index)
            if ret:
                # Process frame with effects
                processed_frame = self.processor.process_frame(frame)
                
                # Convert to RGB for display
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                
                # Convert to PhotoImage
                from PIL import Image, ImageTk
                img = Image.fromarray(rgb_frame)
                
                # Resize to fit canvas
                canvas_width = self.canvas.winfo_width()
                canvas_height = self.canvas.winfo_height()
                
                if canvas_width > 1 and canvas_height > 1:
                    img_ratio = img.width / img.height
                    canvas_ratio = canvas_width / canvas_height
                    
                    if img_ratio > canvas_ratio:
                        new_width = canvas_width
                        new_height = int(canvas_width / img_ratio)
                    else:
                        new_height = canvas_height
                        new_width = int(canvas_height * img_ratio)
                    
                    img = img.resize((new_width, new_height), Image.LANCZOS)
                
                # Keep reference to avoid garbage collection
                self.preview_frame = ImageTk.PhotoImage(img)
                
                # Update canvas
                self.canvas.delete("all")
                self.canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    anchor=tk.CENTER,
                    image=self.preview_frame
                )
                
                # Update frame label
                self.frame_label.config(text=f"{self.current_frame_index + 1} / {self.processor.frame_count}")
                
                # Force UI update
                self.root.update_idletasks()
        except Exception as e:
            print(f"Preview update error: {e}")
            self._stop_playback()

    def _add_effect(self):
        """Add a new effect"""
        # Create a dialog to select effect type
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Effect")
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Set minimum size
        dialog.minsize(400, 300)
        
        # Create main frame with padding
        main_frame = ttk.Frame(dialog, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(main_frame, text="Select effect type:").pack(pady=10)
        
        # Create frame for listbox and scrollbar
        list_frame = ttk.Frame(main_frame)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        effect_var = tk.StringVar()
        effect_listbox = tk.Listbox(
            list_frame,
            width=50,
            height=15,
            font=('TkDefaultFont', 10),
            yscrollcommand=scrollbar.set
        )
        effect_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=effect_listbox.yview)
        
        # Get available effects
        effect_classes = self.plugin_manager.get_effect_classes()
        effect_names = sorted(effect_classes.keys())
        
        for name in effect_names:
            effect_class = effect_classes[name]
            effect_listbox.insert(tk.END, f"{effect_class.name} - {effect_class.description}")
        
        def on_select():
            if effect_listbox.curselection():
                index = effect_listbox.curselection()[0]
                selected_name = effect_names[index]
                effect_class = effect_classes[selected_name]
                
                # Create new effect instance
                effect = effect_class()
                
                # If this is a PixelSonificationEffect, set the video processor
                if effect.__class__.__name__ == "PixelSonificationEffect":
                    effect.video_processor = self.processor
                
                # Add to chain
                self.processor.effect_chain.add_effect(effect)
                
                # Update UI
                self._refresh_effects_list()
                self._update_preview()
                
                dialog.destroy()
        
        # Create button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(button_frame, text="Add", command=on_select).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _remove_effect(self):
        """Remove selected effect"""
        selection = self.effects_list.selection()
        if selection:
            index = self.effects_list.index(selection[0])
            self.processor.effect_chain.remove_effect(index)
            self._refresh_effects_list()
            self._update_preview()
    
    def _move_effect_up(self):
        """Move selected effect up in the chain"""
        selection = self.effects_list.selection()
        if selection:
            index = self.effects_list.index(selection[0])
            if index > 0:
                self.processor.effect_chain.move_effect(index, index - 1)
                self._refresh_effects_list()
                self._update_preview()
                
                # Reselect the moved item
                item_id = self.effects_list.get_children()[index - 1]
                self.effects_list.selection_set(item_id)
    
    def _move_effect_down(self):
        """Move selected effect down in the chain"""
        selection = self.effects_list.selection()
        if selection:
            index = self.effects_list.index(selection[0])
            if index < len(self.processor.effect_chain.effects) - 1:
                self.processor.effect_chain.move_effect(index, index + 1)
                self._refresh_effects_list()
                self._update_preview()
                
                # Reselect the moved item
                item_id = self.effects_list.get_children()[index + 1]
                self.effects_list.selection_set(item_id)
    
    def _refresh_effects_list(self):
        """Refresh the effects list"""
        # Clear current items
        for item in self.effects_list.get_children():
            self.effects_list.delete(item)
        
        # Add current effects
        for i, effect in enumerate(self.processor.effect_chain.effects):
            self.effects_list.insert("", tk.END, values=(effect.name,))
    
    def _on_effect_selected(self, event):
        """Handle effect selection"""
        selection = self.effects_list.selection()
        if not selection:
            return

        # Clear current parameters UI
        for widget in self.params_frame.winfo_children():
            widget.destroy()

        # Get selected effect
        index = self.effects_list.index(selection[0])
        effect = self.processor.effect_chain.effects[index]

        # Create parameters UI
        for param_name, param_details in effect.parameters.items():
            frame = ttk.Frame(self.params_frame)
            frame.pack(fill=tk.X, pady=4)

            ttk.Label(frame, text=f"{param_name}:").pack(side=tk.LEFT)

            param_type = param_details["type"]
            param_value = effect.params[param_name]
            default_value = param_details.get("default")

            # Display value label
            value_var = tk.StringVar(value=str(param_value))
            value_label = ttk.Label(frame, textvariable=value_var, width=6)
            value_label.pack(side=tk.RIGHT, padx=(5, 0))

            def update_label(val): value_var.set(str(val))

            # Reset button
            def reset_param():
                effect.set_param(param_name, default_value)
                value_var.set(str(default_value))
                self._update_preview()
                self._on_effect_selected(None)  # Refresh controls

            ttk.Button(frame, text="⟲", width=2, command=reset_param).pack(side=tk.RIGHT)

            if param_type in [int, float]:
                var = tk.DoubleVar(value=param_value)

                slider = ttk.Scale(
                    frame,
                    from_=param_details["min"],
                    to=param_details["max"],
                    orient=tk.HORIZONTAL,
                    variable=var
                )
                slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

                def update_param(*_, name=param_name, var=var):
                    val = var.get()
                    if param_type == int:
                        val = int(val)
                    effect.set_param(name, val)
                    update_label(val)
                    self._update_preview()

                var.trace_add("write", update_param)

            elif param_type == bool:
                var = tk.BooleanVar(value=param_value)

                checkbox = ttk.Checkbutton(frame, variable=var)
                checkbox.pack(side=tk.LEFT, padx=5)

                def update_param(*_, name=param_name, var=var):
                    effect.set_param(name, var.get())
                    update_label(var.get())
                    self._update_preview()

                var.trace_add("write", update_param)

            elif param_type == str:
                var = tk.StringVar(value=param_value)

                entry = ttk.Entry(frame, textvariable=var)
                entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

                def update_param(*_, name=param_name, var=var):
                    effect.set_param(name, var.get())
                    update_label(var.get())
                    self._update_preview()

                var.trace_add("write", update_param)

            elif param_type == "choice":
                var = tk.StringVar(value=param_value)
                combobox = ttk.Combobox(frame, textvariable=var, values=param_details["options"], state="readonly")
                combobox.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

                def update_param(name=param_name, var=var):
                    val = var.get()
                    effect.set_param(name, val)
                    update_label(val)
                    self._update_preview()

                combobox.bind("<<ComboboxSelected>>", lambda e: update_param())

    def _on_window_close(self):
        """Handle window closing"""
        try:
            self._stop_playback()
            if self.processor:
                self.processor.release()
            self.root.destroy()
        except Exception as e:
            print(f"Error closing window: {e}")
            self.root.destroy()


# ===== Main Application =====

def main():
    root = tk.Tk()
    root.geometry("1200x800")
    app = VideoGlitchGUI(root)
    
    # Update preview frame size when window resizes
    def on_resize(event):
        if hasattr(app, "_update_preview"):
            app._update_preview()
    
    root.bind("<Configure>", on_resize)
    
    root.mainloop()

if __name__ == "__main__":
    main()
