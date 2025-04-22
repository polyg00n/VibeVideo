"""
Main application window with improved organization and performance.
"""
import tkinter as tk
from tkinter import ttk, filedialog
import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from .preview import VideoPreview
from ..core.processor import VideoProcessor
from ..core.plugin_manager import PluginManager

class MainWindow:
    """Main application window"""
    
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Video Glitch Player")
        
        # Initialize core components
        self.processor = VideoProcessor()
        self.plugin_manager = PluginManager()
        self.plugin_manager.discover_plugins()
        
        # Setup logging
        self._setup_logging()
        
        # Build UI
        self._build_ui()
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self._on_window_close)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self._logger = logging.getLogger(__name__)
    
    def _build_ui(self):
        """Build the main UI"""
        # Main layout using PanedWindow
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        paned = ttk.Panedwindow(main_frame, orient=tk.HORIZONTAL)
        paned.pack(fill=tk.BOTH, expand=True)

        # === Left side: video preview and controls ===
        left_frame = ttk.Frame(paned)
        paned.add(left_frame, weight=3)

        # Video controls
        controls_frame = ttk.LabelFrame(left_frame, text="Video Controls", padding=5)
        controls_frame.pack(fill=tk.X, pady=5)

        ttk.Button(controls_frame, text="Open Video", command=self._open_video).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Export Video", command=self._export_video).pack(side=tk.LEFT, padx=5)

        # Video preview
        preview_frame = ttk.LabelFrame(left_frame, text="Preview", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.preview = VideoPreview(preview_frame)

        # Playback controls
        playback_frame = ttk.Frame(preview_frame)
        playback_frame.pack(fill=tk.X, pady=5)

        self.play_button = ttk.Button(playback_frame, text="▶", width=3, command=self._toggle_playback)
        self.play_button.pack(side=tk.LEFT, padx=5)

        # Frame slider
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

        # Effects chain
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

        # Effect buttons
        effects_buttons_frame = ttk.Frame(effects_frame)
        effects_buttons_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)

        ttk.Button(effects_buttons_frame, text="Add", command=self._add_effect).pack(fill=tk.X, pady=2)
        ttk.Button(effects_buttons_frame, text="Remove", command=self._remove_effect).pack(fill=tk.X, pady=2)
        ttk.Button(effects_buttons_frame, text="Move Up", command=self._move_effect_up).pack(fill=tk.X, pady=2)
        ttk.Button(effects_buttons_frame, text="Move Down", command=self._move_effect_down).pack(fill=tk.X, pady=2)

        # Parameters frame
        self.params_frame = ttk.LabelFrame(right_frame, text="Effect Parameters", padding=5)
        self.params_frame.pack(fill=tk.BOTH, expand=False, pady=5)

        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(fill=tk.X, pady=(5, 0))

        # Playback state
        self.is_playing = False
        self.play_after_id = None
        self.current_frame_index = 0
    
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
            self._logger.error(f"Error opening video: {e}")
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
            # Create progress dialog
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
            
            # Process in background
            def process_thread():
                success = self.processor.process_video(file_path, update_progress)
                self.root.after(0, lambda: self._on_export_complete(progress_window, success, file_path))
            
            import threading
            thread = threading.Thread(target=process_thread)
            thread.daemon = True
            thread.start()
    
    def _on_export_complete(self, progress_window, success, file_path):
        """Handle export completion"""
        progress_window.destroy()
        
        if success:
            self.status_var.set(f"Video exported to: {os.path.basename(file_path)}")
        else:
            self.status_var.set("Error exporting video")
    
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
        """Play the next frame"""
        if not self.is_playing:
            return

        try:
            next_frame = (self.current_frame_index + 1) % self.processor.frame_count
            self.current_frame_index = next_frame
            self.frame_slider.set(next_frame)
            self._update_preview()

            # Calculate delay based on video FPS
            delay = int(1000 / max(1, self.processor.fps))
            self.play_after_id = self.root.after(delay, self._play_next_frame)
        except Exception as e:
            self._logger.error(f"Playback error: {e}")
            self._stop_playback()
    
    def _stop_playback(self):
        """Stop video playback"""
        self.is_playing = False
        self.play_button.configure(text="▶")
        if self.play_after_id:
            self.root.after_cancel(self.play_after_id)
            self.play_after_id = None
    
    def _update_preview(self):
        """Update the preview with current frame"""
        if not self.processor.source_path:
            return
            
        try:
            ret, frame = self.processor.get_frame(self.current_frame_index)
            if ret:
                processed = self.processor.process_frame(frame, self.current_frame_index)
                self.preview.update_frame(processed)
                self.frame_label.config(text=f"{self.current_frame_index + 1} / {self.processor.frame_count}")
        except Exception as e:
            self._logger.error(f"Preview update error: {e}")
            self._stop_playback()
    
    def _on_frame_slider_change(self, value):
        """Handle frame slider change"""
        try:
            new_index = int(float(value))
            if new_index != self.current_frame_index:
                self.current_frame_index = new_index
                self._update_preview()
                
                if self.is_playing:
                    self._stop_playback()
        except Exception as e:
            self._logger.error(f"Slider error: {e}")
            self._stop_playback()
    
    def _add_effect(self):
        """Add a new effect"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Add Effect")
        dialog.transient(self.root)
        dialog.grab_set()
        
        ttk.Label(dialog, text="Select effect type:").pack(pady=10)
        
        effect_listbox = tk.Listbox(dialog, width=40, height=10)
        effect_listbox.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        
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
                
                effect = effect_class()
                self.processor.effect_chain.add_effect(effect)
                
                self._refresh_effects_list()
                self._update_preview()
                
                dialog.destroy()
        
        ttk.Button(dialog, text="Add", command=on_select).pack(pady=10)
    
    def _remove_effect(self):
        """Remove selected effect"""
        selection = self.effects_list.selection()
        if selection:
            index = self.effects_list.index(selection[0])
            self.processor.effect_chain.remove_effect(index)
            self._refresh_effects_list()
            self._update_preview()
    
    def _move_effect_up(self):
        """Move selected effect up"""
        selection = self.effects_list.selection()
        if selection:
            index = self.effects_list.index(selection[0])
            if index > 0:
                self.processor.effect_chain.move_effect(index, index - 1)
                self._refresh_effects_list()
                self._update_preview()
                
                item_id = self.effects_list.get_children()[index - 1]
                self.effects_list.selection_set(item_id)
    
    def _move_effect_down(self):
        """Move selected effect down"""
        selection = self.effects_list.selection()
        if selection:
            index = self.effects_list.index(selection[0])
            if index < len(self.processor.effect_chain.effects) - 1:
                self.processor.effect_chain.move_effect(index, index + 1)
                self._refresh_effects_list()
                self._update_preview()
                
                item_id = self.effects_list.get_children()[index + 1]
                self.effects_list.selection_set(item_id)
    
    def _refresh_effects_list(self):
        """Refresh the effects list"""
        for item in self.effects_list.get_children():
            self.effects_list.delete(item)
        
        for effect in self.processor.effect_chain.effects:
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

            param_type = param_details.type
            param_value = effect.get_param(param_name)
            default_value = param_details.default

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
                self._on_effect_selected(None)

            ttk.Button(frame, text="⟲", width=2, command=reset_param).pack(side=tk.RIGHT)

            if param_type in [int, float]:
                var = tk.DoubleVar(value=param_value)

                slider = ttk.Scale(
                    frame,
                    from_=param_details.min,
                    to=param_details.max,
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
                combobox = ttk.Combobox(frame, textvariable=var, values=param_details.options, state="readonly")
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
            self._logger.error(f"Error closing window: {e}")
            self.root.destroy() 