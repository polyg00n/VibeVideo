"""
Main entry point for the Video Glitch Player application.
"""
import tkinter as tk
import logging
from .gui.main_window import MainWindow

def main():
    """Main entry point"""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create main window
    root = tk.Tk()
    root.geometry("1200x800")
    
    # Create application
    app = MainWindow(root)
    
    # Start main loop
    root.mainloop()

if __name__ == "__main__":
    main() 