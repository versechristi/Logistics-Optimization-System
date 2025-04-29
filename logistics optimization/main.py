# main.py
# -*- coding: utf-8 -*-
"""
Main entry point for the Logistics Optimization application.

Sets up the necessary paths and launches the GUI. Ensures only one
instance of the main application runs.
"""

import os
import sys
import traceback

# --- Path Setup ---
# Ensures modules within the project can be found regardless of how main.py is run.
try:
    main_dir = os.path.dirname(os.path.abspath(__file__))
    # Assuming main.py is in the project root alongside 'core', 'gui', etc.
    project_root_main = main_dir
    if project_root_main not in sys.path:
        sys.path.insert(0, project_root_main)
        print(f"Main.py: Added project root to sys.path: {project_root_main}")
    # else: print(f"Main.py: Project root already in sys.path.") # Optional: uncomment for debug
except Exception as path_e:
     print(f"CRITICAL ERROR setting up sys.path in main.py: {path_e}")
     # Attempt to show graphical error before exiting
     try: import tkinter as tk; from tkinter import messagebox; r = tk.Tk(); r.withdraw(); messagebox.showerror("Path Error", f"Failed path setup: {path_e}"); r.destroy()
     except: pass
     sys.exit(1)


# --- Import and Run GUI ---
try:
    # Import the function that initializes and runs the Tkinter main loop
    from gui.main_window import main as run_gui_main
except ImportError as e:
    print(f"CRITICAL ERROR: Failed to import GUI entry point 'main' from 'gui.main_window'.")
    print(f"Error details: {e}")
    print(f"Check module existence, project structure, and sys.path: {sys.path}")
    traceback.print_exc()
    try: import tkinter as tk; from tkinter import messagebox; r = tk.Tk(); r.withdraw(); messagebox.showerror("Import Error", "Failed to load GUI components."); r.destroy()
    except: pass
    sys.exit(1)
except Exception as e:
    print(f"CRITICAL ERROR: An unexpected error occurred during GUI import:")
    traceback.print_exc()
    sys.exit(1)


# --- Main Execution Block ---
if __name__ == "__main__":
    """Ensures this code runs only when main.py is executed directly."""
    print("Main.py: Starting the GUI application...")
    try:
        # Execute the imported GUI main function
        run_gui_main()
    except Exception as e:
        print("CRITICAL ERROR: An exception occurred while running the application:")
        traceback.print_exc()
        try: import tkinter as tk; from tkinter import messagebox; r = tk.Tk(); r.withdraw(); messagebox.showerror("Runtime Error", f"Application error:\n{e}"); r.destroy()
        except: pass
        sys.exit(1)

    print("Main.py: Application exited.")
    sys.exit(0) # Explicit successful exit