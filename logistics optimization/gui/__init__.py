# gui/__init__.py
# -*- coding: utf-8 -*-

# Import key components from the gui package to make them available
# when 'gui' is imported or for easier access like 'from gui import MainWindow'

# Import the main window class
from .main_window import MainWindow
# Import the main function used as the entry point for the GUI
from .main_window import main as run_gui

# Import utility functions if they are intended for external use (less common for gui utils)
# from .utils import create_label, create_entry, create_button, create_checkbox

# Define the public API of the gui package using __all__
# This controls what 'from gui import *' imports.
# Typically, you'd export the main window and potentially the run function.
__all__ = [
    "MainWindow",
    "run_gui",
    # Add utils components here if needed:
    # "create_label",
    # "create_entry",
    # "create_button",
    # "create_checkbox",
]