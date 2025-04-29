# gui/main_window.py
# -*- coding: utf-8 -*-
"""
Main window class for the Logistics Optimization GUI application,
adapted for Multi-Depot, Two-Echelon VRP with Drones and Split Deliveries (MD-2E-VRPSD).

This file orchestrates the user interface, parameter input, data generation,
algorithm selection, optimization execution, and results visualization.
It interacts with the core modules (route_optimizer, problem_utils, etc.),
algorithm implementations, and visualization/utility modules.

Optimization execution is performed in a separate thread to keep the GUI responsive.
This version integrates with updated backend modules (route_optimizer, plot_generator, report_generator).
"""

# --- Standard Library Imports ---
import tkinter as tk
import traceback
from tkinter import ttk, filedialog, messagebox, font as tkFont, scrolledtext
import numpy as np
import random
import configparser
import os
import copy
import webbrowser
import sys
import math # For checking inf/nan
import threading # Essential for running optimization in a separate thread
import json # For saving/loading complex params in config
import warnings # Using warnings instead of direct prints for non-critical issues
import logging # Added for better logging
import time as pytime
from typing import Dict, List, Any, Optional, Tuple, Union

# --- Setup Logging ---
# Configure logging for the GUI module.
# In a larger app, configure this centrally.
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] (%(module)s) %(message)s"
)
logger = logging.getLogger(__name__)


# --- Matplotlib Imports (Optional, for Plots) ---
_MATPLOTLIB_AVAILABLE = False
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    _MATPLOTLIB_AVAILABLE = True
    logger.info("Matplotlib imported successfully.")
except ImportError:
    logger.warning("Matplotlib not found. Plots will be disabled. Install using: pip install matplotlib")

# --- Path Setup ---
# Ensure the project root directory (containing 'core', 'algorithm', etc.) is in the Python path
_PATH_SETUP_SUCCESS = False
try:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    logger.info(f"Appended project root to sys.path: {project_root}")
    _PATH_SETUP_SUCCESS = True
except Exception as e:
    logger.critical(f"CRITICAL ERROR during path setup: {e}", exc_info=True)


# --- Constants ---
CONFIG_FILENAME = 'default_config_md_2e.ini' # Keep consistent config filename
CONFIG_DIR = os.path.join(project_root, 'config')
CONFIG_FILE_PATH = os.path.join(CONFIG_DIR, CONFIG_FILENAME)
DEFAULT_OUTPUT_DIR = os.path.join(project_root, 'output') # Default base dir

# Ensure base output dir exists on startup, specific run dirs created by optimizer
try:
    os.makedirs(CONFIG_DIR, exist_ok=True)
    os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
except OSError as e:
     logger.error(f"Could not create base config/output directory: {e}")
     # Potentially show error popup here, but let initialization continue

# --- Core Project Module Imports ---
# Use robust try-except blocks and define dummies if imports fail.
_CORE_OPTIMIZER_AVAILABLE = False
_DATA_GENERATOR_AVAILABLE = False
_MAP_GENERATOR_AVAILABLE = False
_PLOT_GENERATOR_AVAILABLE = False
_REPORT_GENERATOR_AVAILABLE = False
_CRITICAL_IMPORT_ERROR = False

# --- Dummy function/class definitions for graceful failure ---
# These allow the GUI to potentially start even if some backend modules fail,
# showing appropriate error messages to the user.

def _dummy_core_run_optimization(*args, **kwargs) -> Dict[str, Any]:
    logger.error("Dummy core_run_optimization called - Core optimizer failed to load.")
    messagebox.showerror("Optimization Error", "Core optimization modules failed to load.")
    return {'overall_status': 'Error', 'error_message': 'Core modules failed to load.', 'results_by_algorithm': {}}

class _dummy_SolutionCandidate:
    # Minimal dummy just for type hints if needed, not functional
    pass

def _dummy_generate_locations(*args, **kwargs) -> Dict[str, List]:
    logger.error("Dummy generate_locations called - Data generator failed to load.")
    messagebox.showerror("Data Error", "Location generation module failed to load.")
    return {"logistics_centers": [], "sales_outlets": [], "customers": []}

def _dummy_generate_demand(*args, **kwargs) -> List[float]:
    logger.error("Dummy generate_demand called - Data generator failed to load.")
    messagebox.showerror("Data Error", "Demand generation module failed to load.")
    return []

def _dummy_generate_folium_map(*args, **kwargs) -> Optional[str]:
    logger.warning("Dummy generate_folium_map called - Map generator unavailable.")
    return None

def _dummy_open_map_in_browser(*args, **kwargs) -> bool:
    logger.warning("Dummy open_map_in_browser called - Map generator unavailable.")
    return False

class _dummy_PlotGenerator:
    """Dummy PlotGenerator if visualization fails to load."""
    def __init__(self): logger.warning("Using dummy PlotGenerator.")
    def plot_iteration_curves(self, results_by_algorithm: Dict, ax: Any):
         logger.warning("Dummy plot_iteration_curves called.")
         if ax:
             try:
                 ax.clear()
                 ax.text(0.5, 0.5, 'Plotting Disabled\n(Check Matplotlib/PlotGenerator)', ha='center', va='center', transform=ax.transAxes, color='grey')
                 ax.figure.canvas.draw_idle()
             except Exception: pass # Ignore errors in dummy
    def plot_comparison_bars(self, results_by_algorithm: Dict, ax_cost: Any, ax_time: Any):
         logger.warning("Dummy plot_comparison_bars called.")
         for ax in [ax_cost, ax_time]:
              if ax:
                  try:
                      ax.clear()
                      ax.text(0.5, 0.5, 'Plotting Disabled', ha='center', va='center', transform=ax.transAxes, color='grey')
                      ax.figure.canvas.draw_idle()
                  except Exception: pass # Ignore errors in dummy

def _dummy_generate_delivery_report(*args, **kwargs) -> str:
    logger.warning("Dummy generate_delivery_report called - Report generator unavailable.")
    return "Report generation module not available."

# --- Attempt actual imports ---
try:
    logger.info("Importing core/data/visualization modules...")
    # Import the main optimization orchestrator (updated version)
    from core.route_optimizer import run_optimization as core_run_optimization
    _CORE_OPTIMIZER_AVAILABLE = True
    logger.info("core.route_optimizer imported.")

    # Import core problem data structures if needed directly (less likely now)
    # from core.problem_utils import SolutionCandidate # Example if needed

    from data.data_generator import generate_locations, generate_demand
    _DATA_GENERATOR_AVAILABLE = True
    logger.info("data.data_generator imported.")

    # Import visualization components
    try:
        # Import updated map generator
        from visualization.map_generator import generate_folium_map, open_map_in_browser
        _MAP_GENERATOR_AVAILABLE = True
        logger.info("visualization.map_generator imported.")
    except ImportError as e:
        logger.warning(f"Failed to import map_generator (requires 'folium'): {e}. Interactive maps disabled.")
        if "folium" in str(e):
             messagebox.showinfo("Missing Dependency", "Python library 'folium' not found.\nInteractive map generation will be disabled.\nInstall using: pip install folium", icon='warning')
        generate_folium_map = _dummy_generate_folium_map
        open_map_in_browser = _dummy_open_map_in_browser

    # Import plot generator (depends on Matplotlib)
    if _MATPLOTLIB_AVAILABLE:
        try:
            # Import updated plot generator
            from visualization.plot_generator import PlotGenerator
            _PLOT_GENERATOR_AVAILABLE = True
            logger.info("visualization.plot_generator imported.")
        except ImportError as e:
            logger.warning(f"Failed to import plot_generator: {e}. Plots disabled.")
            PlotGenerator = _dummy_PlotGenerator # Fallback to dummy
    else:
         PlotGenerator = _dummy_PlotGenerator # Use dummy if matplotlib itself is missing

    # Import report generator
    try:
        # Import updated report generator
        from utils.report_generator import generate_delivery_report
        _REPORT_GENERATOR_AVAILABLE = True
        logger.info("utils.report_generator imported.")
    except ImportError as e:
        logger.warning(f"Failed to import report_generator: {e}. Report tab disabled.")
        generate_delivery_report = _dummy_generate_delivery_report

    logger.info("Core/data/visualization modules import process complete.")

except ImportError as e:
    error_msg = f"CRITICAL ERROR: Failed to import core/data base modules: {e}"
    logger.critical(error_msg, exc_info=True)
    _CRITICAL_IMPORT_ERROR = True
    # Assign dummy functions for critical parts
    core_run_optimization = _dummy_core_run_optimization
    # SolutionCandidate = _dummy_SolutionCandidate # If needed
    generate_locations = _dummy_generate_locations
    generate_demand = _dummy_generate_demand
    generate_folium_map = _dummy_generate_folium_map
    open_map_in_browser = _dummy_open_map_in_browser
    PlotGenerator = _dummy_PlotGenerator
    generate_delivery_report = _dummy_generate_delivery_report


# --- GUI Helper Functions (Integrated from gui.utils for self-containment) ---
# (These remain unchanged to preserve UI structure)
def create_label(parent, text, row, col, sticky="w", padx=2, pady=2, columnspan=1, **kwargs):
    """Creates and grids a ttk.Label."""
    label = ttk.Label(parent, text=text, **kwargs)
    label.grid(row=row, column=col, sticky=sticky, padx=padx, pady=pady, columnspan=columnspan)
    return label

def create_entry(parent, textvariable, row, col, sticky="ew", padx=2, pady=2, columnspan=1, width=10, **kwargs):
    """Creates and grids a ttk.Entry."""
    entry = ttk.Entry(parent, textvariable=textvariable, width=width, **kwargs)
    entry.grid(row=row, column=col, sticky=sticky, padx=padx, pady=pady, columnspan=columnspan)
    return entry

def create_button(parent, text, command, row, col, sticky="ew", padx=2, pady=2, columnspan=1, **kwargs):
    """Creates and grids a ttk.Button."""
    button = ttk.Button(parent, text=text, command=command, **kwargs)
    button.grid(row=row, column=col, sticky=sticky, padx=padx, pady=pady, columnspan=columnspan)
    return button

def create_checkbox(parent, text, variable, row, col, sticky="w", padx=2, pady=2, columnspan=1, **kwargs):
    """Creates and grids a ttk.Checkbutton."""
    checkbox = ttk.Checkbutton(parent, text=text, variable=variable, **kwargs)
    checkbox.grid(row=row, column=col, sticky=sticky, padx=padx, pady=pady, columnspan=columnspan)
    return checkbox

def create_separator(parent, row, col, columnspan=2, orient=tk.HORIZONTAL, pady=5, **kwargs):
    """Creates and grids a ttk.Separator."""
    sep = ttk.Separator(parent, orient=orient, **kwargs)
    sep.grid(row=row, column=col, columnspan=columnspan, sticky="ew", pady=pady)
    return sep


# --- MainWindow Class Definition ---
class MainWindow(tk.Tk):
    """
    The main application window for the Logistics Optimization GUI.
    Manages the UI layout, user input, triggers optimization runs,
    and displays results and visualizations for MD-2E-VRPSD.
    """
    def __init__(self):
        """Initializes the main window and its components."""
        # --- Critical Initialization Checks ---
        if not _PATH_SETUP_SUCCESS:
            self.show_critical_error_and_exit(
                "Fatal Path Error",
                "Could not configure project paths.\nApplication cannot start.\nPlease check console output."
            )
            return # Stop initialization
        if _CRITICAL_IMPORT_ERROR:
            self.show_critical_error_and_exit(
                "Fatal Import Error",
                "Failed to import critical core/data modules.\n"
                "The application cannot start.\n"
                "Please check console output, project structure, and dependencies."
            )
            return # Stop initialization

        logger.info("Initializing MainWindow...")
        super().__init__() # Initialize the Tkinter window
        logger.info("tk.Tk initialized.")

        self.title("Logistics Optimization System (MD-2E-VRPSD)")
        self.geometry("1450x900") # Keep original size
        self.protocol("WM_DELETE_WINDOW", self._on_closing)

        # --- Class Attributes ---
        self.config = configparser.ConfigParser(interpolation=None)
        self.config_file = CONFIG_FILE_PATH

        # --- Data Storage ---
        self.current_problem_data: Dict[str, Any] = {"locations": {}, "demands": []}
        self.current_optimization_results: Optional[Dict[str, Any]] = None # Stores results from route_optimizer
        self.available_algorithms: Dict[str, str] = { # Map internal key to display name
            'greedy_heuristic': 'Greedy Heuristic',
            'genetic_algorithm': 'Genetic Algorithm',
            'simulated_annealing': 'Simulated Annealing',
            'pso_optimizer': 'PSO',
        }
        # Check which algorithms actually loaded
        algorithm_module = sys.modules.get('algorithm')  # 先获取模块对象
        if algorithm_module:  # 确保模块已加载
            self.loaded_algorithms = {k: v for k, v in self.available_algorithms.items() if
                                      hasattr(algorithm_module, f'run_{k}')}
        else:
            logger.error("Algorithm package not found in sys.modules.")  # 添加日志记录
            self.loaded_algorithms = {}  # 如果模块没加载，则加载的算法为空
        if not self.loaded_algorithms:
            messagebox.showerror("Algorithm Error", "No optimization algorithms loaded successfully. Optimization will not be possible.", icon='error')

        self.selected_algorithms_vars = {key: tk.BooleanVar(value=(key == 'genetic_algorithm')) for key in self.available_algorithms} # Default GA selected


        # --- Tkinter Variables ---
        self._initialize_tk_variables()

        # --- Configuration and Directories ---
        # Directories already created above or by optimizer
        self._ensure_config_exists() # Create default if needed
        self.load_configuration() # Load config values into tk.Vars

        # --- Visualization Components ---
        # Instantiate plot generator (real or dummy)
        self.plot_generator = PlotGenerator() if _PLOT_GENERATOR_AVAILABLE else _dummy_PlotGenerator()
        self.history_figure: Optional[plt.Figure] = None
        self.history_ax: Optional[plt.Axes] = None
        self.history_canvas_agg: Optional[FigureCanvasTkAgg] = None
        self.comparison_figure: Optional[plt.Figure] = None
        self.comp_ax_cost: Optional[plt.Axes] = None
        self.comp_ax_time: Optional[plt.Axes] = None
        self.comparison_canvas_agg: Optional[FigureCanvasTkAgg] = None

        # --- State Variables ---
        self.optimization_thread: Optional[threading.Thread] = None
        self.status_var = tk.StringVar(value="Ready")

        # --- GUI Layout Setup ---
        logger.info("Setting up UI...")
        self._set_matplotlib_defaults() # Set defaults if available
        self.create_widgets() # Create all GUI elements (layout unchanged)
        logger.info("UI setup complete.")

        # --- Initial Plot Placeholders ---
        if _MATPLOTLIB_AVAILABLE:
            self._create_initial_plot_placeholders()
        else:
             # Display unavailable message in plot tabs
             self._display_matplotlib_unavailable_message(target_frame=getattr(self, 'iter_canvas_widget_container', None))
             self._display_matplotlib_unavailable_message(target_frame=getattr(self, 'comp_canvas_widget_container', None))

        # --- Final Checks & Warnings ---
        if not _CORE_OPTIMIZER_AVAILABLE:
            messagebox.showwarning("Core Unavailable", "Core optimization function failed to load. Optimization disabled.", icon='warning')
            if hasattr(self, 'run_button'): self.run_button.config(state=tk.DISABLED)
        if not _DATA_GENERATOR_AVAILABLE:
            messagebox.showwarning("Data Gen Unavailable", "Data generation functions failed to load. Data generation disabled.", icon='warning')
            if hasattr(self, 'generate_button'): self.generate_button.config(state=tk.DISABLED)

        logger.info("MainWindow initialization complete.")


    def show_critical_error_and_exit(self, title: str, message: str):
        """Shows a critical error message box and attempts to exit."""
        logger.critical(f"{title}: {message}")
        try:
            # Create a temporary root to show the message box if the main one failed
            root_err = tk.Tk()
            root_err.withdraw()
            messagebox.showerror(title, message, parent=None, icon='error')
            root_err.destroy()
        except Exception as e:
            logger.error(f"Could not display critical error message box: {e}")
        sys.exit(1) # Exit the script

    # --- Widget Creation Methods ---
    # (Keep these methods largely unchanged to preserve UI layout)

    def create_widgets(self):
        """Creates all the main widgets for the window layout."""
        main_paned_window = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main_paned_window.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_panel_container = ttk.Frame(main_paned_window, width=450)
        left_panel_container.pack_propagate(False)
        main_paned_window.add(left_panel_container, weight=1)

        left_panel_container.grid_rowconfigure(0, weight=1)
        left_panel_container.grid_columnconfigure(0, weight=1)

        left_canvas = tk.Canvas(left_panel_container, borderwidth=0, highlightthickness=0)
        left_scrollbar = ttk.Scrollbar(left_panel_container, orient="vertical", command=left_canvas.yview)
        self.scrollable_frame = ttk.Frame(left_canvas)

        self.scrollable_frame.bind("<Configure>", lambda e: left_canvas.configure(scrollregion=left_canvas.bbox("all")))
        # Bind mouse wheel scrolling for common platforms
        def _on_mousewheel(event):
            # Determine scroll direction and magnitude (platform dependent)
            if event.num == 5 or event.delta < 0: # Linux wheel down, Windows/macOS wheel down
                delta = 1
            elif event.num == 4 or event.delta > 0: # Linux wheel up, Windows/macOS wheel up
                delta = -1
            else: delta = 0
            left_canvas.yview_scroll(delta, "units")
        # Bind for Linux (button 4/5) and Windows/macOS (MouseWheel)
        left_canvas.bind("<Button-4>", _on_mousewheel)
        left_canvas.bind("<Button-5>", _on_mousewheel)
        left_canvas.bind("<MouseWheel>", _on_mousewheel)
        # Bind scrollable frame too, in case focus is there
        self.scrollable_frame.bind("<MouseWheel>", _on_mousewheel)


        left_canvas_window_id = left_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        left_canvas.configure(yscrollcommand=left_scrollbar.set)

        left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scrollbar.grid(row=0, column=1, sticky="ns")
        self.scrollable_frame.grid_columnconfigure(0, weight=1)

        # --- Widgets inside the scrollable_frame ---
        current_row = 0
        self.param_notebook = ttk.Notebook(self.scrollable_frame)
        self.param_notebook.grid(row=current_row, column=0, sticky="nsew", padx=5, pady=(5,10))
        self.param_notebook.enable_traversal()
        current_row += 1
        self.create_parameter_tabs(self.param_notebook)

        self.algo_params_notebook = ttk.Notebook(self.scrollable_frame)
        self.algo_params_notebook.grid(row=current_row, column=0, sticky="nsew", padx=5, pady=5)
        self.algo_params_notebook.enable_traversal()
        current_row += 1
        self.create_algorithm_params_widgets(self.algo_params_notebook)

        data_action_frame = ttk.Frame(self.scrollable_frame)
        data_action_frame.grid(row=current_row, column=0, sticky="ew", padx=5, pady=5)
        data_action_frame.grid_columnconfigure(0, weight=1)
        data_action_frame.grid_columnconfigure(1, weight=1)
        data_action_frame.grid_columnconfigure(2, weight=1)
        current_row += 1

        style = ttk.Style()
        style.configure("Accent.TButton", font=('Segoe UI', 9, 'bold'), foreground="white", background="#0078D4") # Example accent style

        self.generate_button = create_button(data_action_frame, "Generate Data", self._generate_data, 0, 0, sticky="ew", padx=(0,2))
        gen_btn_state = tk.DISABLED if not _DATA_GENERATOR_AVAILABLE else tk.NORMAL
        self.generate_button.config(state=gen_btn_state)
        try: self.generate_button.configure(style="Accent.TButton")
        except tk.TclError: pass # Ignore style errors if theme doesn't support it

        self.load_config_button = create_button(data_action_frame, "Load Config", self._load_config_dialog, 0, 1, sticky="ew", padx=2)
        self.save_config_button = create_button(data_action_frame, "Save Config", self._save_config_dialog, 0, 2, sticky="ew", padx=(2,0))

        self.create_algorithm_selection_widgets(self.scrollable_frame, current_row)
        current_row += 1

        self.status_label = create_label(self.scrollable_frame, "", current_row, 0, sticky="ew", textvariable=self.status_var, relief=tk.SUNKEN, anchor='w', wraplength=400) # Allow wrapping
        current_row += 1

        # --- Right Results Panel ---
        right_panel = ttk.Frame(main_paned_window, borderwidth=1, relief=tk.SUNKEN)
        main_paned_window.add(right_panel, weight=3)

        right_panel.grid_rowconfigure(0, weight=1)
        right_panel.grid_columnconfigure(0, weight=1)

        self.results_notebook = ttk.Notebook(right_panel)
        self.results_notebook.grid(row=0, column=0, sticky="nsew")
        self.results_notebook.enable_traversal()
        self.create_results_tabs(self.results_notebook)

    def create_parameter_tabs(self, parent_notebook):
        """Creates the main parameter input tabs."""
        self.data_gen_frame = ttk.Frame(parent_notebook, padding="10")
        parent_notebook.add(self.data_gen_frame, text="Data Generation")
        self.create_data_generation_widgets(self.data_gen_frame)

        self.vehicle_params_frame = ttk.Frame(parent_notebook, padding="10")
        parent_notebook.add(self.vehicle_params_frame, text="Vehicle")
        self.create_vehicle_params_widgets(self.vehicle_params_frame)

        self.drone_params_frame = ttk.Frame(parent_notebook, padding="10")
        parent_notebook.add(self.drone_params_frame, text="Drone")
        self.create_drone_params_widgets(self.drone_params_frame)

        self.objective_func_frame = ttk.Frame(parent_notebook, padding="10")
        parent_notebook.add(self.objective_func_frame, text="Objective")
        self.create_objective_func_widgets(self.objective_func_frame)

    def create_data_generation_widgets(self, parent_frame):
        """Widgets for the Data Generation tab."""
        parent_frame.grid_columnconfigure(1, weight=1)
        r = 0
        create_label(parent_frame, "Logistics Centers:", r, 0); create_entry(parent_frame, self.num_logistics_centers_var, r, 1, width=7); r+=1
        create_label(parent_frame, "Sales Outlets:", r, 0); create_entry(parent_frame, self.num_sales_outlets_var, r, 1, width=7); r+=1
        create_label(parent_frame, "Customers:", r, 0); create_entry(parent_frame, self.num_customers_var, r, 1, width=7); r+=1
        create_separator(parent_frame, r, 0, columnspan=2) ; r+=1
        create_label(parent_frame, "Min Demand:", r, 0); create_entry(parent_frame, self.min_demand_var, r, 1, width=7); r+=1
        create_label(parent_frame, "Max Demand:", r, 0); create_entry(parent_frame, self.max_demand_var, r, 1, width=7); r+=1
        create_separator(parent_frame, r, 0, columnspan=2) ; r+=1
        create_checkbox(parent_frame, "Solomon-like Distr.", self.use_solomon_like_var, r, 0, columnspan=2, sticky="w"); r+=1
        create_label(parent_frame, "Center Lat:", r, 0); create_entry(parent_frame, self.center_latitude_var, r, 1, width=12); r+=1
        create_label(parent_frame, "Center Lon:", r, 0); create_entry(parent_frame, self.center_longitude_var, r, 1, width=12); r+=1
        create_label(parent_frame, "Radius (km):", r, 0); create_entry(parent_frame, self.radius_km_var, r, 1, width=7); r+=1

    def create_vehicle_params_widgets(self, parent_frame):
        """Widgets for the Vehicle Parameters tab."""
        parent_frame.grid_columnconfigure(1, weight=1)
        r = 0
        create_label(parent_frame, "Max Payload (kg):", r, 0); create_entry(parent_frame, self.vehicle_payload_var, r, 1); r+=1
        create_label(parent_frame, "Cost per km:", r, 0); create_entry(parent_frame, self.vehicle_cost_var, r, 1); r+=1
        create_label(parent_frame, "Speed (km/h):", r, 0); create_entry(parent_frame, self.vehicle_speed_var, r, 1); r+=1

    def create_drone_params_widgets(self, parent_frame):
        """Widgets for the Drone Parameters tab."""
        parent_frame.grid_columnconfigure(1, weight=1)
        r = 0
        create_label(parent_frame, "Max Payload (kg):", r, 0); create_entry(parent_frame, self.drone_payload_var, r, 1); r+=1
        create_label(parent_frame, "Cost per km:", r, 0); create_entry(parent_frame, self.drone_cost_var, r, 1); r+=1
        create_label(parent_frame, "Speed (km/h):", r, 0); create_entry(parent_frame, self.drone_speed_var, r, 1); r+=1
        create_label(parent_frame, "Max Range (km):", r, 0); create_entry(parent_frame, self.drone_range_var, r, 1); r+=1

    def create_objective_func_widgets(self, parent_frame):
        """Widgets for Objective Function Weights and Penalty tab."""
        parent_frame.grid_columnconfigure(1, weight=1)
        parent_frame.grid_columnconfigure(2, weight=0)
        r = 0
        create_label(parent_frame, "Cost Weight:", r, 0)
        cost_scale = ttk.Scale(parent_frame, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.cost_weight_var, length=150)
        cost_scale.grid(row=r, column=1, padx=(0,5), pady=2, sticky="ew")
        create_entry(parent_frame, self.cost_weight_var, r, 2, width=6); r+=1

        create_label(parent_frame, "Time Weight:", r, 0)
        time_scale = ttk.Scale(parent_frame, from_=0, to=1, orient=tk.HORIZONTAL, variable=self.time_weight_var, length=150)
        time_scale.grid(row=r, column=1, padx=(0,5), pady=2, sticky="ew")
        create_entry(parent_frame, self.time_weight_var, r, 2, width=6); r+=1

        create_label(parent_frame, "Unmet Penalty:", r, 0)
        create_entry(parent_frame, self.unmet_demand_penalty_var, r, 1, columnspan=2, width=10); r+=1
        create_label(parent_frame, "(High value -> prioritize feasibility)", r, 0, columnspan=3, sticky="w", font=('TkDefaultFont', 8))

    def create_algorithm_params_widgets(self, parent_notebook):
        """Widgets for the Algorithm Parameters notebook (nested tabs)."""
        # Only create tabs for algorithms that were successfully loaded
        if 'genetic_algorithm' in self.loaded_algorithms:
            ga_tab = ttk.Frame(parent_notebook, padding="10")
            parent_notebook.add(ga_tab, text="GA")
            self._setup_ga_params_tab(ga_tab)

        if 'simulated_annealing' in self.loaded_algorithms:
            sa_tab = ttk.Frame(parent_notebook, padding="10")
            parent_notebook.add(sa_tab, text="SA")
            self._setup_sa_params_tab(sa_tab)

        if 'pso_optimizer' in self.loaded_algorithms:
            pso_tab = ttk.Frame(parent_notebook, padding="10")
            parent_notebook.add(pso_tab, text="PSO")
            self._setup_pso_params_tab(pso_tab)

        if 'greedy_heuristic' in self.loaded_algorithms:
            greedy_tab = ttk.Frame(parent_notebook, padding="10")
            parent_notebook.add(greedy_tab, text="Greedy")
            create_label(greedy_tab, "No specific parameters for Greedy Heuristic.", 0, 0)

        if not parent_notebook.tabs():
             # If no algorithm tabs were added (e.g., none loaded)
             no_algo_frame = ttk.Frame(parent_notebook, padding="10")
             parent_notebook.add(no_algo_frame, text="Algorithms")
             create_label(no_algo_frame, "No algorithms loaded successfully.", 0, 0, foreground="red")


    def _setup_ga_params_tab(self, parent_frame):
        """Sets up the widgets for the GA parameters tab."""
        parent_frame.grid_columnconfigure(1, weight=1)
        r = 0
        create_label(parent_frame, "Population Size:", r, 0); create_entry(parent_frame, self.ga_pop_size_var, r, 1); r+=1
        create_label(parent_frame, "Generations:", r, 0); create_entry(parent_frame, self.ga_gens_var, r, 1); r+=1
        create_label(parent_frame, "Mutation Rate:", r, 0); create_entry(parent_frame, self.ga_mut_rate_var, r, 1); r+=1
        create_label(parent_frame, "Crossover Rate:", r, 0); create_entry(parent_frame, self.ga_cross_rate_var, r, 1); r+=1
        create_label(parent_frame, "Elitism Count:", r, 0); create_entry(parent_frame, self.ga_elitism_var, r, 1); r+=1
        create_label(parent_frame, "Tournament Size:", r, 0); create_entry(parent_frame, self.ga_tourn_size_var, r, 1); r+=1

    def _setup_sa_params_tab(self, parent_frame):
        """Sets up the widgets for the SA parameters tab."""
        parent_frame.grid_columnconfigure(1, weight=1)
        r = 0
        create_label(parent_frame, "Initial Temp:", r, 0); create_entry(parent_frame, self.sa_init_temp_var, r, 1); r+=1
        create_label(parent_frame, "Cooling Rate:", r, 0); create_entry(parent_frame, self.sa_cool_rate_var, r, 1); r+=1
        create_label(parent_frame, "Max Iterations:", r, 0); create_entry(parent_frame, self.sa_iters_var, r, 1); r+=1
        create_label(parent_frame, "Min Temp:", r, 0); create_entry(parent_frame, self.sa_min_temp_var, r, 1); r+=1

    def _setup_pso_params_tab(self, parent_frame):
        """Sets up the widgets for the PSO parameters tab."""
        parent_frame.grid_columnconfigure(1, weight=1)
        r = 0
        create_label(parent_frame, "Swarm Size:", r, 0); create_entry(parent_frame, self.pso_swarm_size_var, r, 1); r+=1
        create_label(parent_frame, "Max Iterations:", r, 0); create_entry(parent_frame, self.pso_max_iters_var, r, 1); r+=1
        create_label(parent_frame, "Inertia (w):", r, 0); create_entry(parent_frame, self.pso_inertia_var, r, 1); r+=1
        create_label(parent_frame, "Cognitive (c1):", r, 0); create_entry(parent_frame, self.pso_cognitive_var, r, 1); r+=1
        create_label(parent_frame, "Social (c2):", r, 0); create_entry(parent_frame, self.pso_social_var, r, 1); r+=1

    def create_algorithm_selection_widgets(self, parent_frame, start_row):
        """Widgets for Algorithm Selection Checkboxes and Run button."""
        self.algo_frame = ttk.LabelFrame(parent_frame, text="Algorithm Selection & Execution", padding="10")
        self.algo_frame.grid(row=start_row, column=0, sticky="nsew", padx=5, pady=5)
        self.algo_frame.grid_columnconfigure(0, weight=0)
        self.algo_frame.grid_columnconfigure(1, weight=1)

        create_label(self.algo_frame, "Select Algorithms to Run:", 0, 0, columnspan=2, sticky='w')

        current_row = 1
        num_cols = 2
        col_idx = 0
        # Use self.loaded_algorithms here
        if not self.loaded_algorithms:
             create_label(self.algo_frame, "No algorithms loaded.", current_row, 0, columnspan=num_cols, sticky='w', foreground='red')
             current_row += 1
        else:
            for key, display_name in self.loaded_algorithms.items():
                var = self.selected_algorithms_vars[key] # Get var using key
                cb = create_checkbox(self.algo_frame, display_name, var, current_row, col_idx, sticky='w')
                col_idx += 1
                if col_idx >= num_cols:
                    col_idx = 0
                    current_row += 1
            if col_idx != 0:
                 current_row += 1

        # Add Run Button
        self.run_button = create_button(self.algo_frame, "Run Optimization", self._run_optimization, current_row, 0, columnspan=num_cols, sticky="ew", pady=(10,0))
        try: self.run_button.configure(style="Accent.TButton")
        except tk.TclError: pass

        # Disable run button if core optimizer or algorithms are unavailable
        run_btn_state = tk.DISABLED if not _CORE_OPTIMIZER_AVAILABLE or not self.loaded_algorithms else tk.NORMAL
        self.run_button.config(state=run_btn_state)

    def create_results_tabs(self, parent_notebook):
        """Creates the tabs for displaying results."""
        # Route Map Tab (Focus on launching Folium map)
        self.map_frame = ttk.Frame(parent_notebook, padding="10")
        parent_notebook.add(self.map_frame, text="Route Map")
        self.create_map_display_widgets(self.map_frame)

        # Iteration Curve Tab (Matplotlib Plot)
        self.iter_curve_frame = ttk.Frame(parent_notebook, padding="2")
        parent_notebook.add(self.iter_curve_frame, text="Iteration Curves")
        self.iter_curve_frame.grid_rowconfigure(0, weight=1)
        self.iter_curve_frame.grid_columnconfigure(0, weight=1)
        self.iter_canvas_widget_container = ttk.Frame(self.iter_curve_frame) # Container for canvas+toolbar
        self.iter_canvas_widget_container.grid(row=0, column=0, sticky="nsew")
        iter_control_frame = ttk.Frame(self.iter_curve_frame, padding="2")
        iter_control_frame.grid(row=1, column=0, sticky="ew")
        iter_control_frame.grid_columnconfigure(0, weight=1)
        self.save_iter_button = create_button(iter_control_frame, "Save Plot", self.save_iteration_plot, 0, 0, sticky="e", state=tk.DISABLED)


        # Results Comparison Tab (Matplotlib Plot)
        self.results_comparison_frame = ttk.Frame(parent_notebook, padding="2")
        parent_notebook.add(self.results_comparison_frame, text="Comparison")
        self.results_comparison_frame.grid_rowconfigure(0, weight=1)
        self.results_comparison_frame.grid_columnconfigure(0, weight=1)
        self.comp_canvas_widget_container = ttk.Frame(self.results_comparison_frame) # Container for canvas+toolbar
        self.comp_canvas_widget_container.grid(row=0, column=0, sticky="nsew")
        comp_control_frame = ttk.Frame(self.results_comparison_frame, padding="2")
        comp_control_frame.grid(row=1, column=0, sticky="ew")
        comp_control_frame.grid_columnconfigure(0, weight=1)
        self.save_comp_button = create_button(comp_control_frame, "Save Plot", self.save_comparison_plot, 0, 0, sticky="e", state=tk.DISABLED)

        # Report Tab
        self.report_frame = ttk.Frame(parent_notebook, padding="2")
        parent_notebook.add(self.report_frame, text="Detailed Report")
        self.create_report_widgets(self.report_frame)

    def create_map_display_widgets(self, parent_frame):
        """Widgets for the Route Map tab (launching Folium)."""
        parent_frame.grid_columnconfigure(1, weight=1)
        r = 0
        create_label(parent_frame, "Select Result Map:", r, 0, sticky='w')

        self.map_selection_var = tk.StringVar(value="Select Result...")
        initial_map_options = ["Select Result..."]
        self.map_selection_menu = ttk.Combobox(parent_frame, textvariable=self.map_selection_var, values=initial_map_options, state='readonly')
        self.map_selection_menu.grid(row=r, column=1, sticky='ew', padx=5)
        self.map_selection_menu.bind("<<ComboboxSelected>>", self._on_map_selection)
        self.map_selection_menu['state'] = 'disabled'
        r += 1

        self.open_map_button = create_button(parent_frame, "Open Selected Map", self._open_selected_map, r, 0, columnspan=2, sticky='ew', pady=5)
        self.open_map_button['state'] = 'disabled'
        r += 1

        self.map_file_path_var = tk.StringVar(value="")
        create_label(parent_frame, "Map File Path:", r, 0, sticky='w')
        map_path_entry = create_entry(parent_frame, self.map_file_path_var, r, 1, state='readonly')
        r += 1

        self.map_info_label = create_label(parent_frame, "Generate data and run optimization to view maps.", r, 0, columnspan=2, sticky='w', foreground='grey', wraplength=parent_frame.winfo_width()-20)

        # Disable if map generator unavailable
        if not _MAP_GENERATOR_AVAILABLE:
             self.map_selection_menu.config(state=tk.DISABLED)
             self.open_map_button.config(state=tk.DISABLED)
             self.map_info_label.config(text="Map generator library (folium) not found or failed to load. Maps unavailable.", foreground='red')


    def create_report_widgets(self, parent_frame):
        """Widgets for the Report tab."""
        parent_frame.grid_rowconfigure(1, weight=1)
        parent_frame.grid_columnconfigure(0, weight=1)

        control_frame = ttk.Frame(parent_frame)
        control_frame.grid(row=0, column=0, sticky='ew', pady=(0,5))
        control_frame.grid_columnconfigure(1, weight=1)

        create_label(control_frame, "Select Report:", 0, 0)
        self.report_selection_var = tk.StringVar(value="Select Result...")
        initial_report_options = ["Select Result..."]
        self.report_selection_menu = ttk.Combobox(control_frame, textvariable=self.report_selection_var, values=initial_report_options, state='readonly')
        self.report_selection_menu.grid(row=0, column=1, sticky='ew', padx=5)
        self.report_selection_menu.bind("<<ComboboxSelected>>", self._on_report_selection)
        self.report_selection_menu['state'] = 'disabled'

        self.save_report_button = create_button(control_frame, "Save Report", self.save_report, 0, 2, sticky='e', state=tk.DISABLED)

        # ScrolledText for display
        try:
             mono_font = tkFont.nametofont("TkFixedFont")
        except tk.TclError:
            # Fallback font if TkFixedFont doesn't exist
            mono_font = ("Courier New", 9) if "Courier New" in tkFont.families() else ("Monospace", 9)
        # Try common monospace fonts for better appearance
        try:
             families = tkFont.families()
             if "Consolas" in families: mono_font = ("Consolas", 9)
             elif "Courier New" in families: mono_font = ("Courier New", 9)
             elif "Menlo" in families: mono_font = ("Menlo", 10) # macOS
             elif "DejaVu Sans Mono" in families: mono_font = ("DejaVu Sans Mono", 9) # Linux
        except Exception: pass # Ignore errors finding specific fonts

        self.report_display_area = scrolledtext.ScrolledText(
            parent_frame, wrap="none", height=10, relief=tk.FLAT,
            borderwidth=1, font=mono_font, state=tk.DISABLED # Start disabled
        )
        self.report_display_area.grid(row=1, column=0, sticky="nsew", padx=2, pady=2)

        # Disable if report generator unavailable
        if not _REPORT_GENERATOR_AVAILABLE:
            self.report_selection_menu.config(state=tk.DISABLED)
            self.save_report_button.config(state=tk.DISABLED)
            self._update_report_area("Report generator module not found.\nReporting is disabled.", clear=True)


    # --- Initialization and Configuration Methods ---
    # (Keep these methods largely unchanged, minor logging additions)

    def _initialize_tk_variables(self):
        """Initializes all Tkinter variables with default values."""
        logger.debug("Initializing Tk variables with default values...")
        # Data Generation
        self.num_logistics_centers_var = tk.IntVar(value=2); self.num_sales_outlets_var = tk.IntVar(value=10)
        self.num_customers_var = tk.IntVar(value=50); self.use_solomon_like_var = tk.BooleanVar(value=False)
        self.center_latitude_var = tk.DoubleVar(value=39.9042); self.center_longitude_var = tk.DoubleVar(value=116.4074)
        self.radius_km_var = tk.DoubleVar(value=15.0); self.min_demand_var = tk.DoubleVar(value=5.0)
        self.max_demand_var = tk.DoubleVar(value=25.0)
        # Vehicle
        self.vehicle_payload_var = tk.DoubleVar(value=100.0); self.vehicle_cost_var = tk.DoubleVar(value=2.0)
        self.vehicle_speed_var = tk.DoubleVar(value=40.0)
        # Drone
        self.drone_payload_var = tk.DoubleVar(value=5.0); self.drone_cost_var = tk.DoubleVar(value=1.0)
        self.drone_speed_var = tk.DoubleVar(value=60.0); self.drone_range_var = tk.DoubleVar(value=10.0)
        # Objective
        self.cost_weight_var = tk.DoubleVar(value=0.6); self.time_weight_var = tk.DoubleVar(value=0.4)
        self.unmet_demand_penalty_var = tk.DoubleVar(value=10000.0)
        # GA
        self.ga_pop_size_var = tk.IntVar(value=50); self.ga_gens_var = tk.IntVar(value=100)
        self.ga_mut_rate_var = tk.DoubleVar(value=0.15); self.ga_cross_rate_var = tk.DoubleVar(value=0.8)
        self.ga_elitism_var = tk.IntVar(value=2); self.ga_tourn_size_var = tk.IntVar(value=5)
        # SA
        self.sa_init_temp_var = tk.DoubleVar(value=1000.0); self.sa_cool_rate_var = tk.DoubleVar(value=0.99)
        self.sa_iters_var = tk.IntVar(value=10000); self.sa_min_temp_var = tk.DoubleVar(value=0.01)
        # PSO
        self.pso_swarm_size_var = tk.IntVar(value=30); self.pso_max_iters_var = tk.IntVar(value=100)
        self.pso_inertia_var = tk.DoubleVar(value=0.7); self.pso_cognitive_var = tk.DoubleVar(value=1.5)
        self.pso_social_var = tk.DoubleVar(value=1.5)
        logger.debug("Tk variables initialized.")

    def _ensure_config_exists(self):
        """Creates a default config file if it doesn't exist."""
        if not os.path.exists(self.config_file):
            logger.info(f"Config file not found at {self.config_file}. Creating default.")
            default_config = configparser.ConfigParser(interpolation=None)
            default_config['DATA_GENERATION'] = {
                'num_logistics_centers': str(self.num_logistics_centers_var.get()), 'num_sales_outlets': str(self.num_sales_outlets_var.get()),
                'num_customers': str(self.num_customers_var.get()), 'use_solomon_like': str(self.use_solomon_like_var.get()),
                'center_latitude': str(self.center_latitude_var.get()), 'center_longitude': str(self.center_longitude_var.get()),
                'radius_km': str(self.radius_km_var.get()), 'min_demand': str(self.min_demand_var.get()), 'max_demand': str(self.max_demand_var.get())
            }
            default_config['VEHICLE'] = {
                'max_payload_kg': str(self.vehicle_payload_var.get()), 'cost_per_km': str(self.vehicle_cost_var.get()), 'speed_kmh': str(self.vehicle_speed_var.get())
            }
            default_config['DRONE'] = {
                'max_payload_kg': str(self.drone_payload_var.get()), 'cost_per_km': str(self.drone_cost_var.get()),
                'speed_kmh': str(self.drone_speed_var.get()), 'max_flight_distance_km': str(self.drone_range_var.get())
            }
            default_config['OBJECTIVE'] = {
                'cost_weight': str(self.cost_weight_var.get()), 'time_weight': str(self.time_weight_var.get()), 'unmet_demand_penalty': str(self.unmet_demand_penalty_var.get())
            }
            default_config['ALGORITHM_PARAMS'] = { # Save params as JSON strings
                'genetic_algorithm': json.dumps({'population_size': self.ga_pop_size_var.get(), 'num_generations': self.ga_gens_var.get(), 'mutation_rate': self.ga_mut_rate_var.get(), 'crossover_rate': self.ga_cross_rate_var.get(), 'elite_count': self.ga_elitism_var.get(), 'tournament_size': self.ga_tourn_size_var.get()}),
                'simulated_annealing': json.dumps({'initial_temperature': self.sa_init_temp_var.get(), 'cooling_rate': self.sa_cool_rate_var.get(), 'max_iterations': self.sa_iters_var.get(), 'min_temperature': self.sa_min_temp_var.get()}),
                'pso_optimizer': json.dumps({'num_particles': self.pso_swarm_size_var.get(), 'max_iterations': self.pso_max_iters_var.get(), 'inertia_weight': self.pso_inertia_var.get(), 'cognitive_weight': self.pso_cognitive_var.get(), 'social_weight': self.pso_social_var.get()}),
                'greedy_heuristic': json.dumps({}) # Empty params for greedy
            }
            try:
                with open(self.config_file, 'w', encoding='utf-8') as configfile:
                    default_config.write(configfile)
                logger.info(f"Default config file created at {self.config_file}")
            except Exception as e:
                logger.error(f"Could not create default config file '{self.config_file}': {e}", exc_info=True)
                messagebox.showerror("Config Error", f"Could not create default config file:\n{self.config_file}\nError: {e}", parent=self, icon='error')

    def load_configuration(self, filename: Optional[str] = None):
        """Loads parameters from the config file and populates GUI vars."""
        load_path = filename if filename else self.config_file
        logger.info(f"Attempting to load configuration from: {load_path}")
        if not os.path.exists(load_path):
            logger.warning(f"Config file not found: {load_path}. Using current/default values.")
            return

        try:
            temp_config = configparser.ConfigParser(interpolation=None)
            read_files = temp_config.read(load_path, encoding='utf-8')
            if not read_files:
                 logger.warning(f"Config file {load_path} was empty or could not be parsed.")
                 return

            # Helper to get value safely and set tk var
            def get_and_set(section, key, var, var_type=float):
                try:
                    value_str = temp_config.get(section, key)
                    current_val = var.get() # Get current value for fallback
                    if var_type == int: var.set(int(float(value_str))) # Read as float first for robustness
                    elif var_type == float: var.set(float(value_str))
                    elif var_type == bool: var.set(value_str.lower() in ['true', 'yes', '1', 'on'])
                    else: var.set(value_str) # String type
                except (configparser.NoSectionError, configparser.NoOptionError):
                     pass # Keep existing value if key not found
                except (ValueError, tk.TclError) as e:
                     logger.warning(f"Config Warning: Invalid value type for [{section}]{key}: '{value_str}'. Error: {e}. Keeping current: {current_val}")
                     var.set(current_val) # Reset to previous value on error
                except Exception as e:
                     logger.error(f"Unexpected error reading config [{section}]{key}: {e}", exc_info=True)

            # Load values using helper
            get_and_set('DATA_GENERATION', 'num_logistics_centers', self.num_logistics_centers_var, int)
            get_and_set('DATA_GENERATION', 'num_sales_outlets', self.num_sales_outlets_var, int)
            get_and_set('DATA_GENERATION', 'num_customers', self.num_customers_var, int)
            get_and_set('DATA_GENERATION', 'use_solomon_like', self.use_solomon_like_var, bool)
            get_and_set('DATA_GENERATION', 'center_latitude', self.center_latitude_var, float)
            get_and_set('DATA_GENERATION', 'center_longitude', self.center_longitude_var, float)
            get_and_set('DATA_GENERATION', 'radius_km', self.radius_km_var, float)
            get_and_set('DATA_GENERATION', 'min_demand', self.min_demand_var, float)
            get_and_set('DATA_GENERATION', 'max_demand', self.max_demand_var, float)

            get_and_set('VEHICLE', 'max_payload_kg', self.vehicle_payload_var, float)
            get_and_set('VEHICLE', 'cost_per_km', self.vehicle_cost_var, float)
            get_and_set('VEHICLE', 'speed_kmh', self.vehicle_speed_var, float)

            get_and_set('DRONE', 'max_payload_kg', self.drone_payload_var, float)
            get_and_set('DRONE', 'cost_per_km', self.drone_cost_var, float)
            get_and_set('DRONE', 'speed_kmh', self.drone_speed_var, float)
            get_and_set('DRONE', 'max_flight_distance_km', self.drone_range_var, float)

            get_and_set('OBJECTIVE', 'cost_weight', self.cost_weight_var, float)
            get_and_set('OBJECTIVE', 'time_weight', self.time_weight_var, float)
            get_and_set('OBJECTIVE', 'unmet_demand_penalty', self.unmet_demand_penalty_var, float)

            # Load Algorithm Params (from JSON strings) safely
            if temp_config.has_section('ALGORITHM_PARAMS'):
                # GA
                if temp_config.has_option('ALGORITHM_PARAMS', 'genetic_algorithm'):
                    try:
                        ga_params = json.loads(temp_config.get('ALGORITHM_PARAMS', 'genetic_algorithm'))
                        if isinstance(ga_params, dict):
                            self.ga_pop_size_var.set(int(ga_params.get('population_size', self.ga_pop_size_var.get())))
                            self.ga_gens_var.set(int(ga_params.get('num_generations', self.ga_gens_var.get()))) # Updated key
                            self.ga_mut_rate_var.set(float(ga_params.get('mutation_rate', self.ga_mut_rate_var.get())))
                            self.ga_cross_rate_var.set(float(ga_params.get('crossover_rate', self.ga_cross_rate_var.get())))
                            self.ga_elitism_var.set(int(ga_params.get('elite_count', self.ga_elitism_var.get()))) # Updated key
                            self.ga_tourn_size_var.set(int(ga_params.get('tournament_size', self.ga_tourn_size_var.get())))
                        else: raise TypeError("GA params JSON is not a dictionary")
                    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e: logger.warning(f"Config Warning: Could not load/parse GA params: {e}")
                # SA
                if temp_config.has_option('ALGORITHM_PARAMS', 'simulated_annealing'):
                    try:
                        sa_params = json.loads(temp_config.get('ALGORITHM_PARAMS', 'simulated_annealing'))
                        if isinstance(sa_params, dict):
                            self.sa_init_temp_var.set(float(sa_params.get('initial_temperature', self.sa_init_temp_var.get())))
                            self.sa_cool_rate_var.set(float(sa_params.get('cooling_rate', self.sa_cool_rate_var.get())))
                            self.sa_iters_var.set(int(sa_params.get('max_iterations', self.sa_iters_var.get()))) # Updated key
                            self.sa_min_temp_var.set(float(sa_params.get('min_temperature', self.sa_min_temp_var.get()))) # Updated key
                        else: raise TypeError("SA params JSON is not a dictionary")
                    except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e: logger.warning(f"Config Warning: Could not load/parse SA params: {e}")
                # PSO
                if temp_config.has_option('ALGORITHM_PARAMS', 'pso_optimizer'):
                     try:
                        pso_params = json.loads(temp_config.get('ALGORITHM_PARAMS', 'pso_optimizer'))
                        if isinstance(pso_params, dict):
                            self.pso_swarm_size_var.set(int(pso_params.get('num_particles', self.pso_swarm_size_var.get()))) # Updated key
                            self.pso_max_iters_var.set(int(pso_params.get('max_iterations', self.pso_max_iters_var.get())))
                            self.pso_inertia_var.set(float(pso_params.get('inertia_weight', self.pso_inertia_var.get()))) # Updated key
                            self.pso_cognitive_var.set(float(pso_params.get('cognitive_weight', self.pso_cognitive_var.get()))) # Updated key
                            self.pso_social_var.set(float(pso_params.get('social_weight', self.pso_social_var.get()))) # Updated key
                        else: raise TypeError("PSO params JSON is not a dictionary")
                     except (json.JSONDecodeError, ValueError, TypeError, KeyError) as e: logger.warning(f"Config Warning: Could not load/parse PSO params: {e}")

            self.config = temp_config
            logger.info(f"Configuration loaded successfully from {load_path}.")

        except configparser.ParsingError as e:
            messagebox.showerror("Config Parse Error", f"Error parsing config file:\n{load_path}\nError: {e}\nUsing current values.", parent=self, icon='error')
        except Exception as e:
            logger.error(f"Unexpected error reading config file '{load_path}': {e}", exc_info=True)
            messagebox.showerror("Config Load Error", f"Unexpected error reading config file:\n{load_path}\nError: {e}\nUsing current values.", parent=self, icon='error')

    def save_configuration(self, filename: Optional[str] = None):
        """Saves current GUI parameters to the configuration file."""
        save_path = filename if filename else self.config_file
        logger.info(f"Attempting to save configuration to: {save_path}")
        temp_config = configparser.ConfigParser(interpolation=None)

        # Helper to safely get tk var value and set in config
        def set_config_value(section, key, tk_var):
            try:
                if not temp_config.has_section(section):
                    temp_config.add_section(section)
                temp_config.set(section, key, str(tk_var.get()))
            except tk.TclError as e: logger.warning(f"Error getting value for config [{section}] {key}: {e}. Skipping.")
            except Exception as e: logger.warning(f"Unexpected error setting config [{section}] {key}: {e}. Skipping.")

        # Save sections
        set_config_value('DATA_GENERATION', 'num_logistics_centers', self.num_logistics_centers_var)
        set_config_value('DATA_GENERATION', 'num_sales_outlets', self.num_sales_outlets_var)
        set_config_value('DATA_GENERATION', 'num_customers', self.num_customers_var)
        set_config_value('DATA_GENERATION', 'use_solomon_like', self.use_solomon_like_var)
        set_config_value('DATA_GENERATION', 'center_latitude', self.center_latitude_var)
        set_config_value('DATA_GENERATION', 'center_longitude', self.center_longitude_var)
        set_config_value('DATA_GENERATION', 'radius_km', self.radius_km_var)
        set_config_value('DATA_GENERATION', 'min_demand', self.min_demand_var)
        set_config_value('DATA_GENERATION', 'max_demand', self.max_demand_var)
        set_config_value('VEHICLE', 'max_payload_kg', self.vehicle_payload_var); set_config_value('VEHICLE', 'cost_per_km', self.vehicle_cost_var); set_config_value('VEHICLE', 'speed_kmh', self.vehicle_speed_var)
        set_config_value('DRONE', 'max_payload_kg', self.drone_payload_var); set_config_value('DRONE', 'cost_per_km', self.drone_cost_var); set_config_value('DRONE', 'speed_kmh', self.drone_speed_var); set_config_value('DRONE', 'max_flight_distance_km', self.drone_range_var)
        set_config_value('OBJECTIVE', 'cost_weight', self.cost_weight_var); set_config_value('OBJECTIVE', 'time_weight', self.time_weight_var); set_config_value('OBJECTIVE', 'unmet_demand_penalty', self.unmet_demand_penalty_var)

        # Save Algorithm Params as JSON strings
        if not temp_config.has_section('ALGORITHM_PARAMS'): temp_config.add_section('ALGORITHM_PARAMS')
        try: # GA
             ga_params = {'population_size': self.ga_pop_size_var.get(), 'num_generations': self.ga_gens_var.get(), 'mutation_rate': self.ga_mut_rate_var.get(), 'crossover_rate': self.ga_cross_rate_var.get(), 'elite_count': self.ga_elitism_var.get(), 'tournament_size': self.ga_tourn_size_var.get()}
             temp_config.set('ALGORITHM_PARAMS', 'genetic_algorithm', json.dumps(ga_params))
        except tk.TclError as e: logger.warning(f"Error getting GA params for save: {e}")
        try: # SA
             sa_params = {'initial_temperature': self.sa_init_temp_var.get(), 'cooling_rate': self.sa_cool_rate_var.get(), 'max_iterations': self.sa_iters_var.get(), 'min_temperature': self.sa_min_temp_var.get()}
             temp_config.set('ALGORITHM_PARAMS', 'simulated_annealing', json.dumps(sa_params))
        except tk.TclError as e: logger.warning(f"Error getting SA params for save: {e}")
        try: # PSO
             pso_params = {'num_particles': self.pso_swarm_size_var.get(), 'max_iterations': self.pso_max_iters_var.get(), 'inertia_weight': self.pso_inertia_var.get(), 'cognitive_weight': self.pso_cognitive_var.get(), 'social_weight': self.pso_social_var.get()}
             temp_config.set('ALGORITHM_PARAMS', 'pso_optimizer', json.dumps(pso_params))
        except tk.TclError as e: logger.warning(f"Error getting PSO params for save: {e}")
        temp_config.set('ALGORITHM_PARAMS', 'greedy_heuristic', json.dumps({})) # Save empty dict for greedy


        # Write to file
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as configfile:
                temp_config.write(configfile)
            self.config = temp_config # Update main config object
            logger.info(f"Configuration saved successfully to {save_path}")
            messagebox.showinfo("Success", f"Parameters saved to:\n{save_path}", parent=self)
        except Exception as e:
            logger.error(f"Error saving config file '{save_path}': {e}", exc_info=True)
            messagebox.showerror("Config Save Error", f"Failed to save configuration.\n{e}", parent=self, icon='error')

    def _load_config_dialog(self):
        """Opens dialog to load config file."""
        filepath = filedialog.askopenfilename(
            parent=self, initialdir=CONFIG_DIR, title="Select Configuration File",
            filetypes=(("INI files", "*.ini"), ("All files", "*.*"))
        )
        if filepath:
            self.load_configuration(filepath)
            # No need for success message here, load_configuration logs/shows errors
        else: logger.info("Config load cancelled.")

    def _save_config_dialog(self):
        """Opens dialog to save config file."""
        filepath = filedialog.asksaveasfilename(
            parent=self, initialdir=CONFIG_DIR, title="Save Configuration File As",
            initialfile=os.path.basename(self.config_file), defaultextension=".ini",
            filetypes=(("INI files", "*.ini"), ("All files", "*.*"))
        )
        if filepath:
            self.save_configuration(filepath) # Shows its own success/error message
        else: logger.info("Config save cancelled.")

    # --- Data Generation & Optimization Execution ---

    def _get_all_parameters(self) -> Optional[Dict[str, Any]]:
        """
        Collects parameters from GUI, performs validation, and returns structured dicts.
        Returns None if validation fails.
        """
        logger.debug("Collecting and validating parameters from GUI...")
        params = {'validation_errors': []}
        param_values = {} # Store successfully parsed values

        # Helper for safe get with type and range check
        def get_validated_var(var, name, var_type=float, min_val=None, max_val=None, allow_zero=True):
            try:
                value = var_type(var.get())
                numeric_type = isinstance(value, (int, float))

                if numeric_type and not allow_zero and abs(value) < 1e-9:
                     params['validation_errors'].append(f"{name} must not be zero.")
                if numeric_type and min_val is not None and value < min_val:
                     params['validation_errors'].append(f"{name} ({value}) must be >= {min_val}.")
                if numeric_type and max_val is not None and value > max_val:
                     params['validation_errors'].append(f"{name} ({value}) must be <= {max_val}.")
                if var_type == float and not math.isfinite(value):
                    params['validation_errors'].append(f"{name} must be a finite number.")

                param_values[name] = value # Store successfully parsed value
                return value
            except (tk.TclError, ValueError):
                params['validation_errors'].append(f"Invalid input for {name} (expected {var_type.__name__}).")
                return None # Indicate failure

        # --- Problem Data Params ---
        data_p = {
            'num_logistics_centers': get_validated_var(self.num_logistics_centers_var, "Num Logistics Centers", int, min_val=1),
            'num_sales_outlets': get_validated_var(self.num_sales_outlets_var, "Num Sales Outlets", int, min_val=0),
            'num_customers': get_validated_var(self.num_customers_var, "Num Customers", int, min_val=0),
            'center_latitude': get_validated_var(self.center_latitude_var, "Center Latitude", float, min_val=-90, max_val=90),
            'center_longitude': get_validated_var(self.center_longitude_var, "Center Longitude", float, min_val=-180, max_val=180),
            'radius_km': get_validated_var(self.radius_km_var, "Radius (km)", float, min_val=0, allow_zero=False),
            'use_solomon_like_distribution': self.use_solomon_like_var.get(), # Boolean
            'min_demand': get_validated_var(self.min_demand_var, "Min Demand", float, min_val=0),
            'max_demand': get_validated_var(self.max_demand_var, "Max Demand", float, min_val=0)
        }
        if param_values.get("Min Demand") is not None and param_values.get("Max Demand") is not None and \
           param_values["Max Demand"] < param_values["Min Demand"]:
             params['validation_errors'].append("Max Demand cannot be less than Min Demand.")
        params['problem_data_params'] = data_p

        # --- Vehicle Params ---
        vehicle_p = {
            'payload': get_validated_var(self.vehicle_payload_var, "Vehicle Payload", float, min_val=0),
            'cost_per_km': get_validated_var(self.vehicle_cost_var, "Vehicle Cost", float, min_val=0),
            'speed_kmph': get_validated_var(self.vehicle_speed_var, "Vehicle Speed", float, min_val=0, allow_zero=False) # Changed key to kmph
        }
        params['vehicle_params'] = vehicle_p

        # --- Drone Params ---
        drone_p = {
            'payload': get_validated_var(self.drone_payload_var, "Drone Payload", float, min_val=0),
            'cost_per_km': get_validated_var(self.drone_cost_var, "Drone Cost", float, min_val=0),
            'speed_kmph': get_validated_var(self.drone_speed_var, "Drone Speed", float, min_val=0, allow_zero=False), # Changed key to kmph
            'max_flight_distance_km': get_validated_var(self.drone_range_var, "Drone Range", float, min_val=0) # Changed key
        }
        params['drone_params'] = drone_p

        # --- Objective Weights ---
        obj_w = {
            'cost_weight': get_validated_var(self.cost_weight_var, "Cost Weight", float, min_val=0, max_val=1),
            'time_weight': get_validated_var(self.time_weight_var, "Time Weight", float, min_val=0, max_val=1)
        }
        params['objective_weights'] = obj_w

        # --- Optimization Params (General + Algorithm Specific nested) ---
        # Initialize the main dictionary for optimizer
        opt_params = {
            'unmet_demand_penalty': get_validated_var(self.unmet_demand_penalty_var, "Unmet Penalty", float, min_val=0),
            'output_dir': DEFAULT_OUTPUT_DIR # Pass base output dir, optimizer will create subdirs
        }
        # This will be populated with algo-specific dicts below

        # --- Selected Algorithms ---
        selected_keys = [key for key, var in self.selected_algorithms_vars.items() if key in self.loaded_algorithms and var.get()]
        params['selected_algorithm_keys'] = selected_keys
        if not selected_keys:
            params['validation_errors'].append("Please select at least one available algorithm to run.")

        # --- Algorithm Specific Params (Collected and nested into opt_params) ---
        # GA
        if 'genetic_algorithm' in selected_keys:
            ga_p = {}
            ga_p['population_size'] = get_validated_var(self.ga_pop_size_var, "GA Pop Size", int, min_val=2)
            ga_p['num_generations'] = get_validated_var(self.ga_gens_var, "GA Generations", int, min_val=0) # Key name matches GA code
            ga_p['mutation_rate'] = get_validated_var(self.ga_mut_rate_var, "GA Mut Rate", float, min_val=0, max_val=1)
            ga_p['crossover_rate'] = get_validated_var(self.ga_cross_rate_var, "GA Cross Rate", float, min_val=0, max_val=1)
            ga_p['elite_count'] = get_validated_var(self.ga_elitism_var, "GA Elitism", int, min_val=0) # Key name matches GA code
            ga_p['tournament_size'] = get_validated_var(self.ga_tourn_size_var, "GA Tourn Size", int, min_val=1)
            # Inter-param checks for GA
            pop_s = param_values.get("GA Pop Size")
            elite_c = param_values.get("GA Elitism")
            tourn_s = param_values.get("GA Tourn Size")
            if pop_s is not None:
                if elite_c is not None and elite_c >= pop_s: params['validation_errors'].append("GA Elite Count must be less than Population Size.")
                if tourn_s is not None and tourn_s > pop_s: params['validation_errors'].append("GA Tournament Size cannot exceed Population Size.")
            # Add GA params to main opt_params dict if all GA params are valid
            if all(v is not None for v in ga_p.values()):
                opt_params['genetic_algorithm_params'] = ga_p
            else: params['validation_errors'].append("Invalid GA parameter(s) entered.")

        # SA
        if 'simulated_annealing' in selected_keys:
            sa_p = {}
            sa_p['initial_temperature'] = get_validated_var(self.sa_init_temp_var, "SA Init Temp", float, min_val=0, allow_zero=False) # Key name matches SA code
            sa_p['cooling_rate'] = get_validated_var(self.sa_cool_rate_var, "SA Cool Rate", float, min_val=0, max_val=1, allow_zero=False)
            sa_p['max_iterations'] = get_validated_var(self.sa_iters_var, "SA Iterations", int, min_val=0) # Key name matches SA code
            sa_p['min_temperature'] = get_validated_var(self.sa_min_temp_var, "SA Min Temp", float, min_val=0) # Key name matches SA code
            init_t = param_values.get("SA Init Temp")
            min_t = param_values.get("SA Min Temp")
            if init_t is not None and min_t is not None and min_t >= init_t: params['validation_errors'].append("SA Min Temperature must be less than Initial Temperature.")
            if all(v is not None for v in sa_p.values()): opt_params['simulated_annealing_params'] = sa_p
            else: params['validation_errors'].append("Invalid SA parameter(s) entered.")

        # PSO
        if 'pso_optimizer' in selected_keys:
            pso_p = {}
            pso_p['num_particles'] = get_validated_var(self.pso_swarm_size_var, "PSO Swarm Size", int, min_val=1) # Key name matches PSO code
            pso_p['max_iterations'] = get_validated_var(self.pso_max_iters_var, "PSO Iterations", int, min_val=0)
            pso_p['inertia_weight'] = get_validated_var(self.pso_inertia_var, "PSO Inertia", float, min_val=0) # Key name matches PSO code
            pso_p['cognitive_weight'] = get_validated_var(self.pso_cognitive_var, "PSO Cognitive", float, min_val=0) # Key name matches PSO code
            pso_p['social_weight'] = get_validated_var(self.pso_social_var, "PSO Social", float, min_val=0) # Key name matches PSO code
            if all(v is not None for v in pso_p.values()): opt_params['pso_optimizer_params'] = pso_p
            else: params['validation_errors'].append("Invalid PSO parameter(s) entered.")

        # Greedy
        if 'greedy_heuristic' in selected_keys:
            opt_params['greedy_heuristic_params'] = {} # No specific params for greedy

        # Assign the fully populated opt_params dict
        params['optimization_params'] = opt_params


        # --- Final Validation Check ---
        if params['validation_errors']:
            error_message = "Parameter validation failed:\n\n" + "\n".join(f"- {e}" for e in params['validation_errors'])
            logger.warning(f"Parameter validation failed:\n{params['validation_errors']}")
            messagebox.showwarning("Input Error", error_message, parent=self, icon='warning')
            return None
        else:
            logger.debug("Parameters collected and validated successfully.")
            # Remove the temporary validation errors list before returning
            del params['validation_errors']
            return params


    def _generate_data(self):
        """Triggers data generation based on GUI parameters."""
        logger.info("Starting data generation...")
        if not _DATA_GENERATOR_AVAILABLE:
            messagebox.showerror("Error", "Data generation functions are not available.", parent=self, icon='error')
            return

        self._disable_gui_elements(running_data_gen=True)
        self.status_var.set("Generating data...")
        self.update_idletasks() # Ensure status update is visible

        # Get relevant parameters (only data generation section needed)
        all_params = self._get_all_parameters()
        if all_params is None: # Validation failed
             self._enable_gui_elements()
             self.status_var.set("Ready (Parameter Error)")
             return

        data_params = all_params.get('problem_data_params', {})

        try:
            logger.info("Calling data_generator.generate_locations...")
            num_customers = data_params.get('num_customers', 0) # Get target num_customers

            generated_locations = generate_locations(
                num_logistics_centers=data_params.get('num_logistics_centers', 1),
                num_sales_outlets=data_params.get('num_sales_outlets', 0),
                num_customers=num_customers,
                center_latitude=data_params.get('center_latitude', 0.0),
                center_longitude=data_params.get('center_longitude', 0.0),
                radius_km=data_params.get('radius_km', 10.0),
                use_solomon_like_distribution=data_params.get('use_solomon_like_distribution', False)
            )
            if not generated_locations or not isinstance(generated_locations, dict) or not generated_locations.get("logistics_centers"):
                 raise RuntimeError("Location generation failed or returned invalid structure.")
            logger.info("Location data generated.")

            # Use actual number of customers generated for demand generation
            actual_num_customers = len(generated_locations.get('customers', []))
            if actual_num_customers != num_customers:
                logger.warning(f"Generated {actual_num_customers} customer locations (requested {num_customers}). Generating demands for {actual_num_customers}.")
                num_customers = actual_num_customers

            logger.info("Calling data_generator.generate_demand...")
            if num_customers > 0:
                generated_demands = generate_demand(
                    num_customers=num_customers,
                    min_demand=data_params.get('min_demand', 0.0),
                    max_demand=data_params.get('max_demand', 0.0)
                )
                if generated_demands is None: # generate_demand should return [] if num_customers=0
                     raise RuntimeError("Demand generation failed (returned None).")
            else:
                 generated_demands = [] # No customers, no demands
            logger.info("Demand data generated.")

            # Store Generated Data
            self.current_problem_data = {
                'locations': generated_locations,
                'demands': generated_demands
            }
            logger.info("Problem data generated and stored.")

            # Reset Previous Results
            self.current_optimization_results = None
            self._clear_results_display()

            # Update GUI
            self.status_var.set("Data Generated Successfully. Ready to Run Optimization.")
            messagebox.showinfo("Data Generation", "Data generation complete.", parent=self)

            # Display generated points on map (if map generator works)
            self._display_generated_data_map()

        except ValueError as ve:
            logger.error(f"Input Error during data generation: {ve}", exc_info=True)
            messagebox.showerror("Input Error", f"Please enter valid numeric values for data generation.\nDetails: {ve}", parent=self, icon='error')
            self.status_var.set("Ready (Data Gen Error)")
        except Exception as e:
            logger.error(f"An error occurred during data generation: {e}", exc_info=True)
            messagebox.showerror("Generation Error", f"An error occurred during data generation: {e}", parent=self, icon='error')
            self.status_var.set("Ready (Data Gen Error)")
        finally:
            self._enable_gui_elements()


    def _display_generated_data_map(self):
         """Displays the generated locations using the map generator."""
         logger.debug("Attempting to display generated data points map...")
         if not _MAP_GENERATOR_AVAILABLE:
             logger.warning("Map generator not available, cannot display initial points map.")
             return
         if not self.current_problem_data or not self.current_problem_data.get('locations'):
             logger.warning("No current problem data available, cannot display initial points map.")
             return

         try:
             # Define path for the initial map within the *default* output dir (not run-specific)
             map_output_path = os.path.join(DEFAULT_OUTPUT_DIR, "maps", "initial_locations_map.html")
             os.makedirs(os.path.dirname(map_output_path), exist_ok=True) # Ensure dir exists

             locations_data = self.current_problem_data.get('locations')
             demands_data = self.current_problem_data.get('demands')

             if not locations_data:
                 raise ValueError("Cannot generate map: Location data is missing.")

             # Call the updated map generator function signature
             generated_map_path = generate_folium_map(
                 problem_data=self.current_problem_data, # Pass full problem data
                 solution_structure=None, # No solution yet
                 vehicle_params={}, # Not relevant for points map
                 drone_params={}, # Not relevant for points map
                 output_path=map_output_path,
                 map_title="Initial Locations"
             )

             if generated_map_path and os.path.exists(generated_map_path):
                 logger.info(f"Generated data map saved to {generated_map_path}. Opening in browser...")
                 # Update GUI display first
                 self._update_map_display_widgets(clear=False, map_path=generated_map_path, selection_text="Initial Locations")
                 # Attempt to open
                 if open_map_in_browser(generated_map_path):
                      self.status_var.set(self.status_var.get() + " (Initial map opened)")
                 else:
                      messagebox.showwarning("Map Open Error", f"Could not automatically open map:\n{generated_map_path}", parent=self, icon='warning')
             else:
                 logger.warning("Failed to generate or find the initial locations map file.")
                 messagebox.showwarning("Map Error", "Failed to generate the initial locations map.", parent=self, icon='warning')
                 self._update_map_display_widgets(clear=True) # Clear display if map gen failed

         except Exception as e:
             logger.error(f"Error displaying generated data map: {e}", exc_info=True)
             messagebox.showwarning("Map Display Error", f"Error generating initial map:\n{e}", parent=self, icon='warning')


    def _run_optimization(self):
        """Triggers the optimization process in a separate thread."""
        logger.info("Initiating optimization run...")
        if not _CORE_OPTIMIZER_AVAILABLE:
            messagebox.showerror("Error", "Cannot run optimization: Core optimizer module not loaded.", parent=self, icon='error')
            return
        if not self.current_problem_data or not self.current_problem_data.get('locations'):
            messagebox.showwarning("Run Optimization", "Please generate or load problem data first.", parent=self, icon='warning')
            return

        # Get all parameters from GUI
        all_params = self._get_all_parameters()
        if all_params is None: return # Validation failed

        selected_keys = all_params.get('selected_algorithm_keys', [])
        if not selected_keys:
            messagebox.showwarning("Run Optimization", "Please select at least one available algorithm.", parent=self, icon='warning')
            return

        # Disable UI elements
        self._disable_gui_elements(running_optimization=True)
        algo_names_str = ", ".join([self.available_algorithms.get(k, k) for k in selected_keys])
        self.status_var.set(f"Running Optimization ({algo_names_str})...")
        self.update_idletasks()

        # Clear previous results display before starting new run
        self._clear_results_display()

        # Start Optimization Thread
        # Pass deep copies of mutable data (like problem_data, params) to the thread
        logger.info(f"Starting optimization thread for algorithms: {selected_keys}")
        self.optimization_thread = threading.Thread(
            target=self.run_optimization_thread,
            args=(copy.deepcopy(self.current_problem_data),
                  copy.deepcopy(all_params['vehicle_params']),
                  copy.deepcopy(all_params['drone_params']),
                  copy.deepcopy(all_params['optimization_params']), # Contains nested algo params
                  selected_keys,
                  copy.deepcopy(all_params['objective_weights'])),
            daemon=True # Allows main window to exit even if thread hangs
        )
        self.optimization_thread.start()

    def run_optimization_thread(self, problem_data, vehicle_params, drone_params,
                                optimization_params, selected_algorithm_keys, objective_weights):
        """Executes core.route_optimizer.run_optimization in the background thread."""
        logger.info("Optimization thread started.")
        results = None
        try:
            # Call the updated core optimizer function
            results = core_run_optimization(
                 problem_data=problem_data,
                 vehicle_params=vehicle_params,
                 drone_params=drone_params,
                 optimization_params=optimization_params,
                 selected_algorithm_keys=selected_algorithm_keys,
                 objective_weights=objective_weights
            )
            logger.info("Optimization thread: core_run_optimization finished.")
        except Exception as e:
            logger.error(f"An unexpected error occurred in the optimization thread: {e}", exc_info=True)
            # Create an error result structure if the call itself failed
            results = {
                'overall_status': 'Error',
                'error_message': f"Optimization thread failed unexpectedly: {e}",
                'results_by_algorithm': {},
                'run_timestamp': pytime.strftime("%Y-%m-%d_%H-%M-%S"), # Add timestamp even for error
                'output_directory': optimization_params.get('output_dir', DEFAULT_OUTPUT_DIR) # Base dir on error
            }
        finally:
            logger.info("Optimization thread finished. Scheduling results display update.")
            # Schedule GUI update back on the main thread using lambda to pass results
            self.after(0, lambda res=results: self._display_results(res))


    def _disable_gui_elements(self, running_data_gen=False, running_optimization=False):
         """Disables GUI elements during processing."""
         logger.debug("Disabling GUI elements.")
         widgets_to_disable = [
             self.generate_button, self.load_config_button, self.save_config_button,
             self.run_button, self.param_notebook, self.algo_params_notebook
             ]
         # Disable algorithm selection checkboxes
         if hasattr(self, 'algo_frame'):
              for child in self.algo_frame.winfo_children():
                   if isinstance(child, ttk.Checkbutton):
                        try: child.config(state=tk.DISABLED)
                        except tk.TclError: pass

         for widget in widgets_to_disable:
             if widget and hasattr(widget, 'state'): # Check if widget supports state config
                 try:
                      # Handle Notebooks specifically
                      if isinstance(widget, ttk.Notebook):
                           for i in range(len(widget.tabs())):
                                try: widget.tab(i, state="disabled")
                                except tk.TclError: pass # Ignore if tab doesn't exist
                      else:
                           widget.config(state=tk.DISABLED)
                 except tk.TclError: pass # Ignore errors if widget destroyed or doesn't support state

         # Also disable results notebook tabs while running
         if hasattr(self, 'results_notebook'):
             try:
                 for i in range(len(self.results_notebook.tabs())):
                     self.results_notebook.tab(i, state="disabled")
             except tk.TclError: pass


    def _enable_gui_elements(self):
         """Enables GUI elements after processing."""
         logger.debug("Enabling GUI elements.")
         widgets_to_enable = [
             self.load_config_button, self.save_config_button,
             self.param_notebook, self.algo_params_notebook
             ]
         # Enable generate button only if data generator is available
         if hasattr(self, 'generate_button'):
             gen_btn_state = tk.DISABLED if not _DATA_GENERATOR_AVAILABLE else tk.NORMAL
             try: self.generate_button.config(state=gen_btn_state)
             except tk.TclError: pass
             widgets_to_enable.append(self.generate_button)

         # Enable checkboxes in algo_frame if algorithms are loaded
         if hasattr(self, 'algo_frame'):
              for child in self.algo_frame.winfo_children():
                   if isinstance(child, ttk.Checkbutton):
                        chk_state = tk.NORMAL if self.loaded_algorithms else tk.DISABLED
                        try: child.config(state=chk_state)
                        except tk.TclError: pass

         for widget in widgets_to_enable:
            if widget and hasattr(widget, 'state'):
                 try:
                      if isinstance(widget, ttk.Notebook):
                           for i in range(len(widget.tabs())):
                                try: widget.tab(i, state="normal")
                                except tk.TclError: pass
                      else:
                           widget.config(state=tk.NORMAL)
                 except tk.TclError: pass

         # Enable results notebook tabs after completion
         if hasattr(self, 'results_notebook'):
              try:
                 for i in range(len(self.results_notebook.tabs())):
                     tab_text = self.results_notebook.tab(i, "text")
                     enable_tab = True
                     if tab_text == "Iteration Curves" and not _PLOT_GENERATOR_AVAILABLE: enable_tab = False
                     if tab_text == "Comparison" and not _PLOT_GENERATOR_AVAILABLE: enable_tab = False
                     if tab_text == "Route Map" and not _MAP_GENERATOR_AVAILABLE: enable_tab = False
                     if tab_text == "Detailed Report" and not _REPORT_GENERATOR_AVAILABLE: enable_tab = False

                     self.results_notebook.tab(i, state="normal" if enable_tab else "disabled")
              except tk.TclError: pass

         # Re-enable/disable run button based on current state
         run_btn_state = tk.DISABLED
         if _CORE_OPTIMIZER_AVAILABLE and self.loaded_algorithms and self.current_problem_data and self.current_problem_data.get('locations'):
             run_btn_state = tk.NORMAL
         if hasattr(self, 'run_button'):
              try: self.run_button.config(state=run_btn_state)
              except tk.TclError: pass


    # --- Results Display ---

    def _clear_results_display(self):
        """Clears the contents of the results display tabs."""
        logger.info("Clearing previous results display...")
        # Clear plots
        if _MATPLOTLIB_AVAILABLE:
            self._clear_matplotlib_canvas(self.history_canvas_agg, self.history_figure, self.history_ax)
            self._clear_matplotlib_canvas(self.comparison_canvas_agg, self.comparison_figure, self.comp_ax_cost, self.comp_ax_time)
            # Disable save buttons after clearing
            if hasattr(self, 'save_iter_button'): self.save_iter_button.config(state=tk.DISABLED)
            if hasattr(self, 'save_comp_button'): self.save_comp_button.config(state=tk.DISABLED)

        # Clear map selection/display
        self._update_map_display_widgets(clear=True)

        # Clear report selection/display
        self._update_report_display_widgets(clear=True)
        self._update_report_area("", clear=True)

        logger.debug("Results display cleared.")


    def _display_results(self, optimization_results: Optional[Dict[str, Any]]):
        """Receives results from optimizer thread and updates the GUI. Runs on the main thread."""
        logger.info("Processing optimization results for display...")
        self.current_optimization_results = optimization_results # Store results

        # --- Initial Check ---
        if optimization_results is None:
            logger.error("Received None for optimization results.")
            messagebox.showerror("Optimization Failed", "Optimization process returned no results. Check logs.", parent=self, icon='error')
            status = "Ready (Optimization Failed: No Results)"
            self._enable_gui_elements()
            self.status_var.set(status)
            return

        # --- Parse Overall Status ---
        overall_status = optimization_results.get('overall_status', 'Unknown')
        error_message = optimization_results.get('error_message')
        run_output_dir = optimization_results.get('output_directory', DEFAULT_OUTPUT_DIR) # Get run-specific output dir
        logger.info(f"Optimization Overall Status: {overall_status}. Output Dir: {run_output_dir}")

        final_status_msg = f"Ready (Status: {overall_status})"

        if error_message:
            logger.error(f"Optimization finished with error: {error_message}")
            messagebox.showerror("Optimization Status", f"Optimization finished with status: {overall_status}\nError: {error_message}", parent=self, icon='error')
            final_status_msg = f"Ready (Optimization Error: {overall_status})"
        elif overall_status not in ['Success', 'No Valid Results']:
             # Handle other non-success statuses
             logger.warning(f"Optimization finished with status: {overall_status}")
             messagebox.showwarning("Optimization Status", f"Optimization finished with status: {overall_status}", parent=self, icon='warning')
        else:
            # Success or No Valid Results
            if overall_status == 'Success': logger.info("Optimization completed successfully.")
            else: logger.warning("Optimization completed, but no valid results found.")
            final_status_msg = "Ready (Optimization Complete)" if overall_status == 'Success' else "Ready (No Valid Results)"


        # --- Update Result Displays ---
        results_by_algorithm = optimization_results.get('results_by_algorithm', {})

        # Update Iteration Curve Plot (Pass the full results dict)
        if _PLOT_GENERATOR_AVAILABLE:
            self._update_iteration_curve(results_by_algorithm)
            # Enable save button if plot has data
            if hasattr(self.history_ax, 'lines') and self.history_ax.lines:
                 if hasattr(self, 'save_iter_button'): self.save_iter_button.config(state=tk.NORMAL)


        # Update Results Comparison Plot (Pass the full results dict)
        if _PLOT_GENERATOR_AVAILABLE:
            self._update_comparison_plot(results_by_algorithm)
            # Enable save button if plot has data
            if (hasattr(self.comp_ax_cost, 'patches') and self.comp_ax_cost.patches) or \
               (hasattr(self.comp_ax_time, 'patches') and self.comp_ax_time.patches):
                 if hasattr(self, 'save_comp_button'): self.save_comp_button.config(state=tk.NORMAL)

        # Update Map Selection Dropdown
        self._update_map_display_widgets(results_by_algorithm=results_by_algorithm, clear=False)

        # Update Report Selection Dropdown
        self._update_report_display_widgets(results_by_algorithm=results_by_algorithm, clear=False)

        # --- Auto-Select/Open Best Result ---
        best_map_path = None
        best_report_path = None
        best_algo_display_name = None

        # Prefer feasible best, fallback to overall best
        best_key = optimization_results.get('fully_served_best_key') or optimization_results.get('best_algorithm_key')

        if best_key and best_key in results_by_algorithm:
             best_result_summary = results_by_algorithm[best_key]
             best_map_path = best_result_summary.get('map_path')
             best_report_path = best_result_summary.get('report_path') # Get report path as well
             best_algo_display_name = self.available_algorithms.get(best_key, best_key)

             # Auto-select best result in dropdowns if found
             if best_algo_display_name:
                 if hasattr(self, 'map_selection_menu') and best_algo_display_name in self.map_selection_menu['values']:
                      self.map_selection_var.set(best_algo_display_name)
                      self._on_map_selection() # Update map path display
                 if hasattr(self, 'report_selection_menu') and best_algo_display_name in self.report_selection_menu['values']:
                     self.report_selection_var.set(best_algo_display_name)
                     self._on_report_selection() # Trigger report display update

        # Attempt to open best map automatically
        if best_map_path and _MAP_GENERATOR_AVAILABLE and os.path.exists(best_map_path):
             logger.info(f"Automatically opening map for best solution ({best_key}): {best_map_path}")
             if open_map_in_browser(best_map_path):
                  final_status_msg += " (Best map opened)"
             else:
                  messagebox.showwarning("Map Open Error", f"Could not automatically open map:\n{best_map_path}", parent=self, icon='warning')


        # --- Finalize GUI State ---
        self._enable_gui_elements()
        self.status_var.set(final_status_msg)


    def _update_comparison_plot(self, results_by_algorithm):
        """Updates the Results Comparison plot tab."""
        if not _PLOT_GENERATOR_AVAILABLE or not _MATPLOTLIB_AVAILABLE:
             logger.warning("Comparison plot skipped (PlotGenerator/Matplotlib unavailable).")
             self._display_matplotlib_unavailable_message(target_frame=self.comp_canvas_widget_container)
             return
        if not hasattr(self, 'comparison_figure') or not self.comp_ax_cost or not self.comp_ax_time:
             logger.warning("Comparison plot axes not initialized.")
             return

        logger.debug("Updating comparison plot...")
        try:
            # Call the updated plot generator method
            self.plot_generator.plot_comparison_bars(
                results_by_algorithm, self.comp_ax_cost, self.comp_ax_time
            )
            # Redraw handled within plot_generator method now
        except Exception as e:
             logger.error(f"Error calling plot_comparison_bars: {e}", exc_info=True)
             # Try to display error on plot
             for ax in [self.comp_ax_cost, self.comp_ax_time]:
                  if ax:
                      try:
                          ax.clear()
                          ax.text(0.5, 0.5, f'Plot Error:\n{e}', ha='center', va='center', color='red', wrap=True)
                          ax.figure.canvas.draw_idle()
                      except Exception: pass


    def _update_iteration_curve(self, results_by_algorithm):
        """Updates the Iteration Curve plot tab."""
        if not _PLOT_GENERATOR_AVAILABLE or not _MATPLOTLIB_AVAILABLE:
            logger.warning("Iteration curve plot skipped (PlotGenerator/Matplotlib unavailable).")
            self._display_matplotlib_unavailable_message(target_frame=self.iter_canvas_widget_container)
            return
        if not hasattr(self, 'history_figure') or not self.history_ax:
            logger.warning("Iteration curve axes not initialized.")
            return

        logger.debug("Updating iteration curve plot...")
        try:
             # Call the updated plot generator method
             self.plot_generator.plot_iteration_curves(
                 results_by_algorithm, self.history_ax
             )
             # Redraw handled within plot_generator method now
        except Exception as e:
             logger.error(f"Error calling plot_iteration_curves: {e}", exc_info=True)
             # Try to display error on plot
             if self.history_ax:
                  try:
                      self.history_ax.clear()
                      self.history_ax.text(0.5, 0.5, f'Plot Error:\n{e}', ha='center', va='center', color='red', wrap=True)
                      self.history_ax.figure.canvas.draw_idle()
                  except Exception: pass


    def _update_map_display_widgets(self, results_by_algorithm=None, map_path=None, selection_text=None, clear=False):
        """Updates the map selection dropdown and path display."""
        logger.debug(f"Updating map display widgets (clear={clear}, selection={selection_text}, path={map_path})")
        if not hasattr(self, 'map_selection_menu') or not hasattr(self, 'open_map_button'): return

        # Store current selection to try and preserve it
        current_selection = self.map_selection_var.get() if not clear else "Select Result..."

        if clear:
            new_options = ["Select Result..."]
            new_selection = "Select Result..."
            new_path = ""
            info_text = "Generate data and run optimization to view maps."
            info_color = 'grey'
        else:
            new_options = ["Select Result..."]
            # Handle adding the 'Initial Locations' map if provided directly
            if selection_text == "Initial Locations" and map_path:
                new_options.append("Initial Locations")

            # Populate dropdown based on results that successfully generated a map
            valid_maps = {}
            if results_by_algorithm:
                for key, summary in results_by_algorithm.items():
                     if summary.get('status') == 'Success' and summary.get('map_path') and os.path.exists(summary['map_path']):
                         display_name = self.available_algorithms.get(key, key)
                         valid_maps[display_name] = summary['map_path']
                         if display_name not in new_options:
                             new_options.append(display_name)

            new_options.sort(key=lambda x: (x == "Select Result...", x == "Initial Locations", x)) # Sort order

            # Determine new selection and path
            if selection_text and map_path: # Direct update (e.g., initial map or best map)
                 new_selection = selection_text
                 new_path = map_path
            elif current_selection in valid_maps: # Try to keep previous valid selection
                 new_selection = current_selection
                 new_path = valid_maps[current_selection]
            elif "Initial Locations" in new_options and current_selection == "Initial Locations": # Keep initial if selected
                 new_selection = "Initial Locations"
                 new_path = os.path.join(DEFAULT_OUTPUT_DIR, "maps", "initial_locations_map.html") # Assume standard path
            else: # Reset selection
                 new_selection = "Select Result..."
                 new_path = ""

            # Determine info text and color
            if new_selection != "Select Result..." and new_path and os.path.exists(new_path):
                 info_text = f"Map available for: {new_selection}"
                 info_color = 'black'
            elif new_selection != "Select Result...":
                 info_text = f"Map file missing or unavailable for {new_selection}."
                 info_color = 'orange'
            else:
                 info_text = "Select a result to view its map." if valid_maps or "Initial Locations" in new_options else "No maps generated or available."
                 info_color = 'grey'


        # Apply updates to widgets
        try:
            self.map_selection_menu['values'] = new_options
            self.map_selection_var.set(new_selection)
            self.map_file_path_var.set(new_path)

            # Enable/disable widgets based on state
            combo_state = 'readonly' if len(new_options) > 1 else 'disabled'
            button_state = 'normal' if new_path and os.path.exists(new_path) and _MAP_GENERATOR_AVAILABLE else 'disabled'

            self.map_selection_menu.config(state=combo_state)
            self.open_map_button.config(state=button_state)
            if hasattr(self, 'map_info_label'): self.map_info_label.config(text=info_text, foreground=info_color)

        except tk.TclError as e:
            logger.warning(f"TclError updating map display widgets: {e}")


    def _on_map_selection(self, event=None):
        """Callback when a map is selected from the dropdown."""
        selected_display_name = self.map_selection_var.get()
        logger.debug(f"Map selection changed to: {selected_display_name}")

        new_path = ""
        info_text = "Select a result to view its map."
        info_color = 'grey'

        if selected_display_name == "Select Result...":
            pass # Keep path empty, info grey
        elif selected_display_name == "Initial Locations":
             new_path = os.path.join(DEFAULT_OUTPUT_DIR, "maps", "initial_locations_map.html")
        elif self.current_optimization_results and self.current_optimization_results.get('results_by_algorithm'):
            # Find the corresponding algorithm key and get its map path
            selected_key = None
            for key, name in self.available_algorithms.items():
                 if name == selected_display_name:
                     selected_key = key
                     break
            if selected_key:
                 results = self.current_optimization_results['results_by_algorithm']
                 new_path = results.get(selected_key, {}).get('map_path', "")

        # Update path variable and button state
        self.map_file_path_var.set(new_path)
        button_state = 'disabled'
        if new_path and os.path.exists(new_path) and _MAP_GENERATOR_AVAILABLE:
             button_state = 'normal'
             info_text = f"Map available for: {selected_display_name}"
             info_color = 'black'
        elif selected_display_name != "Select Result...":
             info_text = f"Map file missing or unavailable for {selected_display_name}."
             info_color = 'orange' # Use orange for missing file

        try:
            if hasattr(self, 'open_map_button'): self.open_map_button.config(state=button_state)
            if hasattr(self, 'map_info_label'): self.map_info_label.config(text=info_text, foreground=info_color)
        except tk.TclError as e:
            logger.warning(f"TclError updating map button/label state: {e}")


    def _open_selected_map(self):
        """Opens the map file specified in the map_file_path_var."""
        map_path = self.map_file_path_var.get()
        if not map_path:
             messagebox.showwarning("Open Map", "No map file selected or path is invalid.", parent=self, icon='warning')
             return
        if not _MAP_GENERATOR_AVAILABLE:
            messagebox.showerror("Open Map Error", "Map generation/opening library not available.", parent=self, icon='error')
            return
        if not os.path.exists(map_path):
             messagebox.showerror("Open Map Error", f"Map file not found:\n{map_path}", parent=self, icon='error')
             return

        logger.info(f"Opening selected map: {map_path}")
        if not open_map_in_browser(map_path):
             messagebox.showerror("Open Map Error", f"Could not open map file in browser.\nPlease check browser settings or open manually:\n{map_path}", parent=self, icon='error')


    def _update_report_display_widgets(self, results_by_algorithm=None, clear=False):
        """Updates the report selection dropdown."""
        logger.debug(f"Updating report display widgets (clear={clear})")
        if not hasattr(self, 'report_selection_menu') or not hasattr(self, 'save_report_button'): return

        current_selection = self.report_selection_var.get() if not clear else "Select Result..."

        if clear:
             new_options = ["Select Result..."]
             new_selection = "Select Result..."
        else:
            new_options = ["Select Result..."]
            valid_reports = {}
            if results_by_algorithm and _REPORT_GENERATOR_AVAILABLE:
                for key, summary in results_by_algorithm.items():
                     # Include results that ran (even with error, to show error msg) or have a report path
                     if summary.get('status') in ['Success', 'Failed'] and summary.get('report_path') and os.path.exists(summary['report_path']):
                          display_name = self.available_algorithms.get(key, key)
                          valid_reports[display_name] = summary['report_path']
                          if display_name not in new_options:
                              new_options.append(display_name)
                     elif summary.get('status') == 'Failed': # Include failed runs even without report file to show error
                          display_name = self.available_algorithms.get(key, key)
                          if display_name not in new_options:
                              new_options.append(f"{display_name} (Failed)") # Mark failed runs

            new_options.sort(key=lambda x: (x == "Select Result...", x)) # Sort

            # Determine selection
            if current_selection in valid_reports or current_selection in new_options: # Try to keep selection
                 new_selection = current_selection
            else:
                 new_selection = "Select Result..."


        # Apply updates
        try:
            self.report_selection_menu['values'] = new_options
            self.report_selection_var.set(new_selection)

            # Enable/disable widgets
            combo_state = 'readonly' if len(new_options) > 1 and _REPORT_GENERATOR_AVAILABLE else 'disabled'
            button_state = tk.DISABLED # Default save button to disabled
            if new_selection != "Select Result..." and not new_selection.endswith("(Failed)") and _REPORT_GENERATOR_AVAILABLE:
                 # Enable save only if a successful report is selected
                 button_state = tk.NORMAL

            self.report_selection_menu.config(state=combo_state)
            self.save_report_button.config(state=button_state)

        except tk.TclError as e:
            logger.warning(f"TclError updating report display widgets: {e}")


    def _on_report_selection(self, event=None):
         """Callback when a report is selected from the dropdown."""
         selected_display_name_raw = self.report_selection_var.get()
         logger.debug(f"Report selection changed to: {selected_display_name_raw}")

         report_content = "" # Default to empty
         button_state = tk.DISABLED

         if selected_display_name_raw == "Select Result...":
             report_content = "Select a result to view its report." if _REPORT_GENERATOR_AVAILABLE else "Report generation is unavailable."
         elif not self.current_optimization_results or not _REPORT_GENERATOR_AVAILABLE:
             report_content = "Report generation is unavailable or no results loaded."
         else:
             # Handle potentially marked "(Failed)" names
             selected_display_name = selected_display_name_raw.replace(" (Failed)", "").strip()

             selected_key = None
             for key, name in self.available_algorithms.items():
                 if name == selected_display_name:
                     selected_key = key
                     break

             if selected_key:
                 results_dict = self.current_optimization_results.get('results_by_algorithm', {})
                 algo_result_summary = results_dict.get(selected_key)

                 if algo_result_summary:
                     report_path = algo_result_summary.get('report_path')
                     run_error = algo_result_summary.get('run_error')
                     report_gen_error = algo_result_summary.get('report_generation_error')

                     if run_error:
                          report_content = f"Algorithm '{selected_display_name}' failed to run.\nError: {run_error}"
                     elif report_gen_error:
                          report_content = f"Report generation failed for '{selected_display_name}'.\nError: {report_gen_error}"
                     elif report_path and os.path.exists(report_path):
                         try:
                             with open(report_path, 'r', encoding='utf-8') as f:
                                 report_content = f.read()
                             button_state = tk.NORMAL # Enable save if file loaded
                         except Exception as e:
                             logger.error(f"Error reading report file '{report_path}': {e}", exc_info=True)
                             report_content = f"Error reading report file:\n{report_path}\n\n{e}"
                     else:
                         report_content = f"Report file not found or not generated for '{selected_display_name}'.\nExpected path: {report_path}"
                 else:
                      report_content = f"No result summary found for algorithm '{selected_key}'."
             else:
                  report_content = f"Could not map display name '{selected_display_name}' back to an algorithm key."


         self._update_report_area(report_content, clear=True)
         # Update save button state based on whether content was successfully loaded
         try:
             if hasattr(self, 'save_report_button'): self.save_report_button.config(state=button_state)
         except tk.TclError as e:
             logger.warning(f"TclError updating save report button state: {e}")


    def _update_report_area(self, text: str, clear: bool = False):
         """ Safely updates the report text area. """
         if not hasattr(self, 'report_display_area') or not self.report_display_area:
             logger.warning("Attempted to update report area, but widget not found.")
             return
         try:
             # Check if widget exists before configuring
             if self.report_display_area.winfo_exists():
                 self.report_display_area.config(state=tk.NORMAL)
                 if clear:
                     self.report_display_area.delete(1.0, tk.END)
                 self.report_display_area.insert(tk.END, text)
                 self.report_display_area.see(tk.END) # Scroll to bottom
                 self.report_display_area.config(state=tk.DISABLED)
             else:
                 logger.warning("Report display area widget no longer exists.")
         except tk.TclError as e:
             logger.warning(f"TclError updating report area: {e}")


    # --- Matplotlib Plotting Methods ---
    # (Keep these largely unchanged, use updated PlotGenerator)

    def _set_matplotlib_defaults(self):
        """Set default Matplotlib settings."""
        if not _MATPLOTLIB_AVAILABLE: return
        try:
            # Try common CJK-supporting fonts first
            font_prefs = ['Microsoft YaHei', 'SimHei', 'PingFang SC', 'WenQuanYi Micro Hei', 'DejaVu Sans', 'Arial Unicode MS', 'sans-serif']
            plt.rcParams['font.sans-serif'] = font_prefs
            plt.rcParams['axes.unicode_minus'] = False # Ensure minus sign displays correctly
            # Set default figure facecolor to match Tkinter background if possible
            try: default_bg = self.cget('bg')
            except: default_bg = 'SystemButtonFace' # Fallback
            plt.rcParams['figure.facecolor'] = default_bg
            plt.rcParams['axes.facecolor'] = default_bg

            logger.debug(f"Attempted Matplotlib font/style setting. Using: {plt.rcParams['font.sans-serif']}")
        except Exception as e: logger.warning(f"Warning: Failed to set preferred font/style for Matplotlib: {e}")

    def _display_matplotlib_unavailable_message(self, target_frame: Optional[tk.Widget]):
        """Displays message in plot tabs if Matplotlib is missing."""
        if not target_frame or not hasattr(target_frame, 'winfo_children'): return
        message = "Matplotlib not found.\nPlease install it:\npip install matplotlib\n\nPlotting disabled."
        try:
             # Clear existing content in the frame
             for widget in target_frame.winfo_children(): widget.destroy()
             # Add the message label
             lbl = ttk.Label(target_frame, text=message, justify=tk.CENTER, foreground="grey", style="Warning.TLabel")
             lbl.pack(expand=True, fill='both', padx=10, pady=10)
        except tk.TclError as e:
             logger.warning(f"TclError while displaying Matplotlib unavailable message: {e}")
        except Exception as e:
             logger.error(f"Error displaying Matplotlib unavailable message: {e}", exc_info=True)


    def _create_initial_plot_placeholders(self):
        """Creates empty Matplotlib figures in the tabs initially."""
        if not _MATPLOTLIB_AVAILABLE: return
        logger.debug("Creating initial Matplotlib placeholders...")
        placeholder_text = "Run Optimization to Generate Plots"
        # --- Iteration Placeholder ---
        try:
            if self.history_figure is None:
                 self.history_figure, self.history_ax = plt.subplots(figsize=(7, 6))
            self.history_ax.clear()
            self.history_ax.set_title("Optimization Cost History")
            self.history_ax.set_xlabel("Iteration / Generation (%)")
            self.history_ax.set_ylabel("Best Cost Found")
            self.history_ax.grid(True, which='both', linestyle='--', linewidth=0.5)
            self.history_ax.text(0.5, 0.5, placeholder_text, ha='center', va='center', transform=self.history_ax.transAxes, color='grey', fontsize='large')
            self.history_figure.tight_layout(pad=1.5)
            if hasattr(self, 'iter_canvas_widget_container'):
                self.history_canvas_agg = self._embed_matplotlib_figure(self.history_figure, self.iter_canvas_widget_container)
        except Exception as e: logger.error(f"Error creating iteration placeholder: {e}", exc_info=True)

        # --- Comparison Placeholder ---
        try:
            if self.comparison_figure is None:
                 self.comparison_figure, axes = plt.subplots(2, 1, figsize=(7, 7), sharex=False) # Don't share x
                 self.comp_ax_cost, self.comp_ax_time = axes
            self.comp_ax_cost.clear()
            self.comp_ax_time.clear()
            self.comparison_figure.suptitle("Final Results Comparison", y=0.99) # Use suptitle
            self.comp_ax_cost.set_ylabel("Weighted Cost")
            self.comp_ax_cost.grid(True, axis='y', linestyle='--', linewidth=0.5)
            self.comp_ax_cost.text(0.5, 0.5, placeholder_text, ha='center', va='center', transform=self.comp_ax_cost.transAxes, color='grey', fontsize='large')
            self.comp_ax_time.set_ylabel("Computation Time (s)")
            self.comp_ax_time.set_xlabel("Algorithm")
            self.comp_ax_time.grid(True, axis='y', linestyle='--', linewidth=0.5)
            self.comp_ax_time.text(0.5, 0.5, placeholder_text, ha='center', va='center', transform=self.comp_ax_time.transAxes, color='grey', fontsize='large')
            # self.comp_ax_cost.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) # Hide x-ticks on top plot
            self.comparison_figure.tight_layout(rect=[0, 0.05, 1, 0.95]) # Adjust layout
            if hasattr(self, 'comp_canvas_widget_container'):
                 self.comparison_canvas_agg = self._embed_matplotlib_figure(self.comparison_figure, self.comp_canvas_widget_container)
        except Exception as e: logger.error(f"Error creating comparison placeholder: {e}", exc_info=True)

    def _embed_matplotlib_figure(self, fig: plt.Figure, parent_container: tk.Widget) -> Optional[FigureCanvasTkAgg]:
        """Embeds a Matplotlib figure and toolbar into a Tkinter container using grid."""
        if not _MATPLOTLIB_AVAILABLE or not fig or not parent_container: return None
        logger.debug(f"Embedding Matplotlib figure into {parent_container}...")
        try:
            # Destroy existing widgets in the container first
            for widget in parent_container.winfo_children():
                widget.destroy()

            parent_container.grid_rowconfigure(0, weight=1) # Canvas row expands
            parent_container.grid_rowconfigure(1, weight=0) # Toolbar row doesn't expand
            parent_container.grid_columnconfigure(0, weight=1)

            canvas = FigureCanvasTkAgg(fig, master=parent_container)
            canvas_widget = canvas.get_tk_widget()
            canvas_widget.grid(row=0, column=0, sticky="nsew")

            # Add toolbar in a separate frame below the canvas
            toolbar_frame = ttk.Frame(parent_container)
            toolbar_frame.grid(row=1, column=0, sticky="ew")
            toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
            toolbar.update()

            canvas.draw_idle()
            logger.debug("Matplotlib figure embedded successfully.")
            return canvas
        except Exception as e:
            logger.error(f"ERROR embedding Matplotlib figure: {e}", exc_info=True)
            # Display error message in the container
            try:
                for widget in parent_container.winfo_children(): widget.destroy()
                lbl = ttk.Label(parent_container, text=f"Error embedding plot:\n{e}", foreground="red", justify=tk.CENTER)
                lbl.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
            except Exception as ie:
                 logger.error(f"Could not display embedding error message: {ie}")
            return None

    def _clear_matplotlib_canvas(self, canvas_agg, fig, *axes):
        """Clears axes and redraws the canvas with placeholder text if needed."""
        if not _MATPLOTLIB_AVAILABLE or not canvas_agg or not fig: return
        logger.debug(f"Clearing Matplotlib canvas for figure: {fig}")
        try:
            all_axes_to_clear = list(axes) if axes else fig.get_axes()
            if not all_axes_to_clear: return

            for ax in all_axes_to_clear:
                if ax: ax.clear()

            # Reset titles/labels and add placeholder text
            placeholder_text = "Run Optimization to Generate Plots"
            if fig == self.history_figure and self.history_ax:
                self.history_ax.set_title("Optimization Cost History"); self.history_ax.set_xlabel("Iteration / Generation (%)"); self.history_ax.set_ylabel("Best Cost Found")
                self.history_ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                self.history_ax.text(0.5, 0.5, placeholder_text, ha='center', va='center', transform=self.history_ax.transAxes, color='grey', fontsize='large')
                fig.tight_layout(pad=1.5)
            elif fig == self.comparison_figure and self.comp_ax_cost and self.comp_ax_time:
                 fig.suptitle("Final Results Comparison", y=0.99)
                 self.comp_ax_cost.set_ylabel("Weighted Cost"); self.comp_ax_cost.grid(True, axis='y', linestyle='--', linewidth=0.5)
                 self.comp_ax_cost.text(0.5, 0.5, placeholder_text, ha='center', va='center', transform=self.comp_ax_cost.transAxes, color='grey', fontsize='large')
                 self.comp_ax_time.set_ylabel("Computation Time (s)"); self.comp_ax_time.set_xlabel("Algorithm"); self.comp_ax_time.grid(True, axis='y', linestyle='--', linewidth=0.5)
                 self.comp_ax_time.text(0.5, 0.5, placeholder_text, ha='center', va='center', transform=self.comp_ax_time.transAxes, color='grey', fontsize='large')
                 # self.comp_ax_cost.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
                 fig.tight_layout(rect=[0, 0.05, 1, 0.95])

            canvas_agg.draw_idle()
        except Exception as e: logger.warning(f"Warning: Error during canvas clear/redraw: {e}", exc_info=True)

    # --- File Saving Methods ---

    def _save_plot(self, fig: Optional[plt.Figure], canvas_agg: Optional[FigureCanvasTkAgg], default_filename: str, title: str):
        """Helper function to save a Matplotlib figure."""
        if not _MATPLOTLIB_AVAILABLE or not fig or not canvas_agg:
            messagebox.showerror("Save Error", f"Cannot save plot: '{title}' plot unavailable (Matplotlib or figure missing).", parent=self, icon='error')
            return

        # Check if the plot actually contains data (simple check: count lines/bars/patches)
        has_data = False
        for ax in fig.get_axes():
            if ax.lines or ax.patches or ax.collections: # Check for lines, bars, or other collections (like scatter)
                # Further check if axes actually have plotted data points (not just empty lines/patches)
                if any(len(line.get_xdata()) > 0 for line in ax.lines) or \
                   any(isinstance(p, plt.Rectangle) and p.get_height() > 0 for p in ax.patches): # Check for bars with height > 0
                    has_data = True
                    break
        if not has_data:
            messagebox.showwarning("Save Plot", f"No data to save in the '{title}' plot.", parent=self, icon='warning')
            return

        # Determine initial directory (use the run-specific output dir if available)
        initial_dir = DEFAULT_OUTPUT_DIR # Fallback
        if self.current_optimization_results:
            run_output_dir = self.current_optimization_results.get('output_directory')
            if run_output_dir and os.path.isdir(run_output_dir):
                 initial_dir = os.path.join(run_output_dir, "charts")
                 os.makedirs(initial_dir, exist_ok=True) # Ensure charts subdir exists

        filepath = filedialog.asksaveasfilename(
            parent=self, initialdir=initial_dir, title=f"Save {title} Plot As",
            initialfile=default_filename, defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("PDF files", "*.pdf"), ("SVG files", "*.svg"), ("All files", "*.*")]
        )
        if filepath:
            try:
                os.makedirs(os.path.dirname(filepath), exist_ok=True) # Ensure dir exists
                fig.savefig(filepath, dpi=300, bbox_inches='tight')
                logger.info(f"{title} plot saved to: {filepath}")
                messagebox.showinfo("Success", f"{title} plot saved to:\n{filepath}", parent=self)
            except Exception as e:
                 logger.error(f"Failed to save {title} plot to '{filepath}': {e}", exc_info=True)
                 messagebox.showerror("Save Error", f"Failed to save {title} plot: {e}", parent=self, icon='error')
        else: logger.info(f"{title} plot save cancelled.")

    def save_iteration_plot(self):
        self._save_plot(self.history_figure, self.history_canvas_agg, "iteration_curves.png", "Iteration Curves")

    def save_comparison_plot(self):
        self._save_plot(self.comparison_figure, self.comparison_canvas_agg, "results_comparison.png", "Comparison")

    def save_report(self):
        """Saves the content of the report text area."""
        if not _REPORT_GENERATOR_AVAILABLE or not hasattr(self, 'report_display_area'):
             messagebox.showerror("Save Error", "Report functionality is disabled or text area not found.", parent=self, icon='error')
             return
        try:
            # Ensure widget exists before accessing
            if not self.report_display_area.winfo_exists():
                 messagebox.showerror("Save Error", "Report display area is not available.", parent=self, icon='error')
                 return

            self.report_display_area.config(state=tk.NORMAL)
            report_content = self.report_display_area.get("1.0", tk.END).strip()
            self.report_display_area.config(state=tk.DISABLED)

            if not report_content or report_content.startswith("Report generation module not available") or report_content.startswith("Select a result"):
                messagebox.showwarning("Save Report", "No valid report content available to save.", parent=self, icon='warning')
                return

            # Suggest a filename based on the selected report
            selected_report_name_raw = self.report_selection_var.get()
            selected_report_name = selected_report_name_raw.replace(" (Failed)", "").strip()
            default_filename = "delivery_report.txt"
            if selected_report_name != "Select Result...":
                 # Sanitize display name for filename
                 safe_algo_name = selected_report_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
                 default_filename = f"{safe_algo_name}_report.txt"

            # Determine initial directory (use the run-specific output dir if available)
            initial_dir = DEFAULT_OUTPUT_DIR # Fallback
            if self.current_optimization_results:
                run_output_dir = self.current_optimization_results.get('output_directory')
                if run_output_dir and os.path.isdir(run_output_dir):
                    initial_dir = os.path.join(run_output_dir, "reports")
                    os.makedirs(initial_dir, exist_ok=True) # Ensure reports subdir exists


            filepath = filedialog.asksaveasfilename(
                parent=self, initialdir=initial_dir, title="Save Report As",
                initialfile=default_filename, defaultextension=".txt",
                filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filepath:
                try:
                    os.makedirs(os.path.dirname(filepath), exist_ok=True) # Ensure dir exists
                    with open(filepath, 'w', encoding='utf-8') as f: f.write(report_content)
                    logger.info(f"Report saved to: {filepath}")
                    messagebox.showinfo("Success", f"Report saved to:\n{filepath}", parent=self)
                except Exception as e:
                     logger.error(f"Failed to save report to '{filepath}': {e}", exc_info=True)
                     messagebox.showerror("Save Error", f"Failed to save report: {e}", parent=self, icon='error')
            else: logger.info("Report save cancelled.")
        except tk.TclError as e:
            logger.error(f"Error getting report content from text widget: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to get report content: {e}", parent=self, icon='error')
        except Exception as e:
            logger.error(f"Unexpected error during report save: {e}", exc_info=True)
            messagebox.showerror("Error", f"Failed to save report: {e}", parent=self, icon='error')


    # --- Window Closing ---
    def _on_closing(self):
        """Handles window close event."""
        logger.info("Close button clicked.")
        # Check if optimization thread is running
        if self.optimization_thread and self.optimization_thread.is_alive():
            logger.warning("Optimization thread is still running.")
            if messagebox.askokcancel("Quit", "Optimization is still running. Quit anyway?", icon='warning', parent=self):
                logger.info("User chose to quit despite running optimization.")
                # Note: Daemon thread will be terminated abruptly. Consider graceful shutdown if needed.
                self.destroy()
            else:
                logger.info("Quit cancelled by user.")
                return # Abort closing
        elif messagebox.askokcancel("Quit", "Do you want to quit the application?", parent=self):
            logger.info("Exiting application.")
            # Optional: Auto-save config on exit?
            # try: self.save_configuration()
            # except Exception as e: logger.error(f"Error auto-saving config on exit: {e}")
            self.destroy()
        else:
             logger.info("Quit cancelled by user.")


# --- Main Application Entry Point ---
def main():
    """Main entry function to set up and run the GUI application."""
    # Setup TTK theme (optional but recommended)
    root_temp = tk.Tk()
    root_temp.withdraw()
    try:
        style = ttk.Style(root_temp)
        available_themes = style.theme_names()
        logger.debug(f"Available TTK themes: {available_themes}")
        # Prefer modern themes if available
        preferred_themes = ['clam', 'alt', 'vista', 'xpnative', 'aqua', 'default']
        chosen_theme = None
        for theme in preferred_themes:
            if theme in available_themes:
                try:
                    logger.debug(f"Attempting TTK theme: {theme}")
                    style.theme_use(theme)
                    chosen_theme = theme
                    break
                except tk.TclError: continue # Ignore themes that fail
        if chosen_theme: logger.info(f"Using TTK theme: {chosen_theme}")
        else: logger.warning("Could not apply preferred TTK theme. Using default.")
        # Define custom styles if needed (e.g., for buttons, labels)
        style.configure("Warning.TLabel", foreground="orange")
        style.configure("Error.TLabel", foreground="red")

    except Exception as e:
        logger.warning(f"Could not configure ttk theme: {e}")
    finally:
        root_temp.destroy()

    logger.info("Creating and running the application...")
    app = MainWindow() # Initialization handles critical error checks now
    if app.winfo_exists(): # Check if window was successfully created
        app.mainloop()
    else:
         logger.critical("Application failed to initialize properly. Exiting.")
    logger.info("Application finished.")


# Standard Python entry point check
if __name__ == "__main__":
    # Path setup and critical imports are checked within MainWindow.__init__ now
    main()