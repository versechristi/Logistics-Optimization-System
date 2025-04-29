# core/__init__.py
# -*- coding: utf-8 -*-
"""
__init__.py for the core package.

Exposes key components and functions from the core modules:
- distance_calculator (haversine)
- cost_function (calculate_total_cost_and_evaluate)
- problem_utils (SolutionCandidate, create_heuristic_trips_split_delivery, etc.)
- route_optimizer (run_optimization - the main orchestration function)

This allows other modules (e.g., gui/main_window.py) to easily
import and use these core components.
"""

import sys
import os
import traceback

# Optional: Add the project root to sys.path for more robust imports.
# If the project is run correctly from the root, this might not be strictly
# necessary here, as core is a top-level package. But it adds robustness.
try:
    # Assumes this file is in project_root/core
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_core_init = os.path.dirname(current_dir)
    if project_root_core_init not in sys.path:
        sys.path.insert(0, project_root_core_init)
        # print(f"Core __init__: Added project root to sys.path: {project_root_core_init}") # Optional debug print
except Exception as e:
    print(f"Warning setting up sys.path in core.__init__: {e}")
    traceback.print_exc()


# Import key components from core modules using relative imports
# Wrap each import in try/except to handle potential errors gracefully.

# from distance_calculator.py
try:
    from .distance_calculator import haversine
    # print("Successfully imported haversine") # Debug print
except ImportError as e:
    print(f"Error importing haversine from distance_calculator.py: {e}")
    # Define a dummy function if import fails
    def haversine(*args, **kwargs):
        print("ERROR: haversine failed to load.")
        return float('inf')
    # traceback.print_exc()


# from cost_function.py
try:
    from .cost_function import calculate_total_cost_and_evaluate
    # print("Successfully imported calculate_total_cost_and_evaluate") # Debug print
except ImportError as e:
    print(f"Error importing calculate_total_cost_and_evaluate from cost_function.py: {e}")
    # Define a dummy function if import fails
    def calculate_total_cost_and_evaluate(*args, **kwargs):
         print("ERROR: calculate_total_cost_and_evaluate failed to load.")
         # Return a structure matching the expected output, with error flags
         total_unmet_on_error = sum(kwargs.get('initial_demands',{}).values()) if isinstance(kwargs.get('initial_demands'), dict) else float('inf')
         return float('inf'), float('inf'), total_unmet_on_error, False, {}, True, True
    # traceback.print_exc()


# from problem_utils.py (the newly created shared utilities)
# We need to expose SolutionCandidate and the Stage 2 generator mainly
try:
    from .problem_utils import (
        SolutionCandidate,
        create_heuristic_trips_split_delivery,
        create_initial_solution_mdsd, # Expose initial solution generator as well
        generate_neighbor_solution_mdsd, # Expose neighbor generator (used by SA)
        swap_mutation, # Expose mutation operators (used by GA, SA neighbor)
        scramble_mutation,
        inversion_mutation,
        # Add any other critical utilities moved here that need external access
    )
    # print("Successfully imported components from problem_utils") # Debug print
except ImportError as e:
    print(f"Error importing components from problem_utils.py: {e}")
    # Define dummy components if import fails
    class SolutionCandidate:
         def __init__(self, sol): self.solution = sol; self.fitness = float('inf')
         def evaluate(self, *args, **kwargs): print("ERROR: Dummy evaluate in Problem Utils!")
         def __lt__(self, other): return False
         def __repr__(self): return "DummySolutionCandidate_Error()"

    def create_heuristic_trips_split_delivery(*args, **kwargs): print("ERROR: Dummy Stage 2 generator!"); return [], [], False
    def create_initial_solution_mdsd(*args, **kwargs): print("ERROR: Dummy initial solution generator!"); return None
    def generate_neighbor_solution_mdsd(*args, **kwargs): print("ERROR: Dummy neighbor generator!"); return None
    def swap_mutation(route): print("ERROR: Dummy mutation!"); return route
    def scramble_mutation(route): print("ERROR: Dummy mutation!"); return route
    def inversion_mutation(route): print("ERROR: Dummy mutation!"); return route
    # traceback.print_exc()


# from route_optimizer.py (the main orchestration logic)
# This is the primary function called by the GUI to start the process.
try:
    from .route_optimizer import run_optimization
    # print("Successfully imported run_optimization") # Debug print
except ImportError as e:
    print(f"Error importing run_optimization from route_optimizer.py: {e}")
    # Define a dummy function if import fails
    def run_optimization(*args, **kwargs):
        print("CRITICAL ERROR: run_optimization failed to load. Cannot run optimization.")
        # Return an empty dictionary indicating no results
        return {}
    # traceback.print_exc()


# Explicitly define what is exposed when someone imports the core package
__all__ = [
    'haversine',
    'calculate_total_cost_and_evaluate',
    'SolutionCandidate',
    'create_heuristic_trips_split_delivery',
    'create_initial_solution_mdsd',
    'generate_neighbor_solution_mdsd',
    'swap_mutation',
    'scramble_mutation',
    'inversion_mutation',
    'run_optimization', # Expose the main optimization runner
    # Add any other names imported above that you want to be part of 'from core import *'
]

# Note: While we expose some utility functions like initial/neighbor generators
# and mutations here, their primary use is within the algorithm files.
# Exposing them via __init__.py is useful if other parts of the code (like tests
# or a command-line interface) need to access them directly.
# The SolutionCandidate and create_heuristic_trips_split_delivery are definitely
# core utilities needed by multiple parts.