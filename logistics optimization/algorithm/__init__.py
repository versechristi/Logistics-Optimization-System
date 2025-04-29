# algorithm/__init__.py
# -*- coding: utf-8 -*-
"""
__init__.py for the algorithm package.

Exposes the main entry functions for each implemented optimization algorithm.
This allows other modules (e.g., core/route_optimizer.py) to easily
import and use the algorithms like:
from algorithm import run_genetic_algorithm, run_simulated_annealing, etc.
"""

# Attempt to import the main run functions from each algorithm module
# These functions are the public interface of the algorithms.
# Wrap in try/except for robustness in case a module is missing or has errors.

import sys
import os
import traceback

# Optional: Add the project root to sys.path for more robust imports, though
# if the project is run correctly from the root, it might not be necessary here.
# This is more crucial in individual script files or submodules.
try:
    # Assumes this file is in project_root/algorithm
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_algo_init = os.path.dirname(current_dir)
    if project_root_algo_init not in sys.path:
        sys.path.insert(0, project_root_algo_init)
        # print(f"Algorithm __init__: Added project root to sys.path: {project_root_algo_init}") # Optional debug print
except Exception as e:
    print(f"Warning setting up sys.path in algorithm.__init__: {e}")
    traceback.print_exc()


# Import the main run functions
try:
    from .genetic_algorithm import run_genetic_algorithm
    # print("Successfully imported run_genetic_algorithm") # Debug print
except ImportError as e:
    print(f"Error importing run_genetic_algorithm from genetic_algorithm.py: {e}")
    # Define a dummy function or set to None if import fails, to prevent NameErrors later
    def run_genetic_algorithm(*args, **kwargs):
        print("ERROR: genetic_algorithm failed to load.")
        return None
    # traceback.print_exc()


try:
    from .simulated_annealing import run_simulated_annealing
    # print("Successfully imported run_simulated_annealing") # Debug print
except ImportError as e:
    print(f"Error importing run_simulated_annealing from simulated_annealing.py: {e}")
    def run_simulated_annealing(*args, **kwargs):
        print("ERROR: simulated_annealing failed to load.")
        return None
    # traceback.print_exc()


try:
    from .pso_optimizer import run_pso_optimizer
    # print("Successfully imported run_pso_optimizer") # Debug print
except ImportError as e:
    print(f"Error importing run_pso_optimizer from pso_optimizer.py: {e}")
    def run_pso_optimizer(*args, **kwargs):
        print("ERROR: pso_optimizer failed to load.")
        return None
    # traceback.print_exc()

try:
    from .greedy_heuristic import run_greedy_heuristic
    # print("Successfully imported run_greedy_heuristic") # Debug print
except ImportError as e:
    print(f"Error importing run_greedy_heuristic from greedy_heuristic.py: {e}")
    def run_greedy_heuristic(*args, **kwargs):
        print("ERROR: greedy_heuristic failed to load.")
        return None
    # traceback.print_exc()


# Explicitly define what is exposed when someone imports the package
# This is good practice (__all__) but optional.
__all__ = [
    'run_genetic_algorithm',
    'run_simulated_annealing',
    'run_pso_optimizer',
    'run_greedy_heuristic',
]

# You could potentially import other algorithm-related helpers here if needed
# e.g., from .utils import some_helper_function
# But since algorithm/utils is minimal now, this might not be necessary.