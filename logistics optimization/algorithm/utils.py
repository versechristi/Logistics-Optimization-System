# algorithm/utils.py
# -*- coding: utf-8 -*-
"""
Algorithm shared utility functions (minimal version).

Note: Core problem-specific utility functions (like Stage 2 trip generation,
SolutionCandidate class, initial/neighbor solution generation, and permutation
mutations) have been moved to core/problem_utils.py to resolve circular dependencies.

This file now primarily serves to ensure correct imports and potentially
contain any remaining algorithm-specific helper functions not related
to core problem logic or solution structure.
"""

import sys
import os
import traceback

# Attempt to ensure the project root is in sys.path for robust imports
try:
    # Assumes this file is in project_root/algorithm
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_algo_utils = os.path.dirname(current_dir)
    if project_root_algo_utils not in sys.path:
        sys.path.insert(0, project_root_algo_utils)
        # print(f"Algorithm Utils: Added project root to sys.path: {project_root_algo_utils}") # Optional debug print

    # --- NO direct imports from core.cost_function or algorithm.utils (self) ---
    # The functions and classes previously imported or defined here
    # (like create_heuristic_trips_split_delivery, SolutionCandidate,
    # create_initial_solution_mdsd, generate_neighbor_solution_mdsd,
    # swap_mutation, scramble_mutation, inversion_mutation)
    # are now located in core.problem_utils.
    # Algorithms should import them from core.problem_utils.

    # You might need to import other standard libraries or truly algorithm-specific
    # helpers here if they exist in your original, non-problem-core utils.py.
    # For now, assuming minimal content.

except ImportError as e:
    print(f"CRITICAL ERROR (algorithm.utils): Failed during initial import block: {e}")
    traceback.print_exc()
    # If path setup fails, other imports will likely also fail.


# --- Remaining algorithm-specific helper functions ---
# Add any functions here that were in your original algorithm/utils.py
# but were *not* moved to core/problem_utils.py and are genuinely
# algorithm-specific (e.g., specific selection methods, crossover logic
# that isn't a general permutation operator, etc.).
# If there are none, this section remains empty.

# Example: (Remove or replace with your actual code if needed)
# def some_algorithm_specific_helper(data, params):
#     """A placeholder for a function specific to algorithms."""
#     pass


# Note: The SolutionCandidate class, initial solution generation,
# neighbor generation, and mutation operators are now in core.problem_utils.
# Algorithms needing these functionalities should import them from there.