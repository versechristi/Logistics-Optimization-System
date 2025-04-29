# core/problem_utils.py
# -*- coding: utf-8 -*-
"""
Core utility functions specific to the Multi-Depot, Two-Echelon
Vehicle Routing Problem with Drones and Split Deliveries (MD-2E-VRPSD).

These functions are shared across different modules (e.g., core, algorithm)
and are placed here to provide common building blocks and data structures
for solving the MD-2E-VRPSD, helping to avoid circular dependencies.

Includes:
- Heuristic Stage 2 trip generation (allowing split deliveries) from a single outlet.
- A class to represent a solution candidate for the multi-depot structure and its evaluation results.
- Functions for generating initial solutions for the multi-depot problem.
- Functions for generating neighbor solutions (perturbations) for multi-depot routes.
- Permutation-based mutation operators used by algorithms.
"""

import copy
import math
import traceback
import sys
import os
import random
import numpy as np
import time # Used for optional timing in standalone execution
import warnings # Use warnings for non-critical issues

# --- Safe Imports for Dependencies ---
# Attempt to import necessary core modules safely.
# This block attempts to add the project root to sys.path if necessary,
# making imports of core modules more robust when scripts are run
# from different directories.
try:
    # Assuming this file is in project_root/core
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_problem_utils = os.path.dirname(current_dir)
    if project_root_problem_utils not in sys.path:
        sys.path.insert(0, project_root_problem_utils)
        # print(f"Problem Utils: Added project root to sys.path: {project_root_problem_utils}") # Optional debug print

    # --- Import necessary components from other core modules ---
    # Import distance calculator - crucial for assignment and cost calculation
    from core.distance_calculator import haversine
    # Import the cost function - needed by SolutionCandidate's evaluate method
    # Note: The cost function itself should accept distance_func and Stage 2 generator as params
    # to avoid direct circular imports if cost_function needed problem_utils
    from core.cost_function import calculate_total_cost_and_evaluate

except ImportError as e:
    print(f"CRITICAL ERROR in core.problem_utils: Failed during initial import block: {e}")
    traceback.print_exc()
    # Define dummy functions/classes if imports fail to prevent immediate crash
    # but indicate severe error.
    def haversine(coord1, coord2):
        warnings.warn("DUMMY haversine called in problem_utils due to import error!")
        # Simple Manhattan distance for dummy
        if not coord1 or not coord2 or len(coord1) != 2 or len(coord2) != 2: return float('inf')
        try: return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) * 100
        except: return float('inf')

    def calculate_total_cost_and_evaluate(*args, **kwargs):
         warnings.warn("DUMMY calculate_total_cost_and_evaluate called in problem_utils due to import error!")
         # Return tuple matching expected structure with error state
         return float('inf'), float('inf'), float('inf'), {}, True, True, {}

    class SolutionCandidate:
        # Dummy version for when imports fail
        def __init__(self, *args, **kwargs):
            warnings.warn("DUMMY SolutionCandidate created due to import error!")
            self._initialize_invalid_state(error_msg="Import Failed")

        def evaluate(self, *args, **kwargs):
             warnings.warn("DUMMY SolutionCandidate evaluate called!")
             pass # Already invalid

        def __lt__(self, other):
             # Comparison prioritizes feasibility then cost
             if not isinstance(other, SolutionCandidate): return NotImplemented
             if self.is_feasible and not other.is_feasible: return True
             if not self.is_feasible and other.is_feasible: return False
             return self.weighted_cost < other.weighted_cost

        def _initialize_invalid_state(self, error_msg="Invalid State"):
             # Helper to set attributes for an invalid state
             self.all_locations = {}
             self.demands = []
             self.vehicle_params = {}
             self.drone_params = {}
             self.unmet_demand_penalty = float('inf')
             self.cost_weight = 1.0
             self.time_weight = 0.0
             self.stage1_routes = {}
             self.outlet_to_depot_assignments = {}
             self.customer_to_outlet_assignments = {}
             self.stage2_trips = {}
             self.is_feasible = False
             self.weighted_cost = float('inf')
             self.evaluated_cost = float('inf')
             self.evaluated_time = float('inf')
             self.evaluated_unmet = float('inf')
             self.served_customer_details = {}
             self.evaluation_stage1_error = True
             self.evaluation_stage2_error = True
             self.initialization_error = error_msg # Store error reason

        def __repr__(self):
            return f"DummySolutionCandidate(Error='{getattr(self, 'initialization_error', 'Unknown')}')"


    warnings.warn("Problem_utils will use dummy functions/classes due to critical import failure.")

except Exception as e:
    print(f"An unexpected error occurred during problem_utils import block: {e}")
    traceback.print_exc()
    # Consider how to handle this - maybe re-raise or exit if core utils are essential
    raise e # Re-raise critical error


# Define a small tolerance for floating-point comparisons (e.g., checking if demand is zero)
FLOAT_TOLERANCE_UTILS = 1e-6


# --- Solution Candidate Class ---
class SolutionCandidate:
    """
    Represents a candidate solution for the Multi-Depot, Two-Echelon VRP with
    Drones and Split Deliveries (MD-2E-VRPSD).

    A solution includes the structure of the routes and assignments,
    as well as the results of its evaluation (cost, time, unmet demand).
    Designed to store and compare solutions in the context of optimization algorithms.
    """
    def __init__(self, problem_data: dict, vehicle_params: dict, drone_params: dict,
                 unmet_demand_penalty: float, cost_weight: float = 1.0, time_weight: float = 0.0,
                 initial_stage1_routes: dict | None = None,
                 initial_outlet_to_depot_assignments: dict | None = None,
                 initial_customer_to_outlet_assignments: dict | None = None):
        """
        Initializes a SolutionCandidate.

        Args:
            problem_data (dict): Dictionary containing all problem instance data:
                                 'locations': {'logistics_centers': [...], 'sales_outlets': [...], 'customers': [...]},
                                 'demands': [...] (list of customer demands).
            vehicle_params (dict): Dictionary of vehicle parameters (e.g., 'payload', 'cost_per_km', 'speed_kmph').
            drone_params (dict): Dictionary of drone parameters (e.g., 'payload', 'cost_per_km', 'speed_kmph', 'max_flight_distance_km').
            unmet_demand_penalty (float): The penalty cost incurred for each unit of unmet demand.
            cost_weight (float): Weight for the raw cost in the weighted objective function. Defaults to 1.0.
            time_weight (float): Weight for the total time (makespan) in the weighted objective function. Defaults to 0.0.
            initial_stage1_routes (dict | None): Optional dictionary mapping depot index to a list of assigned outlet indices
                                                representing the Stage 1 route from that depot. If None, will be initialized
                                                as empty or via heuristic later.
                                                Example: {0: [3, 1], 1: [4, 2]}
            initial_outlet_to_depot_assignments (dict | None): Optional dictionary mapping outlet index to the index of its assigned depot.
                                                             If None, will be initialized via heuristic later.
                                                             Example: {0: 0, 1: 0, 2: 1, 3: 0, 4: 1}
            initial_customer_to_outlet_assignments (dict | None): Optional dictionary mapping customer index to the index of its assigned sales outlet.
                                                                 If None, will be initialized via heuristic later.
                                                                 Example: {0: 3, 1: 3, 2: 1, 3: 4, 4: 2, ...}

        Attributes:
            all_locations (dict): Stores the locations data from problem_data.
            demands (list): Stores the customer demands from problem_data.
            vehicle_params (dict): Stores vehicle parameters.
            drone_params (dict): Stores drone parameters.
            unmet_demand_penalty (float): Penalty for unmet demand.
            cost_weight (float): Weight for cost.
            time_weight (float): Weight for time.

            stage1_routes (dict): Key: depot index (int), Value: list of assigned sales outlet indices (int) in visit order.
                                 This is the primary part modified by optimization algorithms.
                                 Example: {0: [3, 1], 1: [4, 2]}
            outlet_to_depot_assignments (dict): Key: sales outlet index (int), Value: depot index (int).
                                              Determined during initialization/preprocessing.
                                              Example: {0: 0, 1: 0, 2: 1, 3: 0, 4: 1}
            customer_to_outlet_assignments (dict): Key: customer index (int), Value: sales outlet index (int).
                                                  Determined during initialization/preprocessing.
                                                  Example: {0: 3, 1: 3, 2: 1, 3: 4, 4: 2, ...}
            stage2_trips (dict): Key: sales outlet index (int), Value: list of generated Stage 2 trip details from this outlet.
                                This is generated heuristically during evaluation, not directly by algorithms.
                                Example: {1: [{'type': 'vehicle', 'route': [2, 5], 'cost': 50, 'time': 1.5, 'load': 150}],
                                          3: [{'type': 'drone', 'route': [0], 'cost': 10, 'time': 0.5, 'load': 30},
                                              {'type': 'vehicle', 'route': [1], 'cost': 20, 'time': 1.0, 'load': 70}]}

            # Evaluation Results (set by the evaluate method)
            is_feasible (bool): True if all customer demand is met (within tolerance).
            weighted_cost (float): The total weighted objective value, including unmet demand penalty. Used for comparison.
            evaluated_cost (float): The raw total cost (Stage 1 + Stage 2).
            evaluated_time (float): The total makespan time (Stage 1 makespan + Stage 2 makespan).
            evaluated_unmet (float): The total amount of unmet customer demand.
            served_customer_details (dict): Key: customer index, Value: {'initial': ..., 'satisfied': ..., 'remaining': ..., 'status': ...}.
            evaluation_stage1_error (bool): True if an error occurred during Stage 1 evaluation.
            evaluation_stage2_error (bool): True if an error occurred during Stage 2 evaluation.
            initialization_error (str | None): Stores any error message during initialization.
        """
        # --- Store Problem Data and Parameters ---
        self.initialization_error = None # Track init errors
        try:
            if not problem_data or 'locations' not in problem_data or 'demands' not in problem_data:
                raise ValueError("Invalid 'problem_data' structure provided.")

            self.all_locations = problem_data.get('locations', {})
            self.demands = problem_data.get('demands', [])
            self.vehicle_params = copy.deepcopy(vehicle_params) if vehicle_params else {}
            self.drone_params = copy.deepcopy(drone_params) if drone_params else {}

            if not isinstance(unmet_demand_penalty, (int, float)): raise ValueError("unmet_demand_penalty must be numeric.")
            if not isinstance(cost_weight, (int, float)): raise ValueError("cost_weight must be numeric.")
            if not isinstance(time_weight, (int, float)): raise ValueError("time_weight must be numeric.")
            self.unmet_demand_penalty = float(unmet_demand_penalty)
            self.cost_weight = float(cost_weight)
            self.time_weight = float(time_weight)

            # Ensure demands list size matches customer count
            num_customers_in_data = len(self.all_locations.get('customers', []))
            if len(self.demands) != num_customers_in_data:
                 warnings.warn(f"Mismatch between customer locations ({num_customers_in_data}) and demand list size ({len(self.demands)}). Adjusting demands list.")
                 # Pad demands with zeros or truncate to match customer count
                 if len(self.demands) < num_customers_in_data:
                      self.demands.extend([0.0] * (num_customers_in_data - len(self.demands)))
                 elif len(self.demands) > num_customers_in_data:
                      self.demands = self.demands[:num_customers_in_data]

        except Exception as e:
             warnings.warn(f"Error during SolutionCandidate parameter initialization: {e}")
             self._initialize_invalid_state(error_msg=str(e)) # Use helper to set invalid state
             return # Stop further initialization

        # --- Initialize Solution Structure (Assignments and Routes) ---
        self.outlet_to_depot_assignments = copy.deepcopy(initial_outlet_to_depot_assignments) if initial_outlet_to_depot_assignments is not None else {}
        self.customer_to_outlet_assignments = copy.deepcopy(initial_customer_to_outlet_assignments) if initial_customer_to_outlet_assignments is not None else {}
        self.stage1_routes = copy.deepcopy(initial_stage1_routes) if initial_stage1_routes is not None else {}
        self.stage2_trips = {} # Always initialized empty, generated during evaluation

        # --- Initialize Evaluation Results ---
        self.is_feasible = False
        self.weighted_cost = float('inf')
        self.evaluated_cost = float('inf')
        self.evaluated_time = float('inf')
        self.evaluated_unmet = float('inf')
        self.served_customer_details = {}
        self.evaluation_stage1_error = False # Assume no error until evaluated
        self.evaluation_stage2_error = False # Assume no error until evaluated

    def _initialize_invalid_state(self, error_msg="Invalid State"):
         """Helper to initialize the solution state to an invalid/error state."""
         self.all_locations = {}
         self.demands = []
         self.vehicle_params = {}
         self.drone_params = {}
         self.unmet_demand_penalty = float('inf') # High penalty for invalid state
         self.cost_weight = 1.0
         self.time_weight = 0.0
         self.stage1_routes = {}
         self.outlet_to_depot_assignments = {}
         self.customer_to_outlet_assignments = {}
         self.stage2_trips = {}
         self.is_feasible = False
         self.weighted_cost = float('inf')
         self.evaluated_cost = float('inf')
         self.evaluated_time = float('inf')
         self.evaluated_unmet = float('inf') # Max possible unmet demand? Or Inf?
         self.served_customer_details = {}
         self.evaluation_stage1_error = True
         self.evaluation_stage2_error = True
         self.initialization_error = error_msg


    def evaluate(self, distance_func: callable, stage2_trip_generator_func: callable,
                 cost_weight: float | None = None, time_weight: float | None = None,
                 unmet_demand_penalty: float | None = None):
        """
        Evaluates the current solution candidate (its routes and assignments).

        This method calculates the total cost, time (makespan), unmet demand,
        and feasibility of the solution based on its current `stage1_routes`,
        `outlet_to_depot_assignments`, and `customer_to_outlet_assignments`.
        It uses the provided distance function and Stage 2 trip generator function.

        Args:
            distance_func (callable): A function that takes two location coordinates (lat, lon)
                                      tuples and returns the distance between them.
                                      Signature: distance_func(coord1, coord2) -> float.
            stage2_trip_generator_func (callable): A function that takes outlet index, assigned
                                                   customer indices, problem data (locations, demands),
                                                   vehicle params, drone params, and global remaining
                                                   demands, and returns a list of generated Stage 2
                                                   trip details from that outlet. It should also
                                                   update the global remaining demands.
                                                   Signature: stage2_trip_generator_func(outlet_index,
                                                   assigned_customer_indices, problem_data,
                                                   vehicle_params, drone_params, demands_remaining_global)
                                                   -> list of trip dicts.
            cost_weight (float | None): Optional weight for the raw cost for this specific evaluation.
                                        If None, uses the weight stored in the object.
            time_weight (float | None): Optional weight for time/makespan for this specific evaluation.
                                        If None, uses the weight stored in the object.
            unmet_demand_penalty (float | None): Optional penalty for unmet demand for this specific evaluation.
                                                 If None, uses the penalty stored in the object.
        """
        # Check if the candidate was initialized with an error
        if self.initialization_error:
            warnings.warn(f"Evaluation skipped: SolutionCandidate was initialized with error: {self.initialization_error}")
            # Ensure evaluation results reflect the error state
            self.is_feasible = False
            self.weighted_cost = float('inf')
            self.evaluation_stage1_error = True
            self.evaluation_stage2_error = True
            return

        # Use provided weights/penalty if not None, otherwise use instance attributes
        eval_cost_weight = cost_weight if cost_weight is not None else self.cost_weight
        eval_time_weight = time_weight if time_weight is not None else self.time_weight
        eval_unmet_penalty = unmet_demand_penalty if unmet_demand_penalty is not None else self.unmet_demand_penalty

        # Reset evaluation results before re-evaluation
        self.is_feasible = False
        self.weighted_cost = float('inf')
        self.evaluated_cost = float('inf')
        self.evaluated_time = float('inf')
        self.evaluated_unmet = float('inf')
        self.served_customer_details = {}
        self.evaluation_stage1_error = False
        self.evaluation_stage2_error = False
        self.stage2_trips = {}

        if not callable(distance_func) or not callable(stage2_trip_generator_func):
            warnings.warn("Evaluation failed: Invalid distance_func or stage2_trip_generator_func provided.")
            self.evaluation_stage1_error = True # Mark as error
            self.evaluation_stage2_error = True
            return

        try:
            # Call the core cost calculation function
            (total_raw_cost, total_time_makespan, final_unmet_demand,
             served_customer_details, eval_s1_error, eval_s2_error,
             stage2_trips_details) = calculate_total_cost_and_evaluate(
                 stage1_routes=self.stage1_routes,
                 outlet_to_depot_assignments=self.outlet_to_depot_assignments,
                 customer_to_outlet_assignments=self.customer_to_outlet_assignments,
                 problem_data={'locations': self.all_locations, 'demands': self.demands},
                 vehicle_params=self.vehicle_params,
                 drone_params=self.drone_params,
                 distance_func=distance_func,
                 stage2_trip_generator_func=stage2_trip_generator_func,
                 unmet_demand_penalty=eval_unmet_penalty,
                 cost_weight=eval_cost_weight,
                 time_weight=eval_time_weight
             )

            # Update the instance attributes with the evaluation results
            self.evaluated_cost = total_raw_cost
            self.evaluated_time = total_time_makespan
            self.evaluated_unmet = final_unmet_demand
            self.served_customer_details = served_customer_details
            self.evaluation_stage1_error = eval_s1_error
            self.evaluation_stage2_error = eval_s2_error
            self.stage2_trips = stage2_trips_details

            # Calculate weighted cost, handling potential infinities
            safe_raw_cost = self.evaluated_cost if not math.isinf(self.evaluated_cost) and not math.isnan(self.evaluated_cost) else float('inf')
            safe_time = self.evaluated_time if not math.isinf(self.evaluated_time) and not math.isnan(self.evaluated_time) else float('inf')
            safe_unmet = self.evaluated_unmet if not math.isinf(self.evaluated_unmet) and not math.isnan(self.evaluated_unmet) else float('inf')

            # Ensure weights/penalty are finite
            safe_cost_weight = eval_cost_weight if math.isfinite(eval_cost_weight) else 0.0
            safe_time_weight = eval_time_weight if math.isfinite(eval_time_weight) else 0.0
            safe_unmet_penalty = eval_unmet_penalty if math.isfinite(eval_unmet_penalty) else 0.0

            term_cost = safe_cost_weight * safe_raw_cost
            term_time = safe_time_weight * safe_time
            term_unmet = safe_unmet_penalty * safe_unmet

            if math.isinf(term_cost) or math.isinf(term_time) or math.isinf(term_unmet):
                self.weighted_cost = float('inf')
            else:
                self.weighted_cost = term_cost + term_time + term_unmet

            # Determine feasibility (within tolerance)
            self.is_feasible = (not self.evaluation_stage1_error and
                                not self.evaluation_stage2_error and
                                self.evaluated_unmet is not None and
                                math.isfinite(self.evaluated_unmet) and
                                abs(self.evaluated_unmet) < FLOAT_TOLERANCE_UTILS)

            # If evaluation had errors, ensure cost is infinite and not feasible
            if self.evaluation_stage1_error or self.evaluation_stage2_error:
                 self.is_feasible = False
                 self.weighted_cost = float('inf')

        except Exception as e:
             warnings.warn(f"Unexpected error during SolutionCandidate evaluation: {e}")
             traceback.print_exc()
             # Set to invalid state on unexpected error during evaluation
             self.is_feasible = False
             self.weighted_cost = float('inf')
             self.evaluated_cost = float('inf')
             self.evaluated_time = float('inf')
             self.evaluated_unmet = float('inf')
             self.evaluation_stage1_error = True
             self.evaluation_stage2_error = True


    def __lt__(self, other):
        """
        Compares this SolutionCandidate to another based on feasibility and weighted cost.
        Used by optimization algorithms to determine which solution is 'better'.

        Comparison Logic (Feasibility First, then Weighted Cost):
        1. A feasible solution is always better than an unfeasible solution.
        2. If both are feasible or both are unfeasible, the one with the lower
           weighted_cost is better.
        3. Handles None/inf/NaN weighted_cost gracefully (inf is considered worse).
        """
        if not isinstance(other, SolutionCandidate):
            # Cannot compare with non-SolutionCandidate types
            return NotImplemented

        # 1. Prioritize feasibility
        if self.is_feasible and not other.is_feasible:
            return True # self is feasible, other is not -> self is better
        if not self.is_feasible and other.is_feasible:
            return False # self is not feasible, other is feasible -> other is better

        # 2. If feasibility is the same (both feasible or both unfeasible), compare weighted cost
        # Handle None, Inf, NaN values for robust comparison
        self_cost = self.weighted_cost if self.weighted_cost is not None and not math.isnan(self.weighted_cost) else float('inf')
        other_cost = other.weighted_cost if other.weighted_cost is not None and not math.isnan(other.weighted_cost) else float('inf')

        # Add tolerance for float comparison if needed, though direct < should usually be okay here
        # Example with tolerance: return self_cost < other_cost - FLOAT_TOLERANCE_UTILS
        return self_cost < other_cost


    def __repr__(self):
        """Provides a developer-friendly string representation of the SolutionCandidate."""
        # Include error status if initialization failed
        if self.initialization_error:
            return f"SolutionCandidate(Error='{self.initialization_error}')"

        cost_str = format_float(self.weighted_cost, 4)
        raw_cost_str = format_float(self.evaluated_cost, 2)
        time_str = format_float(self.evaluated_time, 2)
        unmet_str = format_float(self.evaluated_unmet, 2)
        num_depots_served = len(self.stage1_routes) if self.stage1_routes else 0

        return (f"SolutionCandidate(Feasible={self.is_feasible}, "
                f"WCost={cost_str}, RawCost={raw_cost_str}, "
                f"Time={time_str}, Unmet={unmet_str}, "
                f"Depots={num_depots_served})")

    def __copy__(self):
        """Creates a shallow copy of the SolutionCandidate. Use deepcopy for safety."""
        # Warning: Shallow copy might not be safe if attributes are modified later.
        # Using deepcopy is generally recommended for complex objects in optimization.
        warnings.warn("Shallow copy used for SolutionCandidate. Use deepcopy for safety.", UserWarning)
        new_solution = SolutionCandidate(
            problem_data={'locations': self.all_locations, 'demands': self.demands},
            vehicle_params=self.vehicle_params,
            drone_params=self.drone_params,
            unmet_demand_penalty=self.unmet_demand_penalty,
            cost_weight=self.cost_weight,
            time_weight=self.time_weight,
            initial_stage1_routes=self.stage1_routes, # Shallow copy of dict/list
            initial_outlet_to_depot_assignments=self.outlet_to_depot_assignments, # Shallow copy
            initial_customer_to_outlet_assignments=self.customer_to_outlet_assignments # Shallow copy
        )
        # Copy evaluation results (numeric/bool are immutable, dicts are shallow copied)
        new_solution.is_feasible = self.is_feasible
        new_solution.weighted_cost = self.weighted_cost
        new_solution.evaluated_cost = self.evaluated_cost
        new_solution.evaluated_time = self.evaluated_time
        new_solution.evaluated_unmet = self.evaluated_unmet
        new_solution.served_customer_details = self.served_customer_details # Shallow copy
        new_solution.evaluation_stage1_error = self.evaluation_stage1_error
        new_solution.evaluation_stage2_error = self.evaluation_stage2_error
        new_solution.stage2_trips = self.stage2_trips # Shallow copy
        new_solution.initialization_error = self.initialization_error

        return new_solution

    def __deepcopy__(self, memo):
        """Creates a deep copy of the SolutionCandidate."""
        # Create new instance without calling __init__ again to avoid re-validation/warnings
        cls = self.__class__
        new_solution = cls.__new__(cls)
        memo[id(self)] = new_solution # Add to memo early to handle cycles

        # Deep copy relevant attributes
        new_solution.all_locations = copy.deepcopy(self.all_locations, memo)
        new_solution.demands = copy.deepcopy(self.demands, memo)
        new_solution.vehicle_params = copy.deepcopy(self.vehicle_params, memo)
        new_solution.drone_params = copy.deepcopy(self.drone_params, memo)
        new_solution.unmet_demand_penalty = self.unmet_demand_penalty # float is immutable
        new_solution.cost_weight = self.cost_weight # float is immutable
        new_solution.time_weight = self.time_weight # float is immutable
        new_solution.stage1_routes = copy.deepcopy(self.stage1_routes, memo)
        new_solution.outlet_to_depot_assignments = copy.deepcopy(self.outlet_to_depot_assignments, memo)
        new_solution.customer_to_outlet_assignments = copy.deepcopy(self.customer_to_outlet_assignments, memo)
        new_solution.stage2_trips = copy.deepcopy(self.stage2_trips, memo)

        # Copy evaluation results (bool/float are immutable, dict needs deepcopy)
        new_solution.is_feasible = self.is_feasible
        new_solution.weighted_cost = self.weighted_cost
        new_solution.evaluated_cost = self.evaluated_cost
        new_solution.evaluated_time = self.evaluated_time
        new_solution.evaluated_unmet = self.evaluated_unmet
        new_solution.served_customer_details = copy.deepcopy(self.served_customer_details, memo)
        new_solution.evaluation_stage1_error = self.evaluation_stage1_error
        new_solution.evaluation_stage2_error = self.evaluation_stage2_error
        new_solution.initialization_error = self.initialization_error # String is immutable

        return new_solution


# --- Heuristic Stage 2 Trip Generation ---
# (Assuming the definition provided previously is correct and reflects the needed logic)
def create_heuristic_trips_split_delivery(outlet_index: int,
                                           assigned_customer_indices: list[int],
                                           problem_data: dict,
                                           vehicle_params: dict,
                                           drone_params: dict,
                                           demands_remaining_global: dict) -> list[dict]:
    """
    Heuristically generates Stage 2 delivery trips from a single sales outlet
    to its assigned customers, allowing for split deliveries using both
    vehicles and drones.

    This is a greedy heuristic for Stage 2. It attempts to fulfill demand
    using available vehicles and drones, prioritizing vehicles or based on
    some simple logic (e.g., vehicle for larger loads, drone for nearby small loads).
    It updates the global `demands_remaining_global` dictionary to track fulfillment.

    Args:
        outlet_index (int): The index of the sales outlet (origin of trips).
        assigned_customer_indices (list[int]): A list of indices of customers assigned
                                               to this sales outlet.
        problem_data (dict): Dictionary containing 'locations' and 'demands'.
        vehicle_params (dict): Parameters for vehicles. Must include 'payload', 'cost_per_km', 'speed_kmph'.
        drone_params (dict): Parameters for drones. Must include 'payload', 'max_flight_distance_km', 'cost_per_km', 'speed_kmph'.
        demands_remaining_global (dict): A dictionary mapping customer index to their
                                         current remaining demand. This dictionary
                                         WILL BE MODIFIED by this function.

    Returns:
        list[dict]: A list of dictionaries, where each dictionary represents a trip
                    from the outlet. Trip details include 'type' ('vehicle' or 'drone'),
                    'route' (list of customer indices visited), 'load', 'cost', 'time'.
                    Returns an empty list if no assigned customers or on error.
    """
    generated_trips = []
    if not assigned_customer_indices: return generated_trips

    try:
        locations = problem_data.get('locations', {})
        all_demands = problem_data.get('demands', []) # Original full demands list
        outlet_locations = locations.get('sales_outlets', [])
        customer_locations = locations.get('customers', [])

        if not (0 <= outlet_index < len(outlet_locations)) or not outlet_locations[outlet_index]:
            warnings.warn(f"Invalid outlet_index {outlet_index} or missing location in create_heuristic_trips.")
            return generated_trips

        outlet_coord = outlet_locations[outlet_index]

        # Filter customers with positive remaining demand for this outlet
        customers_to_serve = [
            (idx, demands_remaining_global.get(idx, 0.0)) for idx in assigned_customer_indices
            if 0 <= idx < len(customer_locations) and demands_remaining_global.get(idx, 0.0) > FLOAT_TOLERANCE_UTILS
        ]

        # Sort by distance for a simple greedy approach
        customers_to_serve.sort(key=lambda item: haversine(outlet_coord, customer_locations[item[0]]))

        # Get vehicle/drone parameters safely
        vehicle_payload = vehicle_params.get('payload', 0.0)
        vehicle_cost_per_km = vehicle_params.get('cost_per_km', 0.0)
        vehicle_speed_kmph = vehicle_params.get('speed_kmph', 1.0) # Default to 1 to avoid div by zero

        drone_payload = drone_params.get('payload', 0.0)
        drone_max_dist = drone_params.get('max_flight_distance_km', 0.0) # Assume this is round trip limit for simplicity
        drone_cost_per_km = drone_params.get('cost_per_km', 0.0)
        drone_speed_kmph = drone_params.get('speed_kmph', 1.0) # Default to 1

        # --- Trip Generation Loop ---
        # Continue as long as there are customers with unmet demand assigned to this outlet
        while any(demands_remaining_global.get(idx, 0.0) > FLOAT_TOLERANCE_UTILS for idx in assigned_customer_indices):

            customers_served_in_this_pass = 0 # Track if any progress is made in a full pass

            # --- Attempt Vehicle Trip ---
            if vehicle_payload > FLOAT_TOLERANCE_UTILS:
                current_vehicle_load = 0.0
                current_vehicle_route_indices = []
                customers_to_remove_from_list = []

                # Greedily fill one vehicle trip
                for cust_idx, remaining_demand in list(customers_to_serve): # Iterate over copy
                    if remaining_demand <= FLOAT_TOLERANCE_UTILS: continue # Already served

                    load_can_add = vehicle_payload - current_vehicle_load
                    if load_can_add <= FLOAT_TOLERANCE_UTILS: break # Vehicle full

                    load_to_take = min(remaining_demand, load_can_add)

                    if load_to_take > FLOAT_TOLERANCE_UTILS:
                         current_vehicle_load += load_to_take
                         current_vehicle_route_indices.append(cust_idx)
                         demands_remaining_global[cust_idx] -= load_to_take
                         customers_served_in_this_pass += 1
                         # Update remaining demand for the *next* iteration/pass
                         new_remaining = demands_remaining_global[cust_idx]
                         if new_remaining < FLOAT_TOLERANCE_UTILS:
                              demands_remaining_global[cust_idx] = 0.0 # Clamp to zero
                              customers_to_remove_from_list.append(cust_idx)


                # If a vehicle trip was formed, calculate its metrics and add it
                if current_vehicle_route_indices:
                     # Calculate distance, time, cost for the vehicle trip (Outlet -> C1 -> ... -> Cn -> Outlet)
                     trip_coords = [outlet_coord] + [customer_locations[i] for i in current_vehicle_route_indices] + [outlet_coord]
                     trip_distance = sum(haversine(trip_coords[i], trip_coords[i+1]) for i in range(len(trip_coords)-1))
                     trip_time = trip_distance / vehicle_speed_kmph if vehicle_speed_kmph > FLOAT_TOLERANCE_UTILS else float('inf')
                     trip_cost = trip_distance * vehicle_cost_per_km

                     generated_trips.append({
                         'type': 'vehicle',
                         'route': current_vehicle_route_indices,
                         'load': current_vehicle_load,
                         'cost': trip_cost,
                         'time': trip_time
                     })

                     # Remove fully served customers from the list for the next pass
                     customers_to_serve = [(idx, dem) for idx, dem in customers_to_serve if idx not in customers_to_remove_from_list]
                     # Update demands in the main list (needed if drones run next)
                     customers_to_serve = [(idx, demands_remaining_global.get(idx, 0.0)) for idx, _ in customers_to_serve]


            # --- Attempt Drone Trips (for remaining demand after vehicle attempt) ---
            if drone_payload > FLOAT_TOLERANCE_UTILS and drone_max_dist > FLOAT_TOLERANCE_UTILS:
                 customers_served_by_drone_this_pass = 0
                 customers_to_remove_from_list_drone = []

                 for cust_idx, remaining_demand in list(customers_to_serve): # Iterate remaining
                      if remaining_demand <= FLOAT_TOLERANCE_UTILS: continue

                      cust_loc = customer_locations[cust_idx]
                      round_trip_dist = 2 * haversine(outlet_coord, cust_loc)

                      # Check range and if drone can carry *anything* useful
                      if round_trip_dist <= drone_max_dist + FLOAT_TOLERANCE_UTILS:
                          load_to_take = min(remaining_demand, drone_payload)

                          if load_to_take > FLOAT_TOLERANCE_UTILS:
                              # Create a 1-to-1 drone trip
                              trip_time = round_trip_dist / drone_speed_kmph if drone_speed_kmph > FLOAT_TOLERANCE_UTILS else float('inf')
                              trip_cost = round_trip_dist * drone_cost_per_km

                              generated_trips.append({
                                  'type': 'drone',
                                  'route': [cust_idx],
                                  'load': load_to_take,
                                  'cost': trip_cost,
                                  'time': trip_time
                              })

                              demands_remaining_global[cust_idx] -= load_to_take
                              customers_served_in_this_pass += 1
                              customers_served_by_drone_this_pass += 1 # Track drone progress specifically
                              new_remaining = demands_remaining_global[cust_idx]
                              if new_remaining < FLOAT_TOLERANCE_UTILS:
                                   demands_remaining_global[cust_idx] = 0.0 # Clamp
                                   customers_to_remove_from_list_drone.append(cust_idx)


                 # Remove fully served customers from the list after drone pass
                 if customers_to_remove_from_list_drone:
                     customers_to_serve = [(idx, dem) for idx, dem in customers_to_serve if idx not in customers_to_remove_from_list_drone]
                     # Update demands in the main list
                     customers_to_serve = [(idx, demands_remaining_global.get(idx, 0.0)) for idx, _ in customers_to_serve]


            # If no customer demand was served in a full pass (vehicle + drone attempts), break the loop
            if customers_served_in_this_pass == 0:
                 break # Avoid infinite loop if remaining demand cannot be served

    except Exception as e:
         warnings.warn(f"Error during Stage 2 trip generation for outlet {outlet_index}: {e}")
         traceback.print_exc()
         # Return potentially partial trips generated so far, but signal error state externally

    return generated_trips


# --- Initial Solution Generation ---
# (Definition provided previously, seems okay)
def create_initial_solution_mdsd(problem_data: dict, vehicle_params: dict, drone_params: dict,
                                 unmet_demand_penalty: float, cost_weight: float = 1.0,
                                 time_weight: float = 0.0) -> SolutionCandidate | None:
    """
    Generates an initial solution candidate for the Multi-Depot, Two-Echelon VRP
    with a greedy heuristic approach for assignments and Stage 1 routing.

    Args:
        problem_data (dict): Dictionary containing 'locations' and 'demands'.
        vehicle_params (dict): Dictionary of vehicle parameters.
        drone_params (dict): Dictionary of drone parameters.
        unmet_demand_penalty (float): The penalty cost per unit of unmet demand.
        cost_weight (float): Weight for raw cost in the objective. Defaults to 1.0.
        time_weight (float): Weight for time/makespan in the objective. Defaults to 0.0.

    Returns:
        SolutionCandidate | None: The initial feasible or unfeasible solution candidate,
                                   or None if initialization fails.
    """
    print("Generating initial solution using greedy heuristic...")
    start_time = time.time()

    try:
        if not problem_data or 'locations' not in problem_data or 'demands' not in problem_data:
            raise ValueError("Invalid 'problem_data' structure.")

        locations = problem_data.get('locations', {})
        demands = problem_data.get('demands', [])
        depot_locations = locations.get('logistics_centers', [])
        outlet_locations = locations.get('sales_outlets', [])
        customer_locations = locations.get('customers', [])

        num_depots = len(depot_locations)
        num_outlets = len(outlet_locations)
        num_customers = len(customer_locations)

        if num_depots == 0 or num_outlets == 0: # Allow num_customers = 0
            warnings.warn("Cannot create meaningful initial solution: requires at least one depot and one outlet.")
            # Return an invalid solution candidate
            return SolutionCandidate(problem_data={}, vehicle_params={}, drone_params={}, unmet_demand_penalty=float('inf'), initialization_error="Zero depots or outlets")


        # --- 1. Assign Sales Outlets to Nearest Depot ---
        outlet_to_depot_assignments = {}
        depot_to_outlets_assigned = {depot_idx: [] for depot_idx in range(num_depots)}
        for outlet_idx in range(num_outlets):
            outlet_coord = outlet_locations[outlet_idx]
            dists = [(depot_idx, haversine(outlet_coord, depot_coord)) for depot_idx, depot_coord in enumerate(depot_locations)]
            if not dists: raise ValueError("No depots found to assign outlets.")
            nearest_depot_idx, _ = min(dists, key=lambda item: item[1])
            outlet_to_depot_assignments[outlet_idx] = nearest_depot_idx
            depot_to_outlets_assigned[nearest_depot_idx].append(outlet_idx)

        # --- 2. Assign Customers to Nearest Sales Outlet ---
        customer_to_outlet_assignments = {}
        outlet_to_customers_assigned = {outlet_idx: [] for outlet_idx in range(num_outlets)}
        for cust_idx in range(num_customers):
            cust_coord = customer_locations[cust_idx]
            dists = [(outlet_idx, haversine(cust_coord, outlet_coord)) for outlet_idx, outlet_coord in enumerate(outlet_locations)]
            if not dists: raise ValueError("No outlets found to assign customers.")
            nearest_outlet_idx, _ = min(dists, key=lambda item: item[1])
            customer_to_outlet_assignments[cust_idx] = nearest_outlet_idx
            outlet_to_customers_assigned[nearest_outlet_idx].append(cust_idx)

        # --- 3. Create Stage 1 Routes for Each Depot using Nearest Neighbor ---
        stage1_routes = {}
        for depot_idx in range(num_depots):
            assigned_outlets_indices = depot_to_outlets_assigned.get(depot_idx, [])
            if not assigned_outlets_indices:
                stage1_routes[depot_idx] = []
                continue

            depot_coord = depot_locations[depot_idx]
            current_route = []
            unvisited_outlets = list(assigned_outlets_indices)
            current_location_coord = depot_coord

            while unvisited_outlets:
                dists_to_unvisited = [(outlet_idx, haversine(current_location_coord, outlet_locations[outlet_idx])) for outlet_idx in unvisited_outlets]
                if not dists_to_unvisited: break # Should not happen if unvisited_outlets is not empty

                nearest_outlet_idx, _ = min(dists_to_unvisited, key=lambda item: item[1])

                current_route.append(nearest_outlet_idx)
                unvisited_outlets.remove(nearest_outlet_idx)
                current_location_coord = outlet_locations[nearest_outlet_idx]

            stage1_routes[depot_idx] = current_route

        # --- 4. Create SolutionCandidate Object ---
        initial_solution = SolutionCandidate(
            problem_data=problem_data,
            vehicle_params=vehicle_params,
            drone_params=drone_params,
            unmet_demand_penalty=unmet_demand_penalty,
            cost_weight=cost_weight,
            time_weight=time_weight,
            initial_stage1_routes=stage1_routes,
            initial_outlet_to_depot_assignments=outlet_to_depot_assignments,
            initial_customer_to_outlet_assignments=customer_to_outlet_assignments
        )

        # --- 5. Evaluate the Initial Solution ---
        initial_solution.evaluate(
            distance_func=haversine,
            stage2_trip_generator_func=create_heuristic_trips_split_delivery
        )

        end_time = time.time()
        print(f"Initial solution generated in {end_time - start_time:.4f} seconds. Feasible: {initial_solution.is_feasible}, WCost: {format_float(initial_solution.weighted_cost, 4)}")
        return initial_solution

    except Exception as e:
        warnings.warn(f"An unexpected error occurred during initial solution generation: {e}")
        traceback.print_exc()
        # Return an invalid state solution candidate
        error_solution = SolutionCandidate(problem_data={}, vehicle_params={}, drone_params={}, unmet_demand_penalty=float('inf'), initialization_error=str(e))
        return error_solution


# --- Neighbor Solution Generation (for Local Search/Mutation) ---
# (Definition provided previously, seems okay)
def generate_neighbor_solution_mdsd(current_solution: SolutionCandidate) -> SolutionCandidate | None:
    """
    Generates a neighbor solution by applying a perturbation operator
    to one of the Stage 1 routes of the current solution.

    Args:
        current_solution (SolutionCandidate): The base solution candidate.

    Returns:
        SolutionCandidate | None: A new neighbor solution candidate (unevaluated),
                                   or None if generation fails.
    """
    if not isinstance(current_solution, SolutionCandidate) or current_solution.initialization_error:
        warnings.warn("Cannot generate neighbor: Invalid or errored current_solution.")
        return None

    try:
        neighbor_solution = copy.deepcopy(current_solution)
    except Exception as e:
        warnings.warn(f"Error deep copying solution for neighbor generation: {e}")
        traceback.print_exc()
        return None

    if not neighbor_solution.stage1_routes:
         warnings.warn("Cannot generate neighbor: No Stage 1 routes found in solution.")
         return None # Return original copy if no routes? Or None? Returning None seems safer.

    depot_indices = [idx for idx, route in neighbor_solution.stage1_routes.items() if route] # Only consider depots with non-empty routes

    if not depot_indices:
         # warnings.warn("Cannot generate neighbor: All depots have empty Stage 1 routes.")
         # Return the unmodified copy if no routes can be perturbed
         return neighbor_solution

    selected_depot_index = random.choice(depot_indices)
    original_route = neighbor_solution.stage1_routes[selected_depot_index] # Already checked it's non-empty

    mutation_operators = [swap_mutation, scramble_mutation, inversion_mutation]
    selected_operator = random.choice(mutation_operators)

    try:
        perturbed_route = selected_operator(original_route)
        neighbor_solution.stage1_routes[selected_depot_index] = perturbed_route
    except Exception as e:
        warnings.warn(f"Error applying mutation operator {selected_operator.__name__} to depot {selected_depot_index} route: {e}")
        traceback.print_exc()
        return None # Return None if perturbation fails

    # Reset evaluation results for the neighbor
    neighbor_solution.is_feasible = False
    neighbor_solution.weighted_cost = float('inf')
    neighbor_solution.evaluated_cost = float('inf')
    neighbor_solution.evaluated_time = float('inf')
    neighbor_solution.evaluated_unmet = float('inf')
    neighbor_solution.served_customer_details = {}
    neighbor_solution.evaluation_stage1_error = False
    neighbor_solution.evaluation_stage2_error = False
    neighbor_solution.stage2_trips = {}

    return neighbor_solution


# --- Permutation Mutation Operators ---
# (Definitions provided previously, seem okay)
def swap_mutation(route: list) -> list:
    """Applies swap mutation to a list (route)."""
    route_copy = list(route)
    size = len(route_copy)
    if size < 2: return route_copy
    idx1, idx2 = random.sample(range(size), 2)
    route_copy[idx1], route_copy[idx2] = route_copy[idx2], route_copy[idx1]
    return route_copy

def scramble_mutation(route: list) -> list:
    """Applies scramble mutation to a list (route)."""
    route_copy = list(route)
    size = len(route_copy)
    if size < 2: return route_copy
    if size == 2:
         start_idx, end_idx = 0, 1
    else:
         start_idx = random.randint(0, size - 2)
         end_idx = random.randint(start_idx + 1, size - 1)
    sublist = route_copy[start_idx : end_idx + 1]
    random.shuffle(sublist)
    route_copy[start_idx : end_idx + 1] = sublist
    return route_copy

def inversion_mutation(route: list) -> list:
    """Applies inversion mutation to a list (route)."""
    route_copy = list(route)
    size = len(route_copy)
    if size < 2: return route_copy
    if size == 2:
         start_idx, end_idx = 0, 1
    else:
         start_idx = random.randint(0, size - 2)
         end_idx = random.randint(start_idx + 1, size - 1)
    route_copy[start_idx : end_idx + 1] = route_copy[start_idx : end_idx + 1][::-1]
    return route_copy


# --- Helper for formatting float values ---
# (Definition provided previously, seems okay)
def format_float(value, precision=4):
    """
    Safely formats a numerical value to a specified precision string.
    Handles None, NaN, Infinity, and non-numeric types gracefully.
    """
    if isinstance(value, (int, float)):
        if math.isnan(value): return "NaN"
        if value == float('inf'): return "Infinity"
        if value == float('-inf'): return "-Infinity"
        try: return f"{value:.{precision}f}"
        except: return str(value)
    elif value is None: return "N/A"
    else:
        try: return str(value)
        except: return "Invalid Value"


# --- Optional Main Execution Block for Standalone Testing ---
if __name__ == '__main__':
    """
    Standalone execution block for testing problem utility functions.
    Requires some dummy problem data.
    """
    print("Running core/problem_utils.py in standalone test mode.")

    # --- Create Dummy Problem Data ---
    try:
        print("\n--- Creating Dummy Problem Data ---")
        dummy_locations = {
            'logistics_centers': [(34.0, -118.0), (34.1, -118.2)],
            'sales_outlets': [(34.05, -118.1), (34.02, -118.05), (34.15, -118.3), (33.95, -118.15)],
            'customers': [(34.06, -118.11), (34.05, -118.09), (34.00, -118.06), (34.16, -118.31), (34.14, -118.28), (33.96, -118.16)]
        }
        dummy_demands = [10.0, 15.0, 8.0, 20.0, 12.0, 5.0]
        dummy_problem_data = {'locations': dummy_locations, 'demands': dummy_demands}
        dummy_vehicle_params = {'payload': 200.0, 'cost_per_km': 1.0, 'speed_kmph': 50.0}
        dummy_drone_params = {'payload': 30.0, 'max_flight_distance_km': 10.0, 'cost_per_km': 0.5, 'speed_kmph': 80.0}
        dummy_unmet_penalty = 1000.0
        dummy_cost_weight = 1.0
        dummy_time_weight = 0.1
        print(f"Dummy data created: {len(dummy_locations['logistics_centers'])} depots, {len(dummy_locations['sales_outlets'])} outlets, {len(dummy_locations['customers'])} customers.")

    except Exception as e:
        print(f"Error creating dummy problem data: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Test Initial Solution Generation ---
    print("\n--- Testing Initial Solution Generation ---")
    initial_solution = create_initial_solution_mdsd(
        problem_data=dummy_problem_data,
        vehicle_params=dummy_vehicle_params,
        drone_params=dummy_drone_params,
        unmet_demand_penalty=dummy_unmet_penalty,
        cost_weight=dummy_cost_weight,
        time_weight=dummy_time_weight
    )

    if initial_solution and not initial_solution.initialization_error:
        print("\nInitial Solution Details:")
        print(f"  Representative String: {initial_solution}") # Uses __repr__
        print(f"  Feasible: {initial_solution.is_feasible}")
        print(f"  Weighted Cost: {format_float(initial_solution.weighted_cost, 4)}")
        print(f"  Raw Cost: {format_float(initial_solution.evaluated_cost, 2)}")
        print(f"  Time (Makespan): {format_float(initial_solution.evaluated_time, 2)}")
        print(f"  Unmet Demand: {format_float(initial_solution.evaluated_unmet, 2)}")
        print("  Outlet-to-Depot Assignments:", initial_solution.outlet_to_depot_assignments)
        print("  Customer-to-Outlet Assignments:", initial_solution.customer_to_outlet_assignments)
        print("  Stage 1 Routes per Depot:", initial_solution.stage1_routes)
        # print("  Stage 2 Trips per Outlet:", initial_solution.stage2_trips) # Can be verbose
    else:
        print("\nInitial solution generation failed or resulted in an error state.")
        if initial_solution: print(f"  Error: {initial_solution.initialization_error}")


    # --- Test Neighbor Solution Generation (if initial solution was successful) ---
    if initial_solution and not initial_solution.initialization_error:
        print("\n--- Testing Neighbor Solution Generation ---")
        num_neighbors_to_generate = 3
        print(f"Generating {num_neighbors_to_generate} neighbor solutions...")

        for i in range(num_neighbors_to_generate):
            neighbor = generate_neighbor_solution_mdsd(initial_solution)
            if neighbor and not neighbor.initialization_error:
                 print(f"  Neighbor {i+1}: Generated. Stage 1 Routes (Perturbed):", neighbor.stage1_routes)
                 # Optionally evaluate the neighbor here for testing
                 # neighbor.evaluate(haversine, create_heuristic_trips_split_delivery)
                 # print(f"    Evaluated Neighbor {i+1}: {neighbor}")
            else:
                 print(f"  Neighbor {i+1}: Generation failed.")

    # --- Test Mutation Operators ---
    print("\n--- Testing Mutation Operators ---")
    test_route = [0, 1, 2, 3, 4, 5]
    print(f"Original Route: {test_route}")
    print(f"Swap Mutation:  {swap_mutation(test_route)}")
    print(f"Scramble Mut: {scramble_mutation(test_route)}")
    print(f"Inversion Mut:{inversion_mutation(test_route)}")


    print("\nStandalone test finished.")