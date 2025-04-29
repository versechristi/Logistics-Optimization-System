# algorithm/greedy_heuristic.py
# -*- coding: utf-8 -*-\
"""
Greedy Heuristic Algorithm adapted for Multi-Depot, Two-Echelon VRP
with Drones and Split Deliveries (MD-2E-VRPSD).

Constructs a solution sequentially using simple greedy rules for assignments
 and Stage 1 routing, then utilizes the Stage 2 heuristic for trip generation.

- Stage 1: Assigns sales outlets to their nearest logistics center (depot).
           Assigns customers to their nearest sales outlet.
           For each depot, creates a Stage 1 route serving its assigned outlets
           using a Nearest Neighbor heuristic starting from the depot.
- Stage 2: Iteratively creates trips from each outlet using the provided
           adapted heuristic (`create_heuristic_trips_split_delivery`) that
           handles capacity constraints and allows split deliveries, operating
           on the remaining customer demands.

This revised version ensures the use of the updated Stage 2 heuristic for
trip generation during construction and evaluates the final constructed solution
using the updated core cost function to report accurate cost, time, and unmet demand,
including feasibility status. It builds the multi-depot structure explicitly.
"""

import random # Not strictly needed for pure greedy construction, but included if initial assignment/NN uses randomness
import copy
import time
import math
import traceback
import sys
import os
import warnings # Use warnings for non-critical issues

# --- Path Setup & Safe Imports ---
# Attempt to ensure the project root is in sys.path for robust imports
try:
    # Assumes this file is in project_root/algorithm
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_greedy = os.path.dirname(current_dir)
    if project_root_greedy not in sys.path:
        sys.path.insert(0, project_root_greedy)
        # print(f"Greedy: Added project root to sys.path: {project_root_greedy}") # Optional debug print

    # Import necessary components from core modules
    # Need SolutionCandidate, the Stage 2 heuristic, distance function, and cost function
    from core.problem_utils import (
        SolutionCandidate, # To represent the final solution structure
        create_heuristic_trips_split_delivery, # The heuristic used for Stage 2
        # We don't need create_initial_solution_mdsd here, as the greedy algorithm
        # performs its own initial construction logic from scratch.
        # We might need permutation mutations if the greedy uses any internal perturbation
        # for construction variants, but typical greedy doesn't.
    )
    # Need distance calculation
    from core.distance_calculator import haversine # For distance calculations in assignments/NN
    # Need the cost function for final evaluation
    from core.cost_function import calculate_total_cost_and_evaluate

except ImportError as e:
    print(f"CRITICAL ERROR in algorithm.greedy_heuristic: Failed during initial import block: {e}")
    traceback.print_exc()
    # Define dummy functions/classes if imports fail to prevent immediate crash
    # but indicate severe error. Optimization cannot run without these.
    class SolutionCandidate:
        def __init__(self, *args, **kwargs): pass
        # Dummy attributes needed if dummy is ever returned
        weighted_cost = float('inf')
        evaluated_cost = float('inf')
        evaluated_time = float('inf')
        evaluated_unmet = float('inf')
        is_feasible = False
        evaluation_stage1_error = True
        evaluation_stage2_error = True
        stage1_routes = {}
        stage2_trips = {}
        served_customer_details = {}
        algorithm_params = {} # Dummy for result structure
        total_computation_time = 0.0
        algorithm_name = 'greedy_heuristic'
        run_error = "Import Error"


    def create_heuristic_trips_split_delivery(*args, **kwargs):
        print("DUMMY create_heuristic_trips_split_delivery called in Greedy due to import error!")
        return [] # Dummy empty trips
    def haversine(*args):
        print("DUMMY haversine called in Greedy due to import error!")
        return float('inf') # Dummy infinite distance
    def calculate_total_cost_and_evaluate(*args, **kwargs):
        print("DUMMY calculate_total_cost_and_evaluate called in Greedy due to import error!")
        # Return tuple format with infinite costs and error flags
        return float('inf'), float('inf'), float('inf'), {}, True, True, {}

    print("Greedy Heuristic will use dummy functions/classes due to critical import failure.")


except Exception as e:
    print(f"An unexpected error occurred during Greedy import block: {e}")
    traceback.print_exc()
    # Define a dummy run function that indicates error
    def run_greedy_heuristic(*args, **kwargs):
        print("CRITICAL ERROR: Greedy Heuristic initialization failed.")
        return {'run_error': f"Initialization failed: {e}"}
    print("Greedy Heuristic will not run due to unexpected import error.")


# Define a small tolerance for floating-point comparisons
FLOAT_TOLERANCE_GREEDY = 1e-6


def run_greedy_heuristic(problem_data: dict, vehicle_params: dict, drone_params: dict,
                         unmet_demand_penalty: float, cost_weight: float, time_weight: float,
                         initial_solution_candidate: SolutionCandidate | None = None, # Greedy doesn't typically use initial_solution_candidate, but include signature compatibility
                         algo_specific_params: dict | None = None # Greedy has few specific params, but include signature compatibility
                        ) -> dict | None:
    """
    Runs the Greedy Heuristic Algorithm to construct a solution for the
    Multi-Depot, Two-Echelon VRP with Drones and Split Deliveries.

    Constructs the solution based on greedy assignment and Stage 1 routing
    heuristics, then utilizes the dedicated Stage 2 trip generation heuristic.

    Args:
        problem_data (dict): Dictionary containing problem instance data.
        vehicle_params (dict): Dictionary of vehicle parameters.
        drone_params (dict): Dictionary of drone parameters.
        unmet_demand_penalty (float): Penalty cost per unit of unmet demand.
        cost_weight (float): Weight for raw cost in objective.
        time_weight (float): Weight for time/makespan in objective.
        initial_solution_candidate (SolutionCandidate | None): Ignored by this algorithm,
                                                               kept for signature compatibility.
        algo_specific_params (dict | None): Ignored by this algorithm, kept for
                                            signature compatibility.

    Returns:
        dict | None: A dictionary containing the constructed solution's evaluation
                     results and the algorithm's run time. Returns None or error dict
                     if the algorithm fails to construct or evaluate a solution.
        Example: {
            'best_solution': SolutionCandidate_object, # Or just the structure components
            'weighted_cost': float, 'evaluated_cost': float, 'evaluated_time': float,
            'evaluated_unmet': float, 'is_feasible': bool,
            'evaluation_stage1_error': bool, 'evaluation_stage2_error': bool,
            'stage1_routes': dict, 'stage2_trips': dict, 'served_customer_details': dict,
            'cost_history': list, # Greedy has no iteration history, usually a list with one entry (final cost)
            'total_computation_time': float,
            'algorithm_name': 'greedy_heuristic',
            'algorithm_params': dict # Parameters effectively used for construction/evaluation
        }
    """
    print("\n--- Starting Greedy Heuristic (MD-SD) ---")
    start_time_greedy = time.time()

    # --- Input Validation and Data Extraction ---
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

        if num_depots == 0 or num_outlets == 0 or num_customers == 0:
             warnings.warn("Greedy Warning: Problem data contains zero depots, outlets, or customers. Solution will be trivial/impossible.")
             # Return an empty/invalid result dictionary
             return {
                'weighted_cost': float('inf'), 'evaluated_cost': float('inf'), 'evaluated_time': float('inf'),
                'evaluated_unmet': float('inf'), 'is_feasible': False,
                'evaluation_stage1_error': True, 'evaluation_stage2_error': True,
                'stage1_routes': {}, 'stage2_trips': {}, 'served_customer_details': {},
                'cost_history': [float('inf')], 'total_computation_time': 0.0,
                'algorithm_name': 'greedy_heuristic', 'algorithm_params': {},
                'run_error': "Zero depots, outlets, or customers in problem data."
             }


        # Basic parameter checks (weights/penalty are used in final evaluation)
        if not isinstance(unmet_demand_penalty, (int, float)) or unmet_demand_penalty < 0:
            warnings.warn(f"Greedy Warning: Invalid unmet_demand_penalty ({unmet_demand_penalty}). Using 0.0 for evaluation.")
            unmet_demand_penalty = 0.0

        if not isinstance(cost_weight, (int, float)) or not isinstance(time_weight, (int, float)):
             warnings.warn("Greedy Warning: Cost or time weight is not numeric. Using defaults (1.0, 0.0) for evaluation.")
             cost_weight = 1.0
             time_weight = 0.0

        # No specific parameters for this simple greedy heuristic, but check if any were passed
        if algo_specific_params:
             warnings.warn(f"Greedy Heuristic received unexpected algorithm-specific parameters: {algo_specific_params}. These will be ignored.")


    except Exception as e:
        print(f"Error during Greedy Heuristic setup or validation: {e}")
        traceback.print_exc()
        return {'run_error': f"Setup or validation failed: {e}"}


    # --- Greedy Construction ---

    # 1. Assign Sales Outlets to Nearest Depot
    outlet_to_depot_assignments = {} # Key: outlet index, Value: depot index
    depot_to_outlets_assigned = {depot_idx: [] for depot_idx in range(num_depots)} # Key: depot index, Value: list of assigned outlet indices

    print("Greedy Construction: Assigning sales outlets to nearest depots...")
    try:
        for outlet_idx in range(num_outlets):
            outlet_coord = outlet_locations[outlet_idx]
            nearest_depot_idx = -1
            min_dist = float('inf')

            for depot_idx in range(num_depots):
                depot_coord = depot_locations[depot_idx]
                dist = haversine(outlet_coord, depot_coord)
                if dist < min_dist:
                    min_dist = dist
                    nearest_depot_idx = depot_idx

            if nearest_depot_idx != -1:
                outlet_to_depot_assignments[outlet_idx] = nearest_depot_idx
                depot_to_outlets_assigned[nearest_depot_idx].append(outlet_idx)
            else:
                 warnings.warn(f"Greedy Warning: Outlet {outlet_idx} could not be assigned to any depot (no depots found?).")

    except Exception as e:
        print(f"Error during greedy outlet-to-depot assignment: {e}")
        traceback.print_exc()
        # Continue, but assignments might be incomplete/wrong
        outlet_to_depot_assignments = {}
        depot_to_outlets_assigned = {depot_idx: [] for depot_idx in range(num_depots)}


    # 2. Assign Customers to Nearest Sales Outlet
    customer_to_outlet_assignments = {} # Key: customer index, Value: outlet index
    outlet_to_customers_assigned = {outlet_idx: [] for outlet_idx in range(num_outlets)} # Key: outlet index, Value: list of assigned customer indices

    print("Greedy Construction: Assigning customers to nearest sales outlets...")
    try:
        for cust_idx in range(num_customers):
            cust_coord = customer_locations[cust_idx]
            nearest_outlet_idx = -1
            min_dist = float('inf')

            # Customers are assigned to ANY sales outlet in the greedy approach
            for outlet_idx in range(num_outlets):
                 outlet_coord = outlet_locations[outlet_idx]
                 dist = haversine(cust_coord, outlet_coord)
                 if dist < min_dist:
                      min_dist = dist
                      nearest_outlet_idx = outlet_idx

            if nearest_outlet_idx != -1:
                customer_to_outlet_assignments[cust_idx] = nearest_outlet_idx
                outlet_to_customers_assigned[nearest_outlet_idx].append(cust_idx)
            else:
                 warnings.warn(f"Greedy Warning: Customer {cust_idx} could not be assigned to any outlet (no outlets found?).")

    except Exception as e:
        print(f"Error during greedy customer-to-outlet assignment: {e}")
        traceback.print_exc()
        # Continue, but assignments might be incomplete/wrong
        customer_to_outlet_assignments = {}
        outlet_to_customers_assigned = {outlet_idx: [] for outlet_idx in range(num_outlets)}


    # 3. Create Stage 1 Routes for Each Depot using Nearest Neighbor
    stage1_routes = {} # Key: depot index, Value: list of assigned outlet indices in visit order

    print("Greedy Construction: Creating Stage 1 routes using Nearest Neighbor...")
    try:
        for depot_idx in range(num_depots):
            assigned_outlets_indices = depot_to_outlets_assigned.get(depot_idx, [])
            num_assigned_outlets = len(assigned_outlets_indices)

            if num_assigned_outlets == 0:
                stage1_routes[depot_idx] = [] # No route needed if no outlets assigned
                continue

            depot_coord = depot_locations[depot_idx]
            current_route = []
            unvisited_outlets = list(assigned_outlets_indices) # Copy list
            current_location_coord = depot_coord

            # Start Nearest Neighbor from the depot
            while unvisited_outlets:
                nearest_outlet_idx = -1
                min_dist = float('inf')
                # Find nearest unvisited outlet from the current location
                for outlet_idx in unvisited_outlets:
                    outlet_coord = outlet_locations[outlet_idx]
                    dist = haversine(current_location_coord, outlet_coord)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_outlet_idx = outlet_idx

                if nearest_outlet_idx != -1:
                    current_route.append(nearest_outlet_idx)
                    unvisited_outlets.remove(nearest_outlet_idx)
                    current_location_coord = outlet_locations[nearest_outlet_idx] # Move to the visited outlet's location
                else:
                    # Should not happen in a connected graph with finite distances if unvisited_outlets is not empty
                    warnings.warn(f"Greedy Warning: Nearest Neighbor failed to find next outlet for Depot {depot_idx}. Route incomplete.")
                    break # Exit loop if stuck

            stage1_routes[depot_idx] = current_route

    except Exception as e:
        print(f"Error during greedy Stage 1 route generation: {e}")
        traceback.print_exc()
        # Continue, but routes might be incomplete/wrong
        stage1_routes = {depot_idx: [] for depot_idx in range(num_depots)}


    print("Greedy Construction complete.")


    # --- Final Evaluation ---
    # Use the core cost function to evaluate the constructed solution
    # This function will internally call the Stage 2 trip generator for each outlet.
    print("Evaluating the constructed greedy solution...")

    # Prepare the solution structure dictionary for evaluation
    # The calculate_total_cost_and_evaluate function expects stage1_routes,
    # outlet_to_depot_assignments, and customer_to_outlet_assignments
    greedy_solution_structure = {
        'stage1_routes': stage1_routes,
        'outlet_to_depot_assignments': outlet_to_depot_assignments,
        'customer_to_outlet_assignments': customer_to_outlet_assignments,
        # Note: Stage 2 trips are generated during evaluation, not constructed here
    }

    evaluation_stage1_error = False
    evaluation_stage2_error = False
    total_raw_cost = float('inf')
    total_time_makespan = float('inf')
    final_unmet_demand = float('inf')
    served_customer_details = {}
    stage2_trips_details_aggregated = {}


    try:
        # Call the core evaluation function
        (total_raw_cost, total_time_makespan, final_unmet_demand,
         served_customer_details, evaluation_stage1_error, evaluation_stage2_error,
         stage2_trips_details_aggregated) = calculate_total_cost_and_evaluate(
             stage1_routes=greedy_solution_structure['stage1_routes'],
             outlet_to_depot_assignments=greedy_solution_structure['outlet_to_depot_assignments'],
             customer_to_outlet_assignments=greedy_solution_structure['customer_to_outlet_assignments'],
             problem_data=problem_data,
             vehicle_params=vehicle_params,
             drone_params=drone_params,
             distance_func=haversine, # Pass the real distance function
             stage2_trip_generator_func=create_heuristic_trips_split_delivery, # Pass the real Stage 2 heuristic
             unmet_demand_penalty=unmet_demand_penalty,
             cost_weight=cost_weight,
             time_weight=time_weight
         )

        # Calculate weighted cost based on evaluation results
        # Handle potential infinity values
        safe_raw_cost = total_raw_cost if not math.isinf(total_raw_cost) and not math.isnan(total_raw_cost) else float('inf')
        safe_time = total_time_makespan if not math.isinf(total_time_makespan) and not math.isnan(total_time_makespan) else float('inf')
        safe_unmet = final_unmet_demand if not math.isinf(final_unmet_demand) and not math.isnan(final_unmet_demand) else float('inf')

        # Use safe weights/penalty as well
        safe_cost_weight = cost_weight if isinstance(cost_weight, (int, float)) and not math.isinf(cost_weight) and not math.isnan(cost_weight) else 0.0
        safe_time_weight = time_weight if isinstance(time_weight, (int, float)) and not math.isinf(time_weight) and not math.isnan(time_weight) else 0.0
        safe_unmet_penalty = unmet_demand_penalty if isinstance(unmet_demand_penalty, (int, float)) and not math.isinf(unmet_demand_penalty) and not math.isnan(unmet_demand_penalty) else 0.0

        weighted_cost = (safe_cost_weight * safe_raw_cost +
                         safe_time_weight * safe_time +
                         safe_unmet_penalty * safe_unmet)

        # If evaluation had errors in either stage, the weighted cost should be infinite
        if evaluation_stage1_error or evaluation_stage2_error:
            weighted_cost = float('inf')


        # Determine feasibility based on evaluation results
        is_feasible = (not evaluation_stage1_error and
                       not evaluation_stage2_error and
                       final_unmet_demand is not None and
                       not math.isinf(final_unmet_demand) and
                       not math.isnan(final_unmet_demand) and
                       final_unmet_demand < FLOAT_TOLERANCE_GREEDY) # Check if close to zero


    except Exception as e:
        print(f"An unexpected error occurred during final evaluation of greedy solution: {e}")
        traceback.print_exc()
        evaluation_stage1_error = True # Assume evaluation failed overall
        evaluation_stage2_error = True
        weighted_cost = float('inf')
        total_raw_cost = float('inf')
        total_time_makespan = float('inf')
        final_unmet_demand = float('inf')
        is_feasible = False
        served_customer_details = {}
        stage2_trips_details_aggregated = {}
        run_error_message = f"Final evaluation failed: {e}"


    # --- Prepare Result Dictionary ---
    end_time_greedy = time.time()
    total_time_greedy = end_time_greedy - start_time_greedy
    print(f"\nGreedy Heuristic (MD-SD) finished in {total_time_greedy:.4f} seconds.")


    # The greedy heuristic doesn't have an iterative history like metaheuristics.
    # The cost history is just the final evaluated cost.
    cost_history = [weighted_cost]

    # Prepare the result dictionary
    greedy_results = {
        # Include the constructed solution structure for visualization/reporting
        'stage1_routes': greedy_solution_structure['stage1_routes'],
        # Assignments are part of the problem structure, but can be included for clarity
        'outlet_to_depot_assignments': greedy_solution_structure['outlet_to_depot_assignments'],
        'customer_to_outlet_assignments': greedy_solution_structure['customer_to_outlet_assignments'],
        # Include Stage 2 trip details aggregated from evaluation
        'stage2_trips': stage2_trips_details_aggregated,
        # Include served customer details from evaluation
        'served_customer_details': served_customer_details,

        # Include evaluation results
        'weighted_cost': weighted_cost,
        'evaluated_cost': total_raw_cost,
        'evaluated_time': total_time_makespan,
        'evaluated_unmet': final_unmet_demand,
        'is_feasible': is_feasible,
        'evaluation_stage1_error': evaluation_stage1_error,
        'evaluation_stage2_error': evaluation_stage2_error,

        # Include history (single entry for greedy)
        'cost_history': cost_history,

        # Add algorithm metadata
        'total_computation_time': total_time_greedy,
        'algorithm_name': 'greedy_heuristic',
        'algorithm_params': { # Store parameters effectively used for construction/evaluation
            'cost_weight': cost_weight,
            'time_weight': time_weight,
            'unmet_demand_penalty': unmet_demand_penalty,
            # Add any specific greedy parameters if they existed (e.g., nearest neighbor variant)
        },
        # Add run error message if evaluation failed
        'run_error': locals().get('run_error_message', None) # Get error message if defined in except block

        # Note: 'best_solution' key is used by iterative algorithms to return SolutionCandidate object.
        # For Greedy, the structure components are returned directly in the dict.
        # A SolutionCandidate could be created here from the structure if consistent
        # object return is strictly required by route_optimizer. Let's create one for consistency.
    }

    # Create a SolutionCandidate object from the constructed structure and evaluation results
    # This requires initializing it with problem data and parameters, then setting its evaluation results.
    try:
        greedy_solution_candidate = SolutionCandidate(
            problem_data=problem_data, # Pass the full problem data
            vehicle_params=vehicle_params,
            drone_params=drone_params,
            unmet_demand_penalty=unmet_demand_penalty,
            cost_weight=cost_weight,
            time_weight=time_weight,
            initial_stage1_routes=greedy_solution_structure['stage1_routes'],
            initial_outlet_to_depot_assignments=greedy_solution_structure['outlet_to_depot_assignments'],
            initial_customer_to_outlet_assignments=greedy_solution_structure['customer_to_outlet_assignments']
            # Note: Stage 2 trips are NOT passed here, they are set as evaluation results below
        )
        # Set the evaluation results directly to the candidate
        greedy_solution_candidate.evaluated_cost = total_raw_cost
        greedy_solution_candidate.evaluated_time = total_time_makespan
        greedy_solution_candidate.evaluated_unmet = final_unmet_demand
        greedy_solution_candidate.served_customer_details = served_customer_details
        greedy_solution_candidate.evaluation_stage1_error = evaluation_stage1_error
        greedy_solution_candidate.evaluation_stage2_error = evaluation_stage2_error
        greedy_solution_candidate.stage2_trips = stage2_trips_details_aggregated # Set generated S2 trips
        greedy_solution_candidate.weighted_cost = weighted_cost
        greedy_solution_candidate.is_feasible = is_feasible
        # Greedy doesn't have cost_history internal to SolutionCandidate
        setattr(greedy_solution_candidate, 'cost_history', cost_history) # Attach history as attribute for consistency


        greedy_results['best_solution'] = greedy_solution_candidate # Add the created candidate object

    except Exception as e:
        warnings.warn(f"Error creating SolutionCandidate object for greedy results: {e}")
        traceback.print_exc()
        # Continue with the dictionary result, but note the object creation failure
        greedy_results['solution_object_creation_error'] = str(e)
        greedy_results['best_solution'] = None # Explicitly set to None if object creation failed


    # Report final evaluated metrics to console
    print(f"Greedy Final Evaluation:")
    print(f"  Weighted Cost: {format_float(weighted_cost, 4)}")
    print(f"  Raw Cost: {format_float(total_raw_cost, 2)}")
    print(f"  Time (Makespan): {format_float(total_time_makespan, 2)}")
    print(f"  Unmet Demand: {format_float(final_unmet_demand, 2)}")
    print(f"  Feasible: {is_feasible}")

    if greedy_results.get('run_error'):
         print(f"  Run Error: {greedy_results['run_error']}")
    if greedy_results.get('solution_object_creation_error'):
         print(f"  Solution Object Creation Error: {greedy_results['solution_object_creation_error']}")


    return greedy_results


# --- Helper functions (Placeholder for format_float) ---
# Assuming format_float is available (e.g., defined in problem_utils or report_generator)
# Copying it here for standalone execution testing, but prefer central definition.
def format_float(value, precision=4):
    """
    Safely formats a numerical value to a specified precision string.
    Handles None, NaN, Infinity, and non-numeric types gracefully.
    """
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return "NaN"
        if value == float('inf'):
            return "Infinity"
        if value == float('-inf'):
            return "-Infinity"
        try:
            return f"{value:.{precision}f}"
        except Exception:
            return str(value)
    elif value is None:
        return "N/A"
    else:
        try:
            return str(value)
        except Exception:
            return "Invalid Value"

# --- Dummy haversine for standalone test if not imported ---
# Define a minimal dummy if the real one failed to import
try:
     # Check if the real function was imported successfully
     _ = haversine # This will raise NameError if not defined/imported
except NameError:
     print("Using DUMMY haversine for standalone Greedy test.")
     def haversine(coord1, coord2):
          # print("DUMMY haversine called in Greedy!") # Optional debug print
          # Return a small positive distance for dummy tests to proceed
          # For robustness, handle invalid coords
          if not coord1 or not coord2 or len(coord1) != 2 or len(coord2) != 2:
               return float('inf')
          # Simple Euclidean-like distance on coords for dummy (not geodetic)
          # This is just to allow the greedy logic (min distance) to work for testing.
          try:
               return math.sqrt((coord1[0] - coord2[0])**2 + (coord1[1] - coord2[1])**2) * 100 # Scale up dummy distance
          except Exception:
               return float('inf')


# --- Dummy create_heuristic_trips_split_delivery for standalone test if not imported ---
# Define a minimal dummy if the real one failed to import
try:
     # Check if the real function was imported successfully
     _ = create_heuristic_trips_split_delivery # This will raise NameError if not defined/imported
except NameError:
     print("Using DUMMY create_heuristic_trips_split_delivery for standalone Greedy test.")
     def create_heuristic_trips_split_delivery(outlet_idx, assigned_customer_indices, problem_data, vehicle_params, drone_params, demands_remaining_global):
          print(f"DUMMY create_heuristic_trips_split_delivery called in Greedy for outlet {outlet_idx}.")
          # This dummy generator simply marks a fraction of demand as met
          # and returns dummy trip info without accurate costs/times.
          # It *must* modify demands_remaining_global.

          generated_trips = []
          outlet_s2_cost = 0.0
          outlet_s2_time = 0.0

          # Need dummy haversine if the real one wasn't imported
          dummy_dist_func = haversine # Use the haversine available in this scope

          for cust_idx in assigned_customer_indices:
              if cust_idx in demands_remaining_global and demands_remaining_global[cust_idx] > FLOAT_TOLERANCE_GREEDY:
                  initial_d = problem_data.get('demands', [])[cust_idx] if cust_idx < len(problem_data.get('demands', [])) and problem_data.get('demands', [])[cust_idx] is not None else 0.0
                  remaining_d_before = demands_remaining_global[cust_idx]

                  # Simulate serving 75% of the remaining demand for this customer
                  served_amount = remaining_d_before * 0.75
                  demands_remaining_global[cust_idx] -= served_amount
                  if demands_remaining_global[cust_idx] < FLOAT_TOLERANCE_GREEDY: # Clamp to zero with tolerance
                       demands_remaining_global[cust_idx] = 0.0

                  # Create a dummy trip entry
                  if served_amount > FLOAT_TOLERANCE_GREEDY:
                       # Dummy cost/time based on distance (assuming a round trip with some dummy vehicle/drone)
                       outlet_locs = problem_data.get('locations', {}).get('sales_outlets', [])
                       cust_locs = problem_data.get('locations', {}).get('customers', [])
                       if outlet_idx < len(outlet_locs) and cust_idx < len(cust_locs) and dummy_dist_func:
                            # Use the dummy haversine available in this scope
                            dummy_dist = dummy_dist_func(outlet_locs[outlet_idx], cust_locs[cust_idx]) * 2 # Round trip
                            dummy_cost = dummy_dist * vehicle_params.get('cost_per_km', 1.0) # Use vehicle cost
                            dummy_time = dummy_dist / vehicle_params.get('speed_kmph', 50.0) # Use vehicle speed
                       else:
                            dummy_cost = 10.0 # Default dummy cost
                            dummy_time = 0.1 # Default dummy time

                       generated_trips.append({
                           'type': 'vehicle', # Dummy type
                           'route': [cust_idx],
                           'load': served_amount,
                           'cost': dummy_cost,
                           'time': dummy_time
                       })
                       outlet_s2_cost += dummy_cost
                       outlet_s2_time = max(outlet_s2_time, dummy_time) # Dummy makespan for this outlet


          return generated_trips


# --- Optional Main Execution Block for Standalone Testing ---
if __name__ == '__main__':
    """
    Standalone execution block for testing the Greedy Heuristic.
    Requires dummy problem data and uses the dummy Stage 2 generator
    and haversine (or real if imported).
    """
    print("Running algorithm/greedy_heuristic.py in standalone test mode.")

    # --- Create Dummy Problem Data ---
    # Needs to be sufficient for the greedy construction logic
    try:
        print("\n--- Creating Dummy Problem Data for Greedy Test ---")
        dummy_locations = {
             'logistics_centers': [
                 (34.0, -118.0), # Depot 0
                 (34.1, -118.2), # Depot 1
             ],
             'sales_outlets': [
                 (34.05, -118.1), # Outlet 0
                 (34.02, -118.05), # Outlet 1
                 (34.15, -118.3), # Outlet 2
                 (33.95, -118.15), # Outlet 3
                 (34.08, -117.95), # Outlet 4
                 (34.03, -118.25),  # Outlet 5
             ],
             'customers': [
                 (34.06, -118.11), # Customer 0
                 (34.05, -118.09), # Customer 1
                 (34.00, -118.06), # Customer 2
                 (34.16, -118.31), # Customer 3
                 (34.14, -118.28), # Customer 4
                 (33.96, -118.16), # Customer 5
                 (33.94, -118.14), # Customer 6
                 (34.09, -117.94), # Customer 7
                 (34.07, -117.96), # Customer 8
                 (34.04, -118.26)  # Customer 9
             ]
         }
        dummy_demands = [10.0, 15.0, 8.0, 20.0, 12.0, 5.0, 25.0, 18.0, 9.0, 22.0] # len = 10

        dummy_problem_data_greedy = {
            'locations': dummy_locations,
            'demands': dummy_demands
        }

        dummy_vehicle_params_greedy = {'payload': 200.0, 'cost_per_km': 1.5, 'speed_kmph': 60.0}
        dummy_drone_params_greedy = {'payload': 30.0, 'max_flight_distance_km': 15.0, 'cost_per_km': 0.8, 'speed_kmph': 100.0}
        dummy_unmet_penalty_greedy = 500.0
        dummy_cost_weight_greedy = 1.0
        dummy_time_weight_greedy = 0.1

        print("Dummy data and evaluation parameters created for Greedy test.")

    except Exception as e:
        print(f"Error creating dummy data for Greedy test: {e}")
        traceback.print_exc()
        sys.exit(1)


    # --- Run the Greedy Heuristic ---
    print("\n--- Running Greedy Heuristic (dummy data) ---")
    try:
        greedy_results = run_greedy_heuristic(
            problem_data=dummy_problem_data_greedy,
            vehicle_params=dummy_vehicle_params_greedy,
            drone_params=dummy_drone_params_greedy,
            unmet_demand_penalty=dummy_unmet_penalty_greedy,
            cost_weight=dummy_cost_weight_greedy,
            time_weight=dummy_time_weight_greedy
        )

        print("\n--- Greedy Heuristic Results Summary ---")
        if greedy_results:
             print(f"Algorithm Name: {greedy_results.get('algorithm_name')}")
             print(f"Run Time: {format_float(greedy_results.get('total_computation_time'), 4)} seconds")
             if greedy_results.get('run_error'):
                  print(f"Run Error: {greedy_results.get('run_error')}")
             elif greedy_results.get('solution_object_creation_error'):
                 print(f"Solution Object Creation Error: {greedy_results.get('solution_object_creation_error')}")
             else:
                  print("\nEvaluated Solution Metrics:")
                  print(f"  Weighted Cost: {format_float(greedy_results.get('weighted_cost', float('inf')), 4)}")
                  print(f"  Raw Cost: {format_float(greedy_results.get('evaluated_cost', float('inf')), 2)}")
                  print(f"  Time (Makespan): {format_float(greedy_results.get('evaluated_time', float('inf')), 2)}")
                  print(f"  Unmet Demand: {format_float(greedy_results.get('evaluated_unmet', float('inf')), 2)}")
                  print(f"  Feasible: {greedy_results.get('is_feasible', False)}")

                  # Print Stage 1 Routes
                  stage1_routes_found = greedy_results.get('stage1_routes', {})
                  print("\nConstructed Stage 1 Routes:")
                  if stage1_routes_found:
                      for depot_idx, route in sorted(stage1_routes_found.items()):
                           print(f"  Depot {depot_idx}: {route}")
                  else:
                      print("  No Stage 1 routes constructed.")

                  # Print summary of Stage 2 Trips (per outlet)
                  stage2_trips_found = greedy_results.get('stage2_trips', {})
                  print("\nGenerated Stage 2 Trips Summary (per Outlet):")
                  if stage2_trips_found:
                      for outlet_idx, trips in sorted(stage2_trips_found.items()):
                           if trips:
                               print(f"  Outlet {outlet_idx}: {len(trips)} trips.")
                               # Optional: print details of first few trips
                               # for i, trip in enumerate(trips[:2]): # Print max 2 trips per outlet
                               #     print(f"    Trip {i+1}: Type={trip.get('type')}, Load={format_float(trip.get('load'),1)}, Cost={format_float(trip.get('cost'),2)}, Time={format_float(trip.get('time'),2)}")
                           else:
                               print(f"  Outlet {outlet_idx}: 0 trips.")
                  else:
                      print("  No Stage 2 trips generated for any outlet.")

                  # Print summary of Served Customer Details
                  served_cust_details = greedy_results.get('served_customer_details', {})
                  print("\nServed Customer Details Summary (first few):")
                  if served_cust_details:
                       sorted_cust_indices = sorted(served_cust_details.keys())
                       for count, cust_idx in enumerate(sorted_cust_indices):
                           if count >= 5: print("  ..."); break # Print max 5
                           details = served_cust_details[cust_idx]
                           print(f"  Customer {cust_idx}: Initial={format_float(details['initial'],2)}, Satisfied={format_float(details['satisfied'],2)}, Remaining={format_float(details['remaining'],2)}, Status={details['status']}")
                  else:
                       print("  No served customer details available.")


        else:
             print("Greedy Heuristic run failed or returned no results.")

    except Exception as e:
        print(f"An unexpected error occurred during Greedy test execution: {e}")
        traceback.print_exc()

    print("\nStandalone test finished.")