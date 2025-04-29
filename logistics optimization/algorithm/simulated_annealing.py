# algorithm/simulated_annealing.py
# -*- coding: utf-8 -*-\
"""
Simulated Annealing (SA) algorithm adapted for Multi-Depot, Two-Echelon VRP
with Drones and Split Deliveries (MD-2E-VRPSD).

Explores the solution space using neighborhood search and a probabilistic
acceptance criterion based on temperature and solution quality difference.
Prioritizes feasible solutions (no unmet demand) during the acceptance decision
and best solution tracking using the SolutionCandidate's built-in comparison logic.

Relies on updated core utility functions (`create_initial_solution_mdsd`,
`generate_neighbor_solution_mdsd`) for initialization and neighborhood generation,
and the updated cost function (`calculate_total_cost_and_evaluate`) for evaluation.

This revised version integrates with the updated core cost function and
problem-specific utilities to prioritize feasible solutions (no unmet demand)
using a feasibility-first comparison strategy in accepting new solutions
and tracking the best solution found so far.
"""

import random
import copy
import time
import math
import traceback
import sys
import os
import numpy as np # Often useful, though not strictly required by this minimal SA structure
import warnings # Use warnings for non-critical issues

# --- Path Setup & Safe Imports ---
# Attempt to ensure the project root is in sys.path for robust imports
try:
    # Assumes this file is in project_root/algorithm
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root_sa = os.path.dirname(current_dir)
    if project_root_sa not in sys.path:
        sys.path.insert(0, project_root_sa)
        # print(f"SA: Added project root to sys.path: {project_root_sa}") # Optional debug print

    # Import necessary components from core modules
    # Need SolutionCandidate, initial solution generator, neighbor generator
    from core.problem_utils import (
        SolutionCandidate,
        create_initial_solution_mdsd, # Potentially used if initial_solution_candidate is not passed fully formed
        generate_neighbor_solution_mdsd, # Crucial for generating neighbors
        # Permutation mutations are used by generate_neighbor_solution_mdsd,
        # so we don't need to import them directly here.
        create_heuristic_trips_split_delivery # Needed by SolutionCandidate.evaluate indirectly
    )
    # Need distance and cost function references, but they are passed to SolutionCandidate.evaluate
    from core.distance_calculator import haversine # Needed by SolutionCandidate.evaluate indirectly
    from core.cost_function import calculate_total_cost_and_evaluate # Needed by SolutionCandidate.evaluate

except ImportError as e:
    print(f"CRITICAL ERROR in algorithm.simulated_annealing: Failed during initial import block: {e}")
    traceback.print_exc()
    # Define dummy functions/classes if imports fail to prevent immediate crash
    # but indicate severe error. Optimization cannot run without these.
    class SolutionCandidate:
        def __init__(self, *args, **kwargs): pass
        def evaluate(self, *args, **kwargs): pass
        def __lt__(self, other): return False # Always worse for dummy
        is_feasible = False
        weighted_cost = float('inf')
        stage1_routes = {}

    def create_initial_solution_mdsd(*args, **kwargs):
        print("DUMMY create_initial_solution_mdsd called in SA due to import error!")
        return None
    def generate_neighbor_solution_mdsd(solution):
        print("DUMMY generate_neighbor_solution_mdsd called in SA due to import error!")
        return None
    def haversine(*args): return float('inf')
    def calculate_total_cost_and_evaluate(*args, **kwargs): return float('inf'), float('inf'), float('inf'), float('inf'), {}, True, True, {}
    def create_heuristic_trips_split_delivery(*args, **kwargs): return [] # Dummy

    print("Simulated Annealing will use dummy functions/classes due to critical import failure.")

except Exception as e:
    print(f"An unexpected error occurred during SA import block: {e}")
    traceback.print_exc()
    # Define a dummy run function that indicates error
    def run_simulated_annealing(*args, **kwargs):
        print("CRITICAL ERROR: Simulated Annealing initialization failed.")
        return {'run_error': f"Initialization failed: {e}"}
    print("Simulated Annealing will not run due to unexpected import error.")


# Define a small tolerance for floating-point comparisons
FLOAT_TOLERANCE_SA = 1e-6


def run_simulated_annealing(problem_data: dict, vehicle_params: dict, drone_params: dict,
                            unmet_demand_penalty: float, cost_weight: float, time_weight: float,
                            initial_solution_candidate: SolutionCandidate | None, # SA can start from a provided initial
                            algo_specific_params: dict) -> dict | None:
    """
    Runs the Simulated Annealing (SA) algorithm to find a good solution for the
    Multi-Depot, Two-Echelon VRP with Drones and Split Deliveries.

    Explores the neighborhood of the current solution by perturbing the Stage 1
    routes and probabilistically accepting worse solutions based on temperature.

    Args:
        problem_data (dict): Dictionary containing problem instance data.
        vehicle_params (dict): Dictionary of vehicle parameters.
        drone_params (dict): Dictionary of drone parameters.
        unmet_demand_penalty (float): Penalty cost per unit of unmet demand.
        cost_weight (float): Weight for raw cost in objective.
        time_weight (float): Weight for time/makespan in objective.
        initial_solution_candidate (SolutionCandidate | None): An optional starting
                                                               SolutionCandidate. If None,
                                                               a greedy initial solution is generated.
                                                               This candidate contains initial assignments
                                                               and a starting Stage 1 route structure.
        algo_specific_params (dict): Dictionary of SA-specific parameters:
                                     'initial_temperature' (float), 'cooling_rate' (float),
                                     'max_iterations' (int).

    Returns:
        dict | None: A dictionary containing the best solution found (as a SolutionCandidate
                     object or a dictionary representing its state), its evaluation results,
                     and the cost history per iteration. Returns None or error dict
                     if the algorithm fails to run or find a valid solution.
        Example: {
            'best_solution': SolutionCandidate_object,
            'weighted_cost': float, 'evaluated_cost': float, 'evaluated_time': float,
            'evaluated_unmet': float, 'is_feasible': bool,
            'evaluation_stage1_error': bool, 'evaluation_stage2_error': bool,
            'stage1_routes': dict, 'stage2_trips': dict, 'served_customer_details': dict,
            'cost_history': list, # List of best weighted cost found so far per iteration
            'current_cost_history': list, # List of current solution's weighted cost per iteration
            'temperature_history': list, # List of temperature per iteration
            'total_computation_time': float,
            'algorithm_name': 'simulated_annealing',
            'algorithm_params': dict # Parameters used
        }
    """
    print("\n--- Starting Simulated Annealing (MD-SD) ---")
    start_time_sa = time.time()

    # --- Default SA Parameters ---
    default_sa_params = {
        'initial_temperature': 1000.0, # High initial temperature
        'cooling_rate': 0.99,       # Exponential cooling factor (0.9 to 0.99)
        'max_iterations': 10000     # Number of iterations (steps)
    }

    # --- Parameter Validation and Extraction ---
    try:
        # Merge default and provided parameters
        sa_params = default_sa_params.copy()
        if isinstance(algo_specific_params, dict):
             sa_params.update(algo_specific_params)
        else:
             warnings.warn("'algo_specific_params' is not a dictionary. Using default SA parameters.")


        initial_temperature = sa_params.get('initial_temperature')
        cooling_rate = sa_params.get('cooling_rate')
        max_iterations = sa_params.get('max_iterations')

        # Validate SA parameters
        if not isinstance(initial_temperature, (int, float)) or initial_temperature <= 0: raise ValueError("initial_temperature must be a positive number.")
        if not isinstance(cooling_rate, (int, float)) or not (0.0 < cooling_rate < 1.0): raise ValueError("cooling_rate must be between 0.0 and 1.0 (exclusive).")
        if not isinstance(max_iterations, int) or max_iterations < 0: raise ValueError("max_iterations must be a non-negative integer.")


        print(f"SA Parameters: Initial Temp={initial_temperature}, Cooling Rate={cooling_rate}, Max Iterations={max_iterations}")

    except Exception as e:
        print(f"Error validating SA parameters: {e}")
        traceback.print_exc()
        return {'run_error': f"Parameter validation failed: {e}"}


    # --- Initialization ---
    try:
        # Get or create the initial solution candidate
        if initial_solution_candidate is None:
            print("No initial solution provided, generating greedy initial solution...")
            # Need to pass full evaluation parameters to initial solution generator
            current_solution = create_initial_solution_mdsd(
                problem_data=problem_data,
                vehicle_params=vehicle_params,
                drone_params=drone_params,
                unmet_demand_penalty=unmet_demand_penalty,
                cost_weight=cost_weight,
                time_weight=time_weight
            )
            if current_solution is None or current_solution.weighted_cost == float('inf'):
                 raise RuntimeError("Failed to create a valid initial greedy solution.")
        else:
            # Use the provided initial solution candidate
            # Ensure it's a copy to avoid modifying the original object passed in
            current_solution = copy.deepcopy(initial_solution_candidate)
            # Ensure the provided solution is evaluated with the current weights/penalty
            print("Using provided initial solution, re-evaluating with current weights...")
            current_solution.evaluate(
                 distance_func=haversine,
                 stage2_trip_generator_func=create_heuristic_trips_split_delivery,
                 cost_weight=cost_weight,
                 time_weight=time_weight,
                 unmet_demand_penalty=unmet_demand_penalty
            )
            if current_solution.weighted_cost == float('inf'):
                 warnings.warn("Provided initial solution evaluated to infinite cost. SA may struggle.")


        # The initial solution is the first best solution found so far
        best_solution_overall = copy.deepcopy(current_solution)
        current_cost = current_solution.weighted_cost
        best_cost_overall = best_solution_overall.weighted_cost

        temperature = initial_temperature
        cost_history = [] # Best cost found so far at each iteration
        current_cost_history = [] # Cost of the current solution at each iteration
        temperature_history = [] # Temperature at each iteration

        print(f"SA Initialization complete. Initial Cost: {format_float(current_cost, 4)}, Initial Feasible: {current_solution.is_feasible}")

    except Exception as e:
        print(f"Error during SA initialization: {e}")
        traceback.print_exc()
        return {'run_error': f"Initialization failed: {e}"}


    # --- Simulated Annealing Loop ---
    print("Starting SA annealing loop...")
    for i in range(max_iterations):
        # print(f"\n--- Iteration {i + 1}/{max_iterations} (Temp: {format_float(temperature, 4)}) ---")

        # Store current state's cost and temperature for history before potentially moving
        cost_history.append(best_cost_overall) # Best cost *so far*
        current_cost_history.append(current_cost) # Cost of the solution we are currently at
        temperature_history.append(temperature)

        # Generate a neighbor solution
        # Uses generate_neighbor_solution_mdsd which perturbs one depot's Stage 1 route
        try:
            neighbor_solution = generate_neighbor_solution_mdsd(current_solution)

            if neighbor_solution is None:
                warnings.warn(f"Failed to generate neighbor in iteration {i}. Skipping.")
                # Continue loop without changing current_solution
                # The history lists should still append the previous state's values.
                continue # Skip to next iteration

            # Evaluate the neighbor solution
            neighbor_solution.evaluate(
                distance_func=haversine,
                stage2_trip_generator_func=create_heuristic_trips_split_delivery,
                cost_weight=cost_weight, # Ensure evaluation uses correct weights
                time_weight=time_weight,
                unmet_demand_penalty=unmet_demand_penalty
            )
            neighbor_cost = neighbor_solution.weighted_cost

        except Exception as e:
            warnings.warn(f"Error generating or evaluating neighbor in iteration {i}: {e}")
            traceback.print_exc()
            # Treat neighbor as having infinite cost if generation/evaluation fails
            neighbor_cost = float('inf')
            neighbor_solution = None # Ensure neighbor object is None if evaluation failed critically


        # --- Acceptance Decision ---
        # Use the __lt__ method for comparison, which prioritizes feasibility
        # cost_difference = neighbor_cost - current_cost # Use weighted cost difference

        # Accept the neighbor if it's better than the current solution (uses SolutionCandidate.__lt__)
        # Or accept probabilistically if it's worse (standard SA criterion)

        accept = False
        if neighbor_solution is not None: # Only consider acceptance if neighbor was generated and evaluated (cost is not inf from critical error)
            if neighbor_solution < current_solution: # Neighbor is strictly better (feasible-first comparison)
                accept = True
                # print(f"  Iteration {i}: Accepted better neighbor (Cost: {format_float(neighbor_cost, 4)})")
            else:
                # Neighbor is not better (either worse or equal).
                # Calculate acceptance probability if not better and not an infinite cost neighbor
                if current_cost is not None and not math.isinf(current_cost) and not math.isnan(current_cost) and \
                   neighbor_cost is not None and not math.isinf(neighbor_cost) and not math.isnan(neighbor_cost):

                    # Use the cost difference for the standard SA probability
                    cost_difference = neighbor_cost - current_cost

                    if cost_difference > 0: # Neighbor is worse (higher cost or less feasible)
                        # Calculate acceptance probability for a worse solution
                        try:
                            acceptance_probability = math.exp(-cost_difference / temperature)
                            # Accept with probability
                            if random.random() < acceptance_probability:
                                accept = True
                                # print(f"  Iteration {i}: Accepted worse neighbor (Cost: {format_float(neighbor_cost, 4)}) with probability {acceptance_probability:.4f}")
                        except OverflowError:
                             # Probability is effectively zero if cost difference is very large or temperature is very low
                             acceptance_probability = 0.0
                             # print(f"  Iteration {i}: Acceptance probability underflow (effectively 0).")

                    # If cost_difference <= 0, the neighbor is equal or better (already handled by neighbor_solution < current_solution)
                    # So, if we reach here and not accepted, it must be strictly worse.

                # else:
                    # print(f"  Iteration {i}: Neighbor cost is inf/NaN ({format_float(neighbor_cost, 4)}) or current cost is inf/NaN. Not accepting worse.")


        if accept:
            # Move to the neighbor solution
            current_solution = neighbor_solution # current_solution now points to the neighbor object
            current_cost = neighbor_cost # Update current cost

            # Update the overall best solution found so far if the accepted neighbor is better
            if current_solution < best_solution_overall: # Use SolutionCandidate.__lt__ for comparison
                 best_solution_overall = copy.deepcopy(current_solution)
                 best_cost_overall = best_solution_overall.weighted_cost
                 # print(f"  Iteration {i}: Found new overall best solution (Cost: {format_float(best_cost_overall, 4)})")
        # else:
            # print(f"  Iteration {i}: Did not accept neighbor. Current Cost: {format_float(current_cost, 4)}")


        # --- Cooling Schedule ---
        # Exponential cooling: T = T0 * alpha^k
        temperature *= cooling_rate

        # Optional: Print progress periodically
        if (i + 1) % (max_iterations // 10 or 1) == 0: # Print roughly 10 times
            print(f"  Iteration {i + 1}/{max_iterations}: Current Cost={format_float(current_cost, 4)}, Best Cost={format_float(best_cost_overall, 4)}, Temp={format_float(temperature, 4)}, Feasible={current_solution.is_feasible}")


    # --- SA Finished ---
    end_time_sa = time.time()
    total_time_sa = end_time_sa - start_time_sa
    print(f"\nSimulated Annealing (MD-SD) finished after {max_iterations} iterations in {total_time_sa:.4f} seconds.")

    # Final evaluation of the overall best solution found (if any)
    if best_solution_overall:
        print("Re-evaluating overall best solution found by SA...")
        try:
            best_solution_overall.evaluate(
                distance_func=haversine,
                stage2_trip_generator_func=create_heuristic_trips_split_delivery,
                cost_weight=cost_weight, # Ensure final evaluation uses the requested weights
                time_weight=time_weight,
                unmet_demand_penalty=unmet_demand_penalty
            )
            print(f"SA Final Best Evaluation: Feasible: {best_solution_overall.is_feasible}, Weighted Cost: {format_float(best_solution_overall.weighted_cost, 4)}")

            # Prepare the result dictionary
            sa_results = {
                'best_solution': best_solution_overall, # Return the SolutionCandidate object
                'weighted_cost': best_solution_overall.weighted_cost,
                'evaluated_cost': best_solution_overall.evaluated_cost,
                'evaluated_time': best_solution_overall.evaluated_time,
                'evaluated_unmet': best_solution_overall.evaluated_unmet,
                'is_feasible': best_solution_overall.is_feasible,
                'evaluation_stage1_error': best_solution_overall.evaluation_stage1_error,
                'evaluation_stage2_error': best_solution_overall.evaluation_stage2_error,
                'stage1_routes': best_solution_overall.stage1_routes, # Include final routes
                'stage2_trips': best_solution_overall.stage2_trips, # Include final trips
                'served_customer_details': best_solution_overall.served_customer_details, # Include customer details
                'cost_history': cost_history, # Return the history of best cost found so far
                'current_cost_history': current_cost_history, # Return the history of the current solution's cost
                'temperature_history': temperature_history, # Return temperature history
                'total_computation_time': total_time_sa,
                'algorithm_name': 'simulated_annealing',
                'algorithm_params': sa_params # Store parameters used
            }
            return sa_results

        except Exception as e:
            print(f"Error during final evaluation of SA best solution: {e}")
            traceback.print_exc()
            # Return partial results with error indicated
            return {
                'best_solution': best_solution_overall, # Return the object even if final eval failed
                'weighted_cost': float('inf'), # Indicate final evaluation failure
                'evaluated_cost': float('inf'),
                'evaluated_time': float('inf'),
                'evaluated_unmet': float('inf'),
                'is_feasible': False,
                'evaluation_stage1_error': True, # Assume error in final eval
                'evaluation_stage2_error': True, # Assume error in final eval
                'stage1_routes': best_solution_overall.stage1_routes if best_solution_overall else {}, # Return the routes found
                'stage2_trips': {}, # Stage 2 trips were not generated in final eval or failed
                'served_customer_details': {},
                'cost_history': cost_history,
                'current_cost_history': current_cost_history,
                'temperature_history': temperature_history,
                'total_computation_time': total_time_sa,
                'algorithm_name': 'simulated_annealing',
                'algorithm_params': sa_params,
                'run_error': f"Final evaluation failed: {e}"
            }

    else:
        print("Simulated Annealing did not find a valid best solution.")
        return {
             'best_solution': None,
             'weighted_cost': float('inf'),
             'evaluated_cost': float('inf'),
             'evaluated_time': float('inf'),
             'evaluated_unmet': float('inf'),
             'is_feasible': False,
             'evaluation_stage1_error': True,
             'evaluation_stage2_error': True,
             'stage1_routes': {},
             'stage2_trips': {},
             'served_customer_details': {},
             'cost_history': cost_history,
             'current_cost_history': current_cost_history,
             'temperature_history': temperature_history,
             'total_computation_time': total_time_sa,
             'algorithm_name': 'simulated_annealing',
             'algorithm_params': sa_params,
             'run_error': "No valid solution found by SA."
        }


# --- Helper functions (Placeholder for format_float) ---
# Assuming format_float is available (e.g., defined in problem_utils or report_generator)
# Copying it here for standalone SA execution testing, but prefer central definition.
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

# --- Dummy create_initial_solution_mdsd for standalone test if not imported ---
# Define a minimal dummy if the real one failed to import
try:
     # Check if the real function was imported successfully
     _ = create_initial_solution_mdsd # This will raise NameError if not defined/imported
except NameError:
     print("Using DUMMY create_initial_solution_mdsd for standalone SA test.")
     def create_initial_solution_mdsd(problem_data, vehicle_params, drone_params, unmet_demand_penalty, cost_weight, time_weight):
          print("DUMMY create_initial_solution_mdsd called in SA.")
          # Create a minimal dummy SolutionCandidate
          dummy_locations = problem_data.get('locations', {})
          num_depots = len(dummy_locations.get('logistics_centers', []))
          num_outlets = len(dummy_locations.get('sales_outlets', []))

          dummy_stage1_routes = {depot_idx: list(range(num_outlets)) for depot_idx in range(num_depots)} # Dummy routes visiting all outlets
          for depot_idx in dummy_stage1_routes: random.shuffle(dummy_stage1_routes[depot_idx]) # Shuffle dummy routes

          dummy_candidate = SolutionCandidate(problem_data=problem_data,
                                              vehicle_params=vehicle_params,
                                              drone_params=drone_params,
                                              unmet_demand_penalty=unmet_demand_penalty,
                                              cost_weight=cost_weight,
                                              time_weight=time_weight,
                                              initial_stage1_routes=dummy_stage1_routes,
                                              initial_outlet_to_depot_assignments={}, # Dummy assignments
                                              initial_customer_to_outlet_assignments={}) # Dummy assignments
          # Evaluate the dummy candidate
          try:
               # Assuming haversine and create_heuristic_trips_split_delivery are available (imported or dummy)
               dummy_candidate.evaluate(haversine, create_heuristic_trips_split_delivery, cost_weight, time_weight, unmet_demand_penalty)
          except Exception as eval_e:
               print(f"Error evaluating DUMMY initial solution in SA: {eval_e}")
               dummy_candidate.is_feasible = False
               dummy_candidate.weighted_cost = float('inf')
               dummy_candidate.evaluation_stage1_error = True
               dummy_candidate.evaluation_stage2_error = True

          return dummy_candidate

# --- Dummy generate_neighbor_solution_mdsd for standalone test if not imported ---
# Define a minimal dummy if the real one failed to import
try:
    # Check if the real function was imported successfully
    _ = generate_neighbor_solution_mdsd # This will raise NameError if not defined/imported
except NameError:
     print("Using DUMMY generate_neighbor_solution_mdsd for standalone SA test.")
     def generate_neighbor_solution_mdsd(current_solution):
          print("DUMMY generate_neighbor_solution_mdsd called in SA.")
          if not isinstance(current_solution, SolutionCandidate) or not current_solution.stage1_routes:
               return None

          # Create a deep copy
          neighbor_solution = copy.deepcopy(current_solution)

          # Perform a simple dummy swap mutation on a random depot's route
          depot_indices = list(neighbor_solution.stage1_routes.keys())
          if not depot_indices: return neighbor_solution

          selected_depot_index = random.choice(depot_indices)
          route = neighbor_solution.stage1_routes.get(selected_depot_index, [])

          if len(route) >= 2:
               idx1, idx2 = random.sample(range(len(route)), 2)
               route[idx1], route[idx2] = route[idx2], route[idx1]
               neighbor_solution.stage1_routes[selected_depot_index] = route # Update the route

          # Mark as unevaluated
          neighbor_solution.is_feasible = False
          neighbor_solution.weighted_cost = float('inf')
          # Reset other evaluation results as well
          neighbor_solution.evaluated_cost = float('inf')
          neighbor_solution.evaluated_time = float('inf')
          neighbor_solution.evaluated_unmet = float('inf')
          neighbor_solution.served_customer_details = {}
          neighbor_solution.evaluation_stage1_error = False
          neighbor_solution.evaluation_stage2_error = False
          neighbor_solution.stage2_trips = {}


          return neighbor_solution


# --- Optional Main Execution Block for Standalone Testing ---
if __name__ == '__main__':
    """
    Standalone execution block for testing the Simulated Annealing algorithm.
    Requires dummy problem data and uses the dummy create_initial_solution_mdsd
    and generate_neighbor_solution_mdsd (or real if imported).
    """
    print("Running algorithm/simulated_annealing.py in standalone test mode.")

    # --- Create Dummy Problem Data ---
    # Needs to be sufficient for create_initial_solution_mdsd and subsequent evaluation
    try:
        print("\n--- Creating Dummy Problem Data for SA Test ---")
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
             ],
             'customers': [
                 (34.06, -118.11), # Customer 0
                 (34.05, -118.09), # Customer 1
                 (34.00, -118.06), # Customer 2
                 (34.16, -118.31), # Customer 3
                 (34.14, -118.28), # Customer 4
                 (33.96, -118.16), # Customer 5
             ]
         }
        dummy_demands = [10.0, 15.0, 8.0, 20.0, 12.0, 5.0] # len = 6

        dummy_problem_data_sa = {
            'locations': dummy_locations,
            'demands': dummy_demands
        }

        dummy_vehicle_params_sa = {'payload': 200.0, 'cost_per_km': 1.5, 'speed_kmph': 60.0}
        dummy_drone_params_sa = {'payload': 30.0, 'max_flight_distance_km': 15.0, 'cost_per_km': 0.8, 'speed_kmph': 100.0}
        dummy_unmet_penalty_sa = 500.0
        dummy_cost_weight_sa = 1.0
        dummy_time_weight_sa = 0.1

        # Dummy SA Parameters
        dummy_sa_params = {
            'initial_temperature': 500.0, # Lower temp for faster test convergence
            'cooling_rate': 0.95,       # Faster cooling
            'max_iterations': 200     # Fewer iterations
        }

        print("Dummy data and SA parameters created.")

    except Exception as e:
        print(f"Error creating dummy data for SA test: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Create a Dummy Initial Solution Candidate ---
    # Use the potentially dummy or real create_initial_solution_mdsd
    print("\n--- Creating Dummy Initial Solution Candidate ---")
    dummy_initial_solution_sa = create_initial_solution_mdsd(
        problem_data=dummy_problem_data_sa,
        vehicle_params=dummy_vehicle_params_sa,
        drone_params=dummy_drone_params_sa,
        unmet_demand_penalty=dummy_unmet_penalty_sa,
        cost_weight=dummy_cost_weight_sa,
        time_weight=dummy_time_weight_sa
    )

    if dummy_initial_solution_sa is None:
        print("Failed to create dummy initial solution. Cannot run SA test.")
        sys.exit(1)
    else:
        print(f"Dummy initial solution created: Feasible={dummy_initial_solution_sa.is_feasible}, Weighted Cost={format_float(dummy_initial_solution_sa.weighted_cost, 4)}")
        print("Stage 1 Routes in initial solution:", dummy_initial_solution_sa.stage1_routes)


    # --- Run the SA ---
    print("\n--- Running Simulated Annealing (dummy data) ---")
    try:
        sa_results = run_simulated_annealing(
            problem_data=dummy_problem_data_sa,
            vehicle_params=dummy_vehicle_params_sa,
            drone_params=dummy_drone_params_sa,
            unmet_demand_penalty=dummy_unmet_penalty_sa,
            cost_weight=dummy_cost_weight_sa,
            time_weight=dummy_time_weight_sa,
            initial_solution_candidate=dummy_initial_solution_sa,
            algo_specific_params=dummy_sa_params
        )

        print("\n--- SA Results Summary ---")
        if sa_results:
             print(f"Algorithm Name: {sa_results.get('algorithm_name')}")
             print(f"Run Time: {format_float(sa_results.get('total_computation_time'), 4)} seconds")
             if sa_results.get('run_error'):
                  print(f"Run Error: {sa_results.get('run_error')}")
             else:
                  best_solution = sa_results.get('best_solution')
                  if best_solution:
                       print("\nBest Solution Found:")
                       print(f"  Feasible: {best_solution.is_feasible}")
                       print(f"  Weighted Cost: {format_float(best_solution.weighted_cost, 4)}")
                       print(f"  Raw Cost: {format_float(best_solution.evaluated_cost, 2)}")
                       print(f"  Time (Makespan): {format_float(best_solution.evaluated_time, 2)}")
                       print(f"  Unmet Demand: {format_float(best_solution.evaluated_unmet, 2)}")
                       print("  Final Stage 1 Routes:", best_solution.stage1_routes)
                       # print("  Final Stage 2 Trips:", best_solution.stage2_trips) # Can be verbose
                       # print("  Served Customer Details:", best_solution.served_customer_details) # Can be verbose

                  print("\nCost History (Best found so far per iteration):")
                  history = sa_results.get('cost_history', [])
                  # Print first few, mid few, and last few
                  if len(history) > 20: # Adjust number based on max_iterations
                       print([format_float(c, 4) for c in history[:5]] + ['...'] + [format_float(c, 4) for c in history[len(history)//2-2 : len(history)//2+3]] + ['...'] + [format_float(c, 4) for c in history[-5:]])
                  else:
                       print([format_float(c, 4) for c in history])

                  # Optional: Print current cost history
                  # print("\nCurrent Cost History per iteration:")
                  # current_history = sa_results.get('current_cost_history', [])
                  # if len(current_history) > 20:
                  #      print([format_float(c, 4) for c in current_history[:5]] + ['...'] + [format_float(c, 4) for c in current_history[len(current_history)//2-2 : len(current_history)//2+3]] + ['...'] + [format_float(c, 4) for c in current_history[-5:]])
                  # else:
                  #      print([format_float(c, 4) for c in current_history])

                  # Optional: Print temperature history
                  # print("\nTemperature History per iteration:")
                  # temp_history = sa_results.get('temperature_history', [])
                  # if len(temp_history) > 20:
                  #      print([format_float(c, 4) for c in temp_history[:5]] + ['...'] + [format_float(c, 4) for c in temp_history[len(temp_history)//2-2 : len(temp_history)//2+3]] + ['...'] + [format_float(c, 4) for c in temp_history[-5:]])
                  # else:
                  #      print([format_float(c, 4) for c in temp_history])


        else:
             print("Simulated Annealing run failed or returned no results.")

    except Exception as e:
        print(f"An unexpected error occurred during SA test execution: {e}")
        traceback.print_exc()

    print("\nStandalone test finished.")