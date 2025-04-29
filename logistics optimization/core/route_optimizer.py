# core/route_optimizer.py
# -*- coding: utf-8 -*-
"""
Orchestrates the Multi-Depot, Two-Echelon Logistics Optimization Process
with Drones and Split Deliveries (MD-2E-VRPSD).

This module acts as the central coordinator for the optimization pipeline.
It handles:
- Input validation and parameter checking.
- Generation of an initial solution structure (including assignments).
- Execution of selected optimization algorithms (GA, SA, PSO, Greedy).
- Consistent evaluation of solutions using the core cost function.
- Aggregation and comparison of results from different algorithms.
- Triggering the generation of visualization (maps) and reports.
"""

import json
import sys
import time as pytime
import os
import traceback
import copy
import math
import warnings
import logging
from typing import List, Dict, Tuple, Optional, Any, Callable

# --- Setup Logging ---
# Configure logging for better diagnostics. Can be adjusted (e.g., write to file).
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# --- Define Constants Locally ---
# Define floating-point tolerance locally within this module
FLOAT_TOLERANCE = 1e-6

# --- Safe Core Imports ---
# Attempt to import necessary components, raising critical errors if missing.
try:
    logger.debug("Importing core utilities and functions...")
    from core.distance_calculator import haversine
    from core.cost_function import calculate_total_cost_and_evaluate
    # Import from problem_utils, REMOVING FLOAT_TOLERANCE from here
    from core.problem_utils import (
        SolutionCandidate,
        create_initial_solution_mdsd,
        create_heuristic_trips_split_delivery,
        # FLOAT_TOLERANCE, # <-- Removed this import
    )

    logger.debug("Core utilities imported successfully.")
except ImportError as e:
    logger.critical(f"Failed to import core utilities (problem_utils, cost_function, distance_calculator): {e}")
    # Raise an error or define dummy functions if you want the application
    # to potentially start in a degraded state (not recommended for core logic).
    raise ImportError(f"Core modules failed to load: {e}") from e

# --- Safe Algorithm Imports ---
# Import available algorithm modules using the package structure.
try:
    logger.debug("Importing algorithm modules...")
    # These imports rely on algorithm/__init__.py exposing the run functions.
    from algorithm import (
        run_genetic_algorithm,
        run_simulated_annealing,
        run_pso_optimizer,
        run_greedy_heuristic,
    )

    # Mapping of algorithm keys (used internally and by GUI) to their run functions
    ALGORITHM_REGISTRY: Dict[str, Callable] = {
        "genetic_algorithm": run_genetic_algorithm,
        "simulated_annealing": run_simulated_annealing,
        "pso_optimizer": run_pso_optimizer,
        "greedy_heuristic": run_greedy_heuristic,
        # Add new algorithms here following the same pattern
    }
    logger.debug(f"Registered algorithms: {list(ALGORITHM_REGISTRY.keys())}")

except ImportError as e:
    logger.critical(f"Failed to import one or more algorithms from the 'algorithm' package: {e}")
    # Define an empty registry or raise an error
    ALGORITHM_REGISTRY = {}
    # Depending on application design, you might allow running without all algorithms
    # raise ImportError(f"Algorithm modules failed to load: {e}") from e

# --- Safe Visualization/Reporting Imports (Optional Components) ---
_VISUALIZATION_AVAILABLE = False
_REPORTING_AVAILABLE = False

# Map Generation
try:
    from visualization.map_generator import generate_folium_map, open_map_in_browser
    _VISUALIZATION_AVAILABLE = True
    logger.debug("Map generator imported successfully.")
except ImportError:
    logger.warning(
        "Map generator ('visualization.map_generator') not available. "
        "Install 'folium' or check path. Map generation will be skipped."
    )
    # Define dummy functions if needed, or just rely on the flag
    def generate_folium_map(*args, **kwargs): logger.warning("Dummy generate_folium_map called."); return None
    def open_map_in_browser(*args, **kwargs): logger.warning("Dummy open_map_in_browser called.")

# Report Generation
try:
    from utils.report_generator import generate_delivery_report
    _REPORTING_AVAILABLE = True
    logger.debug("Report generator imported successfully.")
except ImportError:
    logger.warning(
        "Report generator ('utils.report_generator') not available. "
        "Check path. Report generation will be skipped."
    )
    # Define dummy function
    def generate_delivery_report(*args, **kwargs): logger.warning("Dummy generate_delivery_report called."); return "Report generation unavailable."


# --- Main Optimization Function ---

def run_optimization(
    problem_data: Dict[str, Any],
    vehicle_params: Dict[str, Any],
    drone_params: Dict[str, Any],
    optimization_params: Dict[str, Any],
    selected_algorithm_keys: List[str],
    objective_weights: Dict[str, float],
) -> Dict[str, Any]:
    """
    Main entry point for the optimization process. Orchestrates data validation,
    initialization, algorithm execution, evaluation, and result aggregation
    for the Multi-Depot, Two-Echelon VRP with Drones and Split Deliveries.

    Args:
        problem_data: Dictionary containing problem instance data:
            'locations': {'logistics_centers': [...], 'sales_outlets': [...], 'customers': [...]},
            'demands': List of initial customer demands.
        vehicle_params: Dictionary of vehicle parameters
                        ('payload', 'cost_per_km', 'speed_kmph').
        drone_params: Dictionary of drone parameters
                      ('payload', 'max_flight_distance_km', 'cost_per_km', 'speed_kmph').
        optimization_params: Dictionary containing parameters for optimization process:
            'unmet_demand_penalty' (float), 'output_dir' (str).
            Must also include nested dictionaries for each selected algorithm's specific parameters,
            keyed as '{algo_key}_params'. E.g., 'genetic_algorithm_params': {...}.
        selected_algorithm_keys: A list of string keys identifying the algorithms to run
                                 (must match keys in ALGORITHM_REGISTRY).
        objective_weights: Dictionary containing weights for the objective function:
                           'cost_weight' (float), 'time_weight' (float).

    Returns:
        A dictionary containing the results of the optimization run(s),
        including overall status, error messages, and detailed results
        for each executed algorithm. Structure includes:
            'overall_status': str ('Success', 'Validation Failed', 'Preprocessing Failed', 'No Valid Results', 'Error')
            'error_message': Optional[str]
            'run_timestamp': str
            'output_directory': str
            'results_by_algorithm': Dict[str, Dict[str, Any]] (details below)
            'best_algorithm_key': Optional[str] (key of the best overall result by weighted cost)
            'best_weighted_cost': float
            'fully_served_best_key': Optional[str] (key of the best feasible result)
            'fully_served_best_cost': float
            'total_computation_time': float
            'parameters_used': Dict[str, Any] (copy of inputs)

        Structure of 'results_by_algorithm'[algo_key]:
            'algorithm_name': str
            'status': str ('Success', 'Failed', 'Skipped')
            'run_error': Optional[str]
            'computation_time': float
            'result_data': Optional[Dict[str, Any]] (containing evaluation metrics, solution structure, history)
            'map_path': Optional[str]
            'report_path': Optional[str]
            'map_generation_error': Optional[str]
            'report_generation_error': Optional[str]
    """
    run_start_time = pytime.time()
    run_timestamp = pytime.strftime("%Y-%m-%d_%H-%M-%S")
    logger.info(f"--- Starting Optimization Run ({run_timestamp}) ---")

    # --- Prepare Output Directory ---
    base_output_dir = optimization_params.get("output_dir", "output")
    run_output_dir = os.path.join(base_output_dir, run_timestamp)
    try:
        os.makedirs(run_output_dir, exist_ok=True)
        # Also ensure subdirs for maps and reports exist within the run directory
        os.makedirs(os.path.join(run_output_dir, "maps"), exist_ok=True)
        os.makedirs(os.path.join(run_output_dir, "reports"), exist_ok=True)
        logger.info(f"Output directory for this run: {run_output_dir}")
    except OSError as e:
        logger.error(f"Failed to create output directory '{run_output_dir}': {e}")
        # Return an error result immediately if output dir creation fails
        return {
            "overall_status": "Error",
            "error_message": f"Failed to create output directory: {e}",
            "run_timestamp": run_timestamp,
            "output_directory": run_output_dir,
            "results_by_algorithm": {},
            "best_algorithm_key": None, "best_weighted_cost": float("inf"),
            "fully_served_best_key": None, "fully_served_best_cost": float("inf"),
            "total_computation_time": pytime.time() - run_start_time,
            "parameters_used": { # Include parameters even on early failure
                "problem_data": problem_data, "vehicle_params": vehicle_params,
                "drone_params": drone_params, "optimization_params": optimization_params,
                "selected_algorithm_keys": selected_algorithm_keys, "objective_weights": objective_weights
            }
        }


    # Initialize overall results structure
    overall_results = {
        "overall_status": "Running",
        "error_message": None,
        "run_timestamp": run_timestamp,
        "output_directory": run_output_dir, # Store the specific output dir for this run
        "results_by_algorithm": {},
        "best_algorithm_key": None, "best_weighted_cost": float("inf"),
        "fully_served_best_key": None, "fully_served_best_cost": float("inf"),
        "total_computation_time": 0.0,
        "parameters_used": {
            "problem_data_summary": { # Store summary instead of full data
                 "num_logistics_centers": len(problem_data.get('locations', {}).get('logistics_centers', [])),
                 "num_sales_outlets": len(problem_data.get('locations', {}).get('sales_outlets', [])),
                 "num_customers": len(problem_data.get('locations', {}).get('customers', [])),
                 "total_initial_demand": sum(problem_data.get('demands', []))
            },
            "vehicle_params": copy.deepcopy(vehicle_params),
            "drone_params": copy.deepcopy(drone_params),
            "optimization_params": copy.deepcopy(optimization_params),
            "selected_algorithm_keys": copy.deepcopy(selected_algorithm_keys),
            "objective_weights": copy.deepcopy(objective_weights),
        },
    }

    # --- Input Validation ---
    logger.info("Validating input data and parameters...")
    validation_errors = []
    try:
        # Problem Data
        if not isinstance(problem_data, dict) or 'locations' not in problem_data or 'demands' not in problem_data:
            validation_errors.append("Invalid 'problem_data' structure (missing 'locations' or 'demands').")
        else:
            locs = problem_data.get('locations', {})
            if not isinstance(locs, dict) or not all(k in locs for k in ['logistics_centers', 'sales_outlets', 'customers']):
                 validation_errors.append("Problem data 'locations' must be a dict with keys 'logistics_centers', 'sales_outlets', 'customers'.")
            elif not locs.get('logistics_centers') or not isinstance(locs.get('logistics_centers'), list):
                 validation_errors.append("Problem data needs at least one logistics center defined as a list.")
            # Allow empty outlets/customers if needed by problem setup
            # Check demand list matches customer count
            demands = problem_data.get('demands', [])
            num_customers = len(locs.get('customers', []))
            if not isinstance(demands, list):
                 validation_errors.append("'demands' must be a list.")
            elif len(demands) != num_customers:
                 validation_errors.append(f"Number of demands ({len(demands)}) does not match number of customers ({num_customers}).")
            elif any(not isinstance(d, (int, float)) or d < 0 for d in demands):
                 validation_errors.append("All customer demands must be non-negative numbers.")

        # Vehicle Params (example checks, add more as needed)
        if not isinstance(vehicle_params, dict) or not all(k in vehicle_params for k in ['payload', 'cost_per_km', 'speed_kmph']):
            validation_errors.append("Incomplete 'vehicle_params'. Required keys: 'payload', 'cost_per_km', 'speed_kmph'.")
        elif not isinstance(vehicle_params.get('speed_kmph'), (int, float)) or vehicle_params['speed_kmph'] <= 0:
            validation_errors.append("Vehicle speed must be a positive number.")

        # Drone Params (example checks)
        if not isinstance(drone_params, dict) or not all(k in drone_params for k in ['payload', 'max_flight_distance_km', 'cost_per_km', 'speed_kmph']):
            validation_errors.append("Incomplete 'drone_params'. Required keys: 'payload', 'max_flight_distance_km', 'cost_per_km', 'speed_kmph'.")
        elif not isinstance(drone_params.get('speed_kmph'), (int, float)) or drone_params['speed_kmph'] <= 0:
            validation_errors.append("Drone speed must be a positive number.")
        elif not isinstance(drone_params.get('max_flight_distance_km'), (int, float)) or drone_params['max_flight_distance_km'] < 0:
            validation_errors.append("Drone max flight distance cannot be negative.")


        # Optimization Params
        if not isinstance(optimization_params, dict) or 'unmet_demand_penalty' not in optimization_params:
            validation_errors.append("'optimization_params' must include 'unmet_demand_penalty'.")
        elif not isinstance(optimization_params.get('unmet_demand_penalty'), (int, float)) or optimization_params['unmet_demand_penalty'] < 0:
             validation_errors.append("Unmet demand penalty must be a non-negative number.")

        # Selected Algorithms
        if not isinstance(selected_algorithm_keys, list) or not selected_algorithm_keys:
            validation_errors.append("'selected_algorithm_keys' must be a non-empty list.")
        else:
            invalid_keys = [k for k in selected_algorithm_keys if k not in ALGORITHM_REGISTRY]
            if invalid_keys:
                 validation_errors.append(f"Unknown or unavailable algorithm keys selected: {invalid_keys}. Available: {list(ALGORITHM_REGISTRY.keys())}")
            # Check if required algo-specific params are present in optimization_params
            for key in selected_algorithm_keys:
                 if key in ALGORITHM_REGISTRY and f'{key}_params' not in optimization_params:
                      validation_errors.append(f"Missing parameters dictionary '{key}_params' in 'optimization_params' for selected algorithm '{key}'.")
                 elif key in ALGORITHM_REGISTRY and not isinstance(optimization_params.get(f'{key}_params'), dict):
                      validation_errors.append(f"Parameters '{key}_params' must be a dictionary.")


        # Objective Weights
        if not isinstance(objective_weights, dict) or not all(k in objective_weights for k in ['cost_weight', 'time_weight']):
             validation_errors.append("Incomplete 'objective_weights'. Required keys: 'cost_weight', 'time_weight'.")
        elif any(not isinstance(w, (int, float)) or w < 0 for w in objective_weights.values()):
             validation_errors.append("Objective weights ('cost_weight', 'time_weight') must be non-negative numbers.")

        # Check for critical import failures
        if not ALGORITHM_REGISTRY:
             validation_errors.append("CRITICAL: No optimization algorithms were loaded successfully.")

        # Final check
        if validation_errors:
             raise ValueError("Input validation failed:\n - " + "\n - ".join(validation_errors))

        logger.info("Input data and parameters validated successfully.")

    except ValueError as ve:
        overall_results['overall_status'] = 'Validation Failed'
        overall_results['error_message'] = str(ve)
        logger.error(f"Input Validation Error: {ve}")
        overall_results['total_computation_time'] = pytime.time() - run_start_time
        return overall_results
    except Exception as e:
        overall_results['overall_status'] = 'Validation Failed'
        error_msg = f"Unexpected error during validation: {e}"
        overall_results['error_message'] = error_msg
        logger.error(error_msg, exc_info=True)
        overall_results['total_computation_time'] = pytime.time() - run_start_time
        return overall_results

    # --- Preprocessing: Generate Initial Solution & Assignments ---
    logger.info("Generating initial solution structure and assignments...")
    try:
        # Use the core utility to create a base solution, which includes assignments.
        base_initial_solution: Optional[SolutionCandidate] = create_initial_solution_mdsd(
            problem_data=problem_data,
            vehicle_params=vehicle_params,
            drone_params=drone_params,
            unmet_demand_penalty=optimization_params['unmet_demand_penalty'],
            cost_weight=objective_weights['cost_weight'],
            time_weight=objective_weights['time_weight']
        )

        if base_initial_solution is None:
            raise RuntimeError("Failed to create base initial solution (returned None).")
        if not base_initial_solution.stage1_routes or not base_initial_solution.outlet_to_depot_assignments or not base_initial_solution.customer_to_outlet_assignments:
             raise RuntimeError("Initial solution structure missing routes or assignments.")
        # Check if initial evaluation failed critically
        if base_initial_solution.evaluation_stage1_error or base_initial_solution.evaluation_stage2_error:
            logger.warning(f"Initial solution evaluation encountered errors (S1: {base_initial_solution.evaluation_stage1_error}, S2: {base_initial_solution.evaluation_stage2_error}). Proceeding, but algorithms might struggle.")


        logger.info(f"Initial solution generated. Initial Weighted Cost: {base_initial_solution.weighted_cost:.4f}, Feasible: {base_initial_solution.is_feasible}")

    except Exception as e:
        overall_results['overall_status'] = 'Preprocessing Failed'
        error_msg = f"Error during initial solution generation: {e}"
        overall_results['error_message'] = error_msg
        logger.error(error_msg, exc_info=True)
        overall_results['total_computation_time'] = pytime.time() - run_start_time
        return overall_results

    # --- Run Selected Algorithms ---
    valid_algorithm_keys = [key for key in selected_algorithm_keys if key in ALGORITHM_REGISTRY]
    logger.info(f"Running selected valid algorithms: {valid_algorithm_keys}")

    for algo_key in valid_algorithm_keys:
        algo_run_func = ALGORITHM_REGISTRY[algo_key]
        algo_results_summary = { # Initialize summary structure for this algorithm
            'algorithm_name': algo_key,
            'status': 'Running',
            'run_error': None,
            'computation_time': 0.0,
            'result_data': None,
            'map_path': None,
            'report_path': None,
            'map_generation_error': None,
            'report_generation_error': None
        }
        overall_results['results_by_algorithm'][algo_key] = algo_results_summary

        logger.info(f"--- Starting Algorithm: {algo_key} ---")
        algo_start_time = pytime.time()

        try:
            # Get algorithm-specific parameters
            algo_specific_params = optimization_params.get(f'{algo_key}_params', {})

            # Call the algorithm's run function
            # Pass necessary data, parameters, and the initial solution candidate
            # Note: Greedy might not use initial_solution_candidate directly if it rebuilds
            # Ensure algorithm signatures match the expected arguments.
            if algo_key == 'greedy_heuristic':
                 # Greedy heuristic might rebuild solution from scratch
                 # Signature assumed: run_greedy_heuristic(problem_data, vehicle_params, drone_params, unmet_penalty, cost_w, time_w)
                 algorithm_raw_result = algo_run_func(
                     problem_data=problem_data,
                     vehicle_params=vehicle_params,
                     drone_params=drone_params,
                     unmet_demand_penalty=optimization_params['unmet_demand_penalty'],
                     cost_weight=objective_weights['cost_weight'],
                     time_weight=objective_weights['time_weight']
                     # Pass algo_specific_params if greedy ever needs them
                     # algo_specific_params=algo_specific_params
                 )
            else:
                 # Iterative algorithms start from the initial solution
                 # Signature assumed: run_*(problem_data, vehicle_params, drone_params, unmet_penalty, cost_w, time_w, initial_solution_candidate, algo_specific_params)
                 algorithm_raw_result = algo_run_func(
                     problem_data=problem_data,
                     vehicle_params=vehicle_params,
                     drone_params=drone_params,
                     unmet_demand_penalty=optimization_params['unmet_demand_penalty'],
                     cost_weight=objective_weights['cost_weight'],
                     time_weight=objective_weights['time_weight'],
                     initial_solution_candidate=copy.deepcopy(base_initial_solution), # Pass a copy
                     algo_specific_params=algo_specific_params
                 )

            # --- Process Algorithm Result ---
            if algorithm_raw_result is None:
                 raise RuntimeError("Algorithm returned None.")

            # Standardize the result into a dictionary format if it's not already
            processed_result_data = {}
            if isinstance(algorithm_raw_result, SolutionCandidate):
                 # If algorithm returns the SolutionCandidate object directly
                 final_solution: SolutionCandidate = algorithm_raw_result
                 logger.info(f"Algorithm {algo_key} returned SolutionCandidate. Re-evaluating for consistency...")
                 # Re-evaluate the final solution *outside* the algorithm for consistency
                 final_solution.evaluate(
                     distance_func=haversine,
                     stage2_trip_generator_func=create_heuristic_trips_split_delivery,
                     cost_weight=objective_weights['cost_weight'],
                     time_weight=objective_weights['time_weight'],
                     unmet_demand_penalty=optimization_params['unmet_demand_penalty']
                 )
                 # Populate the standard result dictionary
                 processed_result_data = {
                      'best_solution_structure': final_solution.to_dict(), # Store structure as dict
                      'weighted_cost': final_solution.weighted_cost,
                      'evaluated_cost': final_solution.evaluated_cost,
                      'evaluated_time': final_solution.evaluated_time,
                      'evaluated_unmet': final_solution.evaluated_unmet,
                      'is_feasible': final_solution.is_feasible,
                      'evaluation_stage1_error': final_solution.evaluation_stage1_error,
                      'evaluation_stage2_error': final_solution.evaluation_stage2_error,
                      'served_customer_details': final_solution.served_customer_details,
                      'cost_history': getattr(final_solution, 'cost_history', []), # Include history if present
                      # Add other relevant history if needed (e.g., avg_cost_history from GA)
                      'avg_cost_history': getattr(final_solution, 'avg_cost_history', []),
                      'current_cost_history': getattr(final_solution, 'current_cost_history', []), # For SA
                      'temperature_history': getattr(final_solution, 'temperature_history', []), # For SA
                 }
                 # Add algorithm-specific params used
                 processed_result_data['algorithm_params'] = algo_specific_params


            elif isinstance(algorithm_raw_result, dict):
                 # If algorithm returns a dictionary (e.g., Greedy or potentially others)
                 logger.info(f"Algorithm {algo_key} returned a dictionary. Re-evaluating for consistency...")
                 # Assume dict contains structure ('stage1_routes', 'stage2_trips', 'served_customer_details')
                 # and potentially pre-calculated metrics. We re-evaluate to ensure consistency.

                 temp_stage1_routes = algorithm_raw_result.get('stage1_routes', {})
                 # Use assignments from the base initial solution (they are fixed)
                 temp_outlet_to_depot_assignments = base_initial_solution.outlet_to_depot_assignments
                 temp_customer_to_outlet_assignments = base_initial_solution.customer_to_outlet_assignments

                 (re_eval_raw_cost, re_eval_time_makespan, re_eval_unmet_demand,
                  re_eval_served_details, re_eval_s1_err, re_eval_s2_err,
                  re_eval_s2_trips_agg) = calculate_total_cost_and_evaluate(
                      stage1_routes=temp_stage1_routes,
                      outlet_to_depot_assignments=temp_outlet_to_depot_assignments,
                      customer_to_outlet_assignments=temp_customer_to_outlet_assignments,
                      problem_data=problem_data,
                      vehicle_params=vehicle_params,
                      drone_params=drone_params,
                      distance_func=haversine,
                      stage2_trip_generator_func=create_heuristic_trips_split_delivery,
                      unmet_demand_penalty=optimization_params['unmet_demand_penalty'],
                      cost_weight=objective_weights['cost_weight'],
                      time_weight=objective_weights['time_weight']
                  )

                 re_eval_weighted_cost = (objective_weights['cost_weight'] * re_eval_raw_cost +
                                          objective_weights['time_weight'] * re_eval_time_makespan +
                                          optimization_params['unmet_demand_penalty'] * re_eval_unmet_demand)

                 # Use locally defined FLOAT_TOLERANCE here
                 re_eval_is_feasible = (not re_eval_s1_err and not re_eval_s2_err and
                                         re_eval_unmet_demand is not None and
                                         not math.isinf(re_eval_unmet_demand) and
                                         not math.isnan(re_eval_unmet_demand) and
                                         re_eval_unmet_demand < FLOAT_TOLERANCE)

                 # Populate the standard result dictionary using re-evaluated values
                 processed_result_data = {
                      'best_solution_structure': { # Store structure
                          'stage1_routes': temp_stage1_routes,
                          'stage2_trips': re_eval_s2_trips_agg, # Use re-evaluated trips
                          'outlet_to_depot_assignments': temp_outlet_to_depot_assignments,
                          'customer_to_outlet_assignments': temp_customer_to_outlet_assignments,
                          },
                      'weighted_cost': re_eval_weighted_cost,
                      'evaluated_cost': re_eval_raw_cost,
                      'evaluated_time': re_eval_time_makespan,
                      'evaluated_unmet': re_eval_unmet_demand,
                      'is_feasible': re_eval_is_feasible,
                      'evaluation_stage1_error': re_eval_s1_err,
                      'evaluation_stage2_error': re_eval_s2_err,
                      'served_customer_details': re_eval_served_details, # Use re-evaluated details
                      # Include history if algorithm provided it in the dict
                      'cost_history': algorithm_raw_result.get('cost_history', []),
                      'avg_cost_history': algorithm_raw_result.get('avg_cost_history', []),
                      'current_cost_history': algorithm_raw_result.get('current_cost_history', []),
                      'temperature_history': algorithm_raw_result.get('temperature_history', []),
                 }
                 # Add algorithm-specific params used
                 processed_result_data['algorithm_params'] = algo_specific_params

            else:
                 # Handle unexpected return type
                 raise TypeError(f"Algorithm '{algo_key}' returned unexpected type: {type(algorithm_raw_result)}")


            # --- Final Checks on Processed Result ---
            if processed_result_data.get('evaluation_stage1_error') or processed_result_data.get('evaluation_stage2_error'):
                logger.warning(f"Algorithm {algo_key} final solution evaluation encountered errors.")
            if processed_result_data.get('weighted_cost') == float('inf'):
                 logger.warning(f"Algorithm {algo_key} final solution resulted in infinite weighted cost.")


            # --- Store Processed Result ---
            algo_results_summary['result_data'] = processed_result_data
            algo_results_summary['status'] = 'Success'
            logger.info(f"Algorithm {algo_key} finished successfully. Final Weighted Cost: {processed_result_data.get('weighted_cost', 'N/A'):.4f}, Feasible: {processed_result_data.get('is_feasible', 'N/A')}")


        except Exception as e:
            error_msg = f"Error during {algo_key} execution or result processing: {e}"
            algo_results_summary['run_error'] = error_msg
            algo_results_summary['status'] = 'Failed'
            logger.error(error_msg, exc_info=True)

        # --- Finalize Algorithm Summary ---
        algo_end_time = pytime.time()
        algo_results_summary['computation_time'] = algo_end_time - algo_start_time
        logger.info(f"--- Algorithm {algo_key} Completed in {algo_results_summary['computation_time']:.4f} seconds (Status: {algo_results_summary['status']}) ---")


    # --- Aggregate Results and Determine Best ---
    logger.info("Aggregating results and determining best solutions...")
    valid_algo_results = [
        (key, res['result_data'])
        for key, res in overall_results['results_by_algorithm'].items()
        if res['status'] == 'Success' and res['result_data'] is not None
    ]

    if not valid_algo_results:
        overall_results['overall_status'] = 'No Valid Results'
        overall_results['error_message'] = "No selected algorithm produced a valid result."
        logger.warning(overall_results['error_message'])
    else:
        overall_results['overall_status'] = 'Success' # At least one algorithm succeeded

        # Find best overall by weighted cost
        best_algo_key_overall = None
        min_weighted_cost = float('inf')
        for key, data in valid_algo_results:
             cost = data.get('weighted_cost', float('inf'))
             # Ensure cost is finite before comparison
             if math.isfinite(cost) and cost < min_weighted_cost:
                  min_weighted_cost = cost
                  best_algo_key_overall = key
        overall_results['best_algorithm_key'] = best_algo_key_overall
        overall_results['best_weighted_cost'] = min_weighted_cost

        # Find best *feasible* solution
        best_feasible_algo_key = None
        min_feasible_weighted_cost = float('inf')
        for key, data in valid_algo_results:
            if data.get('is_feasible', False):
                 cost = data.get('weighted_cost', float('inf'))
                 # Ensure cost is finite before comparison
                 if math.isfinite(cost) and cost < min_feasible_weighted_cost:
                      min_feasible_weighted_cost = cost
                      best_feasible_algo_key = key
        overall_results['fully_served_best_key'] = best_feasible_algo_key
        overall_results['fully_served_best_cost'] = min_feasible_weighted_cost

        logger.info(f"Best overall solution: {best_algo_key_overall} (Cost: {min_weighted_cost:.4f})")
        if best_feasible_algo_key:
            logger.info(f"Best feasible solution: {best_feasible_algo_key} (Cost: {min_feasible_weighted_cost:.4f})")
        else:
            logger.warning("No feasible solution found among successful algorithm runs.")


    # --- Post-processing: Generate Maps and Reports ---
    logger.info("--- Starting Post-processing (Maps & Reports) ---")
    # Pass necessary data to map/report generators

    # Extract initial demands list for report generator
    initial_demands_list = problem_data.get('demands', [])

    for algo_key, algo_result_summary in overall_results['results_by_algorithm'].items():
         if algo_result_summary['status'] == 'Success' and algo_result_summary.get('result_data'):
             result_data = algo_result_summary['result_data']
             solution_structure_dict = result_data.get('best_solution_structure') # Use the stored dict structure

             # Generate Map (if available)
             if _VISUALIZATION_AVAILABLE and solution_structure_dict:
                 try:
                     map_filename = os.path.join(run_output_dir, "maps", f"{algo_key}_route_map.html")
                     logger.info(f"Generating map for {algo_key} -> {map_filename}")
                     # Pass the necessary components for the map generator
                     generated_map_path = generate_folium_map(
                         problem_data=problem_data, # Pass original locations/demands
                         solution_structure=solution_structure_dict, # Pass the solution structure dict
                         vehicle_params=vehicle_params,
                         drone_params=drone_params,
                         output_path=map_filename,
                         map_title=f"{algo_key.replace('_', ' ').title()} Route Map"
                     )
                     if generated_map_path:
                          algo_result_summary['map_path'] = generated_map_path
                     else:
                          # Log if map generation returned None, indicating internal failure
                          err_msg = "Map generation function returned None."
                          algo_result_summary['map_generation_error'] = err_msg
                          logger.warning(f"Map generation failed for {algo_key}: {err_msg}")
                 except Exception as e:
                     error_msg = f"Error generating map for {algo_key}: {e}"
                     algo_result_summary['map_generation_error'] = error_msg
                     logger.error(error_msg, exc_info=False) # Log error but continue

             # Generate Report (if available)
             if _REPORTING_AVAILABLE:
                 try:
                     report_filename = os.path.join(run_output_dir, "reports", f"{algo_key}_report.txt")
                     logger.info(f"Generating report for {algo_key} -> {report_filename}")
                     # Call generate_delivery_report with the required arguments
                     # The result_data dictionary created earlier should contain the necessary info
                     report_content = generate_delivery_report(
                         algorithm_name=algo_key.replace('_', ' ').title(),
                         result_data=result_data, # Pass the processed result dictionary
                         points_data=problem_data.get('locations', {}), # Pass original locations
                         initial_demands_list=initial_demands_list # Pass the initial demands
                     )
                     # Check if report generator returned an error message
                     if report_content.startswith("Error:") or report_content.startswith("CRITICAL WARNING"):
                          raise RuntimeError(f"Report generator returned an error/warning: {report_content[:200]}...")

                     with open(report_filename, 'w', encoding='utf-8') as f:
                          f.write(report_content)
                     algo_result_summary['report_path'] = report_filename
                 except Exception as e:
                     error_msg = f"Error generating report for {algo_key}: {e}"
                     algo_result_summary['report_generation_error'] = error_msg
                     logger.error(error_msg, exc_info=False) # Log error but continue


    # --- Finalize Overall Results ---
    run_end_time = pytime.time()
    overall_results['total_computation_time'] = run_end_time - run_start_time
    logger.info(f"--- Optimization Run Finished in {overall_results['total_computation_time']:.4f} seconds ---")

    return overall_results


# --- Standalone Testing Block (Optional) ---
if __name__ == '__main__':
    """
    Standalone execution block for testing the run_optimization function.
    Uses dummy problem data and requires algorithms to be importable.
    Note: This will run the actual algorithms if they are available.
    """
    logger.info("Running core/route_optimizer.py in standalone test mode.")

    # --- Create Dummy Problem Data ---
    try:
        logger.info("--- Creating Dummy Problem Data for Optimizer Test ---")
        dummy_locations = {
            'logistics_centers': [(34.0, -118.0), (34.1, -118.2)],
            'sales_outlets': [(34.05, -118.1), (34.02, -118.05), (34.15, -118.3), (33.95, -118.15)],
            'customers': [(34.06, -118.11), (34.05, -118.09), (34.00, -118.06), (34.16, -118.31), (34.14, -118.28), (33.96, -118.16)]
        }
        dummy_demands = [10.0, 15.0, 8.0, 20.0, 12.0, 5.0]

        dummy_problem_data_opt = {'locations': dummy_locations, 'demands': dummy_demands}
        dummy_vehicle_params_opt = {'payload': 200.0, 'cost_per_km': 1.5, 'speed_kmph': 60.0}
        dummy_drone_params_opt = {'payload': 5.0, 'max_flight_distance_km': 5.0, 'cost_per_km': 0.8, 'speed_kmph': 80.0}

        dummy_optimization_params = {
            'unmet_demand_penalty': 10000.0,
            'output_dir': 'output/optimizer_standalone_test', # Use a specific test dir
            'genetic_algorithm_params': {'population_size': 10, 'num_generations': 5, 'mutation_rate': 0.15, 'crossover_rate': 0.8, 'elite_count': 1, 'tournament_size': 3},
            'simulated_annealing_params': {'initial_temperature': 500, 'cooling_rate': 0.98, 'max_iterations': 50, 'min_temperature': 0.1},
            'pso_optimizer_params': {'num_particles': 10, 'max_iterations': 5, 'inertia_weight': 0.7, 'cognitive_weight': 1.5, 'social_weight': 1.5},
            'greedy_heuristic_params': {},
        }
        # Ensure selected algos exist in registry for test
        registered_algos = list(ALGORITHM_REGISTRY.keys())
        dummy_selected_algorithms = [algo for algo in ['greedy_heuristic', 'genetic_algorithm'] if algo in registered_algos]
        if not dummy_selected_algorithms:
             logger.error("No registered algorithms available for standalone test. Exiting.")
             sys.exit(1)

        dummy_objective_weights = {'cost_weight': 0.7, 'time_weight': 0.3}

        logger.info("Dummy data and parameters created for optimizer test.")

    except Exception as e:
        logger.error(f"Error creating dummy data for optimizer test: {e}", exc_info=True)
        sys.exit(1)

    # --- Run the main optimization function ---
    logger.info(f"--- Running run_optimization function with dummy data ({dummy_selected_algorithms}) ---")
    try:
        optimizer_results = run_optimization(
            problem_data=dummy_problem_data_opt,
            vehicle_params=dummy_vehicle_params_opt,
            drone_params=dummy_drone_params_opt,
            optimization_params=dummy_optimization_params,
            selected_algorithm_keys=dummy_selected_algorithms,
            objective_weights=dummy_objective_weights
        )

        # --- Print Results Summary ---
        # Use Python's print for simple output in standalone test
        print("\n" + "="*30 + " Optimization Results Summary " + "="*30)
        print(f"Overall Status: {optimizer_results.get('overall_status')}")
        if optimizer_results.get('error_message'):
            print(f"Overall Error Message: {optimizer_results.get('error_message')}")
        print(f"Output Directory: {optimizer_results.get('output_directory')}")
        print(f"Total Computation Time: {optimizer_results.get('total_computation_time', 0.0):.4f} seconds")
        print(f"Best overall algorithm: {optimizer_results.get('best_algorithm_key', 'N/A')} (Weighted Cost: {optimizer_results.get('best_weighted_cost', float('inf')):.4f})")
        print(f"Best feasible algorithm: {optimizer_results.get('fully_served_best_key', 'N/A')} (Weighted Cost: {optimizer_results.get('fully_served_best_cost', float('inf')):.4f})")

        print("\nResults by Algorithm:")
        results_by_algo = optimizer_results.get('results_by_algorithm', {})
        if results_by_algo:
             sorted_algo_keys = sorted(results_by_algo.keys())
             for algo_key in sorted_algo_keys:
                 res = results_by_algo[algo_key]
                 print(f"\n  --- {res.get('algorithm_name', algo_key).replace('_', ' ').title()} ---")
                 print(f"    Status: {res.get('status', 'Unknown')}")
                 print(f"    Run Time: {res.get('computation_time', 0.0):.4f} seconds")
                 if res.get('run_error'):
                     print(f"    Run Error: {res.get('run_error')}")
                 if res.get('result_data'):
                     data = res['result_data']
                     print(f"    Weighted Cost: {data.get('weighted_cost', float('inf')):.4f}")
                     print(f"    Raw Cost: {data.get('evaluated_cost', float('inf')):.2f}")
                     print(f"    Time (Makespan): {data.get('evaluated_time', float('inf')):.2f}")
                     print(f"    Unmet Demand: {data.get('evaluated_unmet', float('inf')):.2f}")
                     print(f"    Feasible: {data.get('is_feasible', False)}")
                     print(f"    Map Path: {res.get('map_path', 'N/A')}")
                     print(f"    Report Path: {res.get('report_path', 'N/A')}")
                     if res.get('map_generation_error'): print(f"    Map Error: {res.get('map_generation_error')}")
                     if res.get('report_generation_error'): print(f"    Report Error: {res.get('report_generation_error')}")
        else:
             print("  No algorithm results available.")
        print("="*80)

    except Exception as e:
        logger.error(f"An unexpected error occurred during run_optimization test execution: {e}", exc_info=True)

    logger.info("Standalone test finished.")