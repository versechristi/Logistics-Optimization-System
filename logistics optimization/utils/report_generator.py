# utils/report_generator.py
# -*- coding: utf-8 -*-
"""
Generates detailed, formatted text reports summarizing the results of a
logistics optimization algorithm run for the MD-2E-VRPSD problem.
"""

import math
import time
import textwrap
import sys
from typing import Dict, List, Any, Optional, Tuple

# Configure logging for this module
import logging
logger = logging.getLogger(__name__)

# Define a small tolerance for floating-point comparisons (e.g., checking if demand is zero)
FLOAT_TOLERANCE_REPORT = 1e-6


def format_float(value: Any, precision: int = 4) -> str:
    """
    Safely formats a numerical value to a specified precision string.
    Handles None, NaN, Infinity, and non-numeric types gracefully.

    Args:
        value: The value to format (int, float, None, or other).
        precision: The number of decimal places for floats.

    Returns:
        The formatted string representation.
    """
    if isinstance(value, (int, float)):
        if math.isnan(value):
            return "NaN"
        if value == float('inf'):
            return "Infinity"
        if value == float('-inf'):
            return "-Infinity"
        # Standard formatting for valid numbers
        format_string = "{:." + str(precision) + "f}"
        try:
            # Handle very small numbers that might format to "-0.0000"
            formatted = format_string.format(value)
            if formatted.startswith('-') and float(formatted) == 0.0:
                return format_string.format(0.0) # Return "0.0000" instead of "-0.0000"
            return formatted
        except Exception: # Catch potential formatting errors with unusual floats
            return str(value) # Fallback to simple string conversion
    elif value is None:
        return "N/A" # Not Available
    else:
        # Attempt to convert other types to string
        try:
            return str(value)
        except Exception:
            return "Invalid Data"


def generate_delivery_report(
    algorithm_name: str,
    result_data: Dict[str, Any],
    points_data: Dict[str, List[Tuple[float, float]]],
    initial_demands_list: List[float]
    ) -> str:
    """
    Generates a detailed delivery report string for a given algorithm's result,
    adapted for the MD-2E-VRPSD solution structure.

    Args:
        algorithm_name: Display name of the algorithm.
        result_data: The processed result dictionary for this algorithm from route_optimizer.
                     Expected keys include: 'weighted_cost', 'evaluated_cost', 'evaluated_time',
                     'evaluated_unmet', 'is_feasible', 'evaluation_stage1_error',
                     'evaluation_stage2_error', 'best_solution_structure' (dict with
                     'stage1_routes', 'stage2_trips', 'outlet_to_depot_assignments',
                     'customer_to_outlet_assignments'), 'served_customer_details',
                     'algorithm_params'.
        points_data: Dictionary containing original location coordinates:
                     'logistics_centers', 'sales_outlets', 'customers'.
        initial_demands_list: The list of initial demands for all customers.

    Returns:
        A formatted report string. Returns an error message string on critical failure.
    """
    report_lines = []
    report_gen_start_time = time.time()
    logger.info(f"Generating report for algorithm: {algorithm_name}")

    # --- Input Validation ---
    if not isinstance(result_data, dict):
        return f"Error: Invalid 'result_data' provided for {algorithm_name} (Expected dict, got {type(result_data)})."
    if not isinstance(points_data, dict) or not all(k in points_data for k in ['logistics_centers', 'sales_outlets', 'customers']):
        return f"Error: Invalid 'points_data' structure for {algorithm_name} (Missing location keys)."
    if not isinstance(initial_demands_list, list):
        # Handle case where demands might be None if data generation failed or no customers
        if initial_demands_list is None and not points_data.get('customers'):
             initial_demands_list = [] # Valid case if no customers
             logger.debug("Initial demands list is None, but no customers exist.")
        else:
             # If demands list is None/invalid but customers exist, it's an error
             logger.error(f"Invalid 'initial_demands_list' provided for {algorithm_name} (Expected list, got {type(initial_demands_list)}). Customer data exists.")
             # Proceed with empty list but log warning heavily
             initial_demands_list = []
             report_lines.append("CRITICAL WARNING: Initial demands list was invalid or None. Customer fulfillment status will be inaccurate.")


    report_lines.append("=" * 70)
    report_lines.append(f" Optimization Report: {algorithm_name}")
    report_lines.append("=" * 70)

    # --- Overall Summary ---
    report_lines.append("\n--- Overall Summary ---")
    weighted_cost = result_data.get('weighted_cost')
    raw_cost = result_data.get('evaluated_cost')
    total_time = result_data.get('evaluated_time') # Makespan
    runtime = result_data.get('computation_time') # Added by route_optimizer summary
    final_unmet = result_data.get('evaluated_unmet')
    is_feasible = result_data.get('is_feasible', False)
    eval_s1_error = result_data.get('evaluation_stage1_error', False)
    eval_s2_error = result_data.get('evaluation_stage2_error', False)

    report_lines.append(f"Feasibility Status: {'Feasible (All demand met)' if is_feasible else 'Infeasible (Unmet demand or errors)'}")
    if eval_s1_error: report_lines.append("  >> Warning: Stage 1 evaluation encountered errors.")
    if eval_s2_error: report_lines.append("  >> Warning: Stage 2 evaluation encountered errors.")
    report_lines.append(f"Final Weighted Cost:  {format_float(weighted_cost, 4)}")
    report_lines.append(f"  Raw Transport Cost: {format_float(raw_cost, 2)}")
    report_lines.append(f"  Total Time (Makespan): {format_float(total_time, 3)} hrs")
    report_lines.append(f"  Final Unmet Demand: {format_float(final_unmet, 4)}")
    report_lines.append(f"Algorithm Runtime:    {format_float(runtime, 3)} seconds")

    # --- Extract Solution Structure ---
    solution_structure = result_data.get('best_solution_structure')
    depot_coords = points_data.get('logistics_centers', [])
    outlet_coords = points_data.get('sales_outlets', [])
    customer_coords = points_data.get('customers', [])

    stage1_routes = {}
    stage2_trips = {}
    outlet_assignments = {}
    customer_assignments = {}

    if solution_structure and isinstance(solution_structure, dict):
        stage1_routes = solution_structure.get('stage1_routes', {})
        stage2_trips = solution_structure.get('stage2_trips', {})
        outlet_assignments = solution_structure.get('outlet_to_depot_assignments', {})
        customer_assignments = solution_structure.get('customer_to_outlet_assignments', {})
        if not stage1_routes and not stage2_trips:
             report_lines.append("\nWarning: Solution structure exists but contains no routes or trips.")
    else:
        report_lines.append("\nError: Solution details (routes/trips) not found in result data.")
        # Bail out if no solution structure? Or continue reporting metrics? Continue for now.


    # --- Stage 1 Routes Details ---
    report_lines.append("\n--- Stage 1 Routes (Depot -> Outlets -> Depot) ---")
    if not stage1_routes or not isinstance(stage1_routes, dict):
        report_lines.append("  No valid Stage 1 route data found.")
    elif not any(routes for routes in stage1_routes.values()): # Check if all route lists are empty
         report_lines.append("  No Stage 1 routes were generated.")
    else:
         num_depots = len(depot_coords)
         for depot_idx in sorted(stage1_routes.keys()):
             if not (0 <= depot_idx < num_depots):
                 report_lines.append(f"  Depot {depot_idx}: Invalid Index!")
                 continue

             depot_loc_str = f"({format_float(depot_coords[depot_idx][0], 4)}, {format_float(depot_coords[depot_idx][1], 4)})" if depot_coords[depot_idx] else "(Unknown Location)"
             report_lines.append(f"\n  Depot {depot_idx} {depot_loc_str}:")

             routes_for_depot = stage1_routes[depot_idx]
             if not routes_for_depot or not isinstance(routes_for_depot, list):
                 report_lines.append(f"    - No routes assigned or invalid data type ({type(routes_for_depot)}).")
                 continue

             # Assuming routes_for_depot is a list of lists (each inner list is a route)
             # If the structure is just one list per depot, adjust accordingly.
             # Based on GA/SA/PSO structure, it's likely one list per depot.
             # Let's assume it's a single list of outlet indices for this depot.
             # If route_optimizer splits into multiple routes per depot later, this needs update.
             # Current implementation seems to be single list per depot.
             route_seq = routes_for_depot
             if not isinstance(route_seq, list):
                  report_lines.append(f"    - Invalid route sequence data type ({type(route_seq)}).")
             elif not route_seq:
                  report_lines.append("    - Route: (Empty - No outlets visited)")
             else:
                  # Format sequence: D -> O1 -> O2 -> ... -> D
                  route_str = " -> ".join([f"O{o_idx}" for o_idx in route_seq])
                  # TODO: Optionally calculate/display cost/time for this specific route if needed
                  report_lines.append(f"    - Route: D{depot_idx} -> {route_str} -> D{depot_idx}")

    # --- Stage 2 Trips Details ---
    report_lines.append("\n--- Stage 2 Trips (Outlet -> Customers -> Outlet) ---")
    if not stage2_trips or not isinstance(stage2_trips, dict):
        report_lines.append("  No valid Stage 2 trip data found.")
    elif not stage2_trips: # Check if dict is empty
        report_lines.append("  No Stage 2 trips were generated for any outlet.")
    else:
         num_outlets = len(outlet_coords)
         for outlet_idx in sorted(stage2_trips.keys()):
              if not (0 <= outlet_idx < num_outlets):
                  report_lines.append(f"  Outlet {outlet_idx}: Invalid Index!")
                  continue

              outlet_loc_str = f"({format_float(outlet_coords[outlet_idx][0], 4)}, {format_float(outlet_coords[outlet_idx][1], 4)})" if outlet_coords[outlet_idx] else "(Unknown Location)"
              report_lines.append(f"\n  Outlet {outlet_idx} {outlet_loc_str}:")

              trips_from_outlet = stage2_trips[outlet_idx]
              if not trips_from_outlet or not isinstance(trips_from_outlet, list):
                  report_lines.append(f"    - No trips assigned or invalid data type ({type(trips_from_outlet)}).")
                  continue

              for i, trip_info in enumerate(trips_from_outlet):
                   if not isinstance(trip_info, dict):
                       report_lines.append(f"    - Trip {i+1}: Invalid trip info format ({type(trip_info)}).")
                       continue

                   trip_type = trip_info.get('type', 'Unknown').capitalize()
                   trip_route = trip_info.get('route', []) # List of customer indices
                   trip_load = trip_info.get('load', None) # Optional: Load carried
                   trip_cost = trip_info.get('cost', None) # Optional: Cost of this trip
                   trip_time = trip_info.get('time', None) # Optional: Time for this trip

                   if not trip_route or not isinstance(trip_route, list):
                        report_lines.append(f"    - Trip {i+1} ({trip_type}): Invalid route data ({type(trip_route)}).")
                        continue

                   # Format trip sequence: O -> C1 -> C2 -> ... -> O
                   trip_str = " -> ".join([f"C{c_idx}" for c_idx in trip_route])
                   details_str = f"(Type: {trip_type}"
                   if trip_load is not None: details_str += f", Load: {format_float(trip_load, 2)}"
                   if trip_cost is not None: details_str += f", Cost: {format_float(trip_cost, 2)}"
                   if trip_time is not None: details_str += f", Time: {format_float(trip_time, 3)}"
                   details_str += ")"

                   report_lines.append(f"    - Trip {i+1}: O{outlet_idx} -> {trip_str} -> O{outlet_idx} {details_str}")

    # --- Customer Fulfillment Details ---
    report_lines.append("\n--- Customer Demand Fulfillment Status ---")
    served_details = result_data.get('served_customer_details')
    num_customers_in_points = len(customer_coords) if customer_coords else 0
    num_demands = len(initial_demands_list)

    # Basic consistency check
    if num_customers_in_points != num_demands and num_customers_in_points > 0:
        report_lines.append(f"  Warning: Mismatch between customer locations ({num_customers_in_points}) and initial demands ({num_demands}). Reporting based on demands list length.")

    if num_demands > 0:
        fully_served_count = 0
        partially_served_count = 0
        unserved_count = 0
        total_initial_demand = 0.0
        total_remaining_demand = 0.0

        customer_status_lines = [] # Build table rows
        customer_status_lines.append("  CustIdx | Initial Demand | Final Remaining | Status")
        customer_status_lines.append("  --------|----------------|-----------------|--------")

        # Iterate using the length of the initial demands list
        for cust_idx in range(num_demands):
            initial_demand = initial_demands_list[cust_idx]
            total_initial_demand += initial_demand

            # Get served details if available
            details = served_details.get(cust_idx) if isinstance(served_details, dict) else None
            remaining = initial_demand # Default: assume unserved if no details
            status = "Unserved" # Default status

            if details and isinstance(details, dict):
                remaining = details.get('remaining', initial_demand) # Get remaining from dict
                # Clamp remaining demand >= 0 for reporting
                remaining = max(0.0, remaining)

                # Determine status based on remaining vs initial (using tolerance)
                if abs(remaining) < FLOAT_TOLERANCE_REPORT:
                    status = "Served"
                    fully_served_count += 1
                    remaining = 0.0 # Ensure exact zero for display
                elif initial_demand > FLOAT_TOLERANCE_REPORT and remaining < initial_demand - FLOAT_TOLERANCE_REPORT:
                    status = "Partial"
                    partially_served_count += 1
                elif abs(remaining - initial_demand) < FLOAT_TOLERANCE_REPORT:
                    status = "Unserved" # Should already be the default
                    unserved_count += 1
                    remaining = initial_demand # Ensure exact initial for display
                else: # Should not happen if remaining is clamped >= 0
                    status = "ERROR"
                    unserved_count += 1 # Count as unserved if status error

            else:
                # If no details dict, assume unserved
                unserved_count += 1
                if not isinstance(served_details, dict):
                     status = "No Detail Data" # Indicate details dict was missing/invalid

            total_remaining_demand += remaining # Sum up actual remaining

            customer_status_lines.append(
                f"  {str(cust_idx).ljust(7)} | "
                f"{format_float(initial_demand, 2).rjust(14)} | "
                f"{format_float(remaining, 4).rjust(15)} | "
                f"{status}"
            )

        # Add summary statistics
        report_lines.append(f"\n  Total Customers: {num_demands}")
        if num_demands > 0:
             report_lines.append(f"    Fully Served:   {fully_served_count} ({fully_served_count/num_demands:.1%})")
             report_lines.append(f"    Partially Served: {partially_served_count} ({partially_served_count/num_demands:.1%})")
             report_lines.append(f"    Unserved:       {unserved_count} ({unserved_count/num_demands:.1%})")
             report_lines.append(f"  Total Initial Demand: {format_float(total_initial_demand, 2)}")
             report_lines.append(f"  Total Remaining Demand: {format_float(total_remaining_demand, 4)}") # Should match 'evaluated_unmet'
        report_lines.append("\n  --- Detailed Status ---")
        report_lines.extend(customer_status_lines)
        report_lines.append("") # Blank line

    else:
        report_lines.append("  No customer demand data provided or empty list.\n")

    # --- Parameters Used ---
    report_lines.append("\n--- Parameters Used ---")
    # Include algorithm-specific parameters if available
    params = result_data.get('algorithm_params')
    if params and isinstance(params, dict):
        report_lines.append("  Algorithm Parameters:")
        if not params:
             report_lines.append("    (No specific parameters recorded)")
        else:
             param_lines = [f"    - {key}: {value}" for key, value in sorted(params.items())]
             wrapped_lines = []
             for line in param_lines:
                 # Wrap lines longer than, e.g., 80 chars, keeping indentation
                 wrapped = textwrap.wrap(line, width=80, initial_indent="", subsequent_indent="      ")
                 wrapped_lines.extend(wrapped)
             report_lines.extend(wrapped_lines)
    else:
        report_lines.append(f"  Algorithm Parameter data not available or invalid type ({type(params)}).")

    # TODO: Optionally add Vehicle, Drone, Objective parameters here if needed,
    # retrieving them from the main results structure passed to the optimizer
    # if route_optimizer adds them to result_data. For now, focusing on algo params.


    # --- Report Footer ---
    report_gen_end_time = time.time()
    report_lines.append(f"\nReport generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Report generation time: {format_float(report_gen_end_time - report_gen_start_time, 3)} seconds.")
    report_lines.append("=" * 70)
    report_lines.append(f"--- End of Report: {algorithm_name} ---")
    report_lines.append("=" * 70)


    return "\n".join(report_lines) + "\n" # Ensure trailing newline


# --- Optional Main Execution Block for Standalone Testing ---
if __name__ == '__main__':
    """
    Standalone execution block for testing the report generator function.
    """
    logger.info("Running report_generator.py in standalone test mode.")

    # --- Create Dummy Data for Testing ---
    dummy_algo_name = "Test Algorithm (Dummy)"

    # Mimic the structure expected in result_data from route_optimizer
    dummy_result_data = {
        'weighted_cost': 1234.5678,
        'evaluated_cost': 987.65,
        'evaluated_time': 3.55,
        'evaluated_unmet': 5.0, # Example with unmet demand
        'is_feasible': False,
        'evaluation_stage1_error': False,
        'evaluation_stage2_error': False,
        'computation_time': 15.2, # Added by route_optimizer summary
        'best_solution_structure': {
             'stage1_routes': {0: [1, 0], 1: [2]}, # D0->O1->O0->D0, D1->O2->D1
             'stage2_trips': {
                  0: [{'type': 'vehicle', 'route': [0], 'load': 8.0, 'cost': 10.5, 'time': 0.2}], # O0 -> C0
                  1: [], # O1 has no trips
                  2: [{'type': 'drone', 'route': [1, 3], 'load': 12.0, 'cost': 5.2, 'time': 0.1}, # O2 -> C1, C3 (drone)
                      {'type': 'vehicle', 'route': [2], 'load': 7.0, 'cost': 8.8, 'time': 0.15}] # O2 -> C2 (vehicle)
             },
            'outlet_to_depot_assignments': {0: 0, 1: 0, 2: 1}, # Example assignments
            'customer_to_outlet_assignments': {0: 0, 1: 2, 2: 2, 3: 2} # Example assignments
        },
        'served_customer_details': {
             0: {'initial': 10.0, 'satisfied': 10.0, 'remaining': 0.0, 'status': 'Served'},
             1: {'initial': 15.0, 'satisfied': 12.0, 'remaining': 3.0, 'status': 'Partial'}, # Partially served
             2: {'initial': 8.0, 'satisfied': 8.0, 'remaining': 0.0, 'status': 'Served'},
             3: {'initial': 20.0, 'satisfied': 18.0, 'remaining': 2.0, 'status': 'Partial'}, # Partially served
             # Customer 4 is missing - assumed unserved
        },
        'algorithm_params': {'param_a': 10, 'param_b': 'test_value', 'long_param': list(range(20))}
    }

    dummy_points_data = {
        'logistics_centers': [(40.0, -74.0), (40.1, -74.2)],
        'sales_outlets': [(40.05, -74.1), (40.02, -74.05), (40.15, -74.3)],
        'customers': [(40.06, -74.11), (40.04, -74.09), (40.00, -74.06), (40.16, -74.31), (40.14, -74.28)] # 5 customers
    }
    dummy_initial_demands = [10.0, 15.0, 8.0, 20.0, 12.0] # Match customer count


    # --- Generate the report ---
    report_output = generate_delivery_report(
        algorithm_name=dummy_algo_name,
        result_data=dummy_result_data,
        points_data=dummy_points_data,
        initial_demands_list=dummy_initial_demands
    )

    # --- Print the report ---
    print("\n" + "="*30 + " Generated Report Output " + "="*30)
    print(report_output)
    print("="*80)

    # --- Test with missing data ---
    print("\n--- Testing with Missing Data ---")
    report_missing = generate_delivery_report(
        algorithm_name="Missing Data Test",
        result_data={'computation_time': 0.1}, # Minimal data
        points_data=dummy_points_data,
        initial_demands_list=dummy_initial_demands
    )
    print(report_missing)
    print("="*80)

    # --- Test with invalid demands list ---
    print("\n--- Testing with Invalid Demands List ---")
    report_invalid_demands = generate_delivery_report(
        algorithm_name="Invalid Demands Test",
        result_data=dummy_result_data,
        points_data=dummy_points_data,
        initial_demands_list=None # Pass None explicitly
    )
    print(report_invalid_demands)
    print("="*80)


    logger.info("Standalone report generator test finished.")