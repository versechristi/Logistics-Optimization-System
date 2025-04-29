# data/data_generator.py
# -*- coding: utf-8 -*-
"""
Module for generating synthetic geographical locations and customer demands
for the Multi-Depot, Two-Echelon Vehicle Routing Problem with Drones
and Split Deliveries (MD-2E-VRPSD).

Provides functions to generate logistics centers, sales outlets, and customers
with various spatial distributions and assign random demands to customers.
Designed to produce data structures compatible with the core optimization
modules.
"""

import math
import os
import random
import numpy as np
import pandas as pd
import sys
import traceback
import warnings

# Constants for Earth radius in kilometers
EARTH_RADIUS_KM = 6371.0

def generate_locations(num_logistics_centers: int, num_sales_outlets: int, num_customers: int,
                       center_latitude: float, center_longitude: float, radius_km: float,
                       use_solomon_like_distribution: bool = False) -> dict | None:
    """
    Generates geographical locations for logistics centers, sales outlets, and customers
    within a specified radius around a center point.

    Locations are generated based on either a uniform or a clustered (Solomon-like)
    distribution within a circular area. The generation method is a simplification
    for VRP instances and assumes a relatively flat projection within the given radius.

    Args:
        num_logistics_centers (int): The number of logistics centers (depots) to generate. Must be non-negative.
        num_sales_outlets (int): The number of sales outlets (cross-docking points) to generate. Must be non-negative.
        num_customers (int): The number of customers to generate. Must be non-negative.
        center_latitude (float): The latitude of the geographical center for generation (in degrees).
        center_longitude (float): The longitude of the geographical center for generation (in degrees).
        radius_km (float): The maximum radial distance (in kilometers) from the center
                           within which locations will be generated. Must be non-negative.
        use_solomon_like_distribution (bool): If True, generates customer locations with clustering
                                              patterns reminiscent of Solomon benchmark
                                              instances (e.g., customers clustered around center).
                                              Logistics centers and sales outlets are typically
                                              generated more uniformly regardless of this flag
                                              for a realistic multi-echelon setup.
                                              If False, generates locations uniformly for all types.

    Returns:
        dict | None: A dictionary containing lists of coordinates (lat, lon) tuples for each
              entity type:
              {'logistics_centers': [(lat, lon), ...],
               'sales_outlets': [(lat, lon), ...],
               'customers': [(lat, lon), ...]}.
              Returns None if input validation fails or if generation encounters an error.
    """
    # --- Input Validation ---
    if not all(isinstance(n, int) and n >= 0 for n in [num_logistics_centers, num_sales_outlets, num_customers]):
        warnings.warn("Invalid input: Number of entities must be non-negative integers.")
        return None
    if not isinstance(center_latitude, (int, float)) or not isinstance(center_longitude, (int, float)):
         warnings.warn("Invalid input: Center coordinates must be numeric.")
         return None
    # Basic range check for coordinates - more for warning than strict validation
    if not (-90 <= center_latitude <= 90) or not (-180 <= center_longitude <= 180):
        warnings.warn(f"Input Warning: Center coordinates ({center_latitude}, {center_longitude}) seem outside standard geographical ranges.")

    if not isinstance(radius_km, (int, float)) or radius_km < 0:
        warnings.warn("Invalid input: Radius must be a non-negative number.")
        return None
    if not isinstance(use_solomon_like_distribution, bool):
         warnings.warn("Invalid input: use_solomon_like_distribution must be a boolean.")
         return None

    generated_points = {
        'logistics_centers': [],
        'sales_outlets': [],
        'customers': []
    }

    # --- Helper function to generate a single point within a circle ---
    def _generate_point_in_circle(center_lat, center_lon, max_radius, distribution_type='uniform'):
        """
        Generates a random point within a circle of given radius centered at (center_lat, center_lon).
        Uses different strategies for radius distribution to mimic uniform or clustered patterns.
        Converts polar coordinates (distance, angle) to latitude and longitude offsets.

        Args:
            center_lat (float): Center latitude (degrees).
            center_lon (float): Center longitude (degrees).
            max_radius (float): Maximum radius for generation (km).
            distribution_type (str): 'uniform' for uniform area distribution,
                                     'solomon-like' for a simple center-biased distribution.

        Returns:
            tuple: (latitude, longitude) of the generated point.
            Returns None if calculation fails (e.g., at poles).
        """
        # Generate random radial distance based on distribution type
        if distribution_type == 'uniform':
            # For uniform distribution over the *area* of the circle,
            # the radius should be sampled from a distribution where P(r) ~ r.
            # This is achieved by sampling uniform from [0, max_radius^2] and taking the sqrt.
            r = max_radius * math.sqrt(random.random())
        elif distribution_type == 'solomon-like':
             # A simple method to bias points towards the center: sample radius
             # from a non-uniform distribution, e.g., power distribution (r^2)
             # or just sampling from a smaller average radius.
             # This example uses r = max_radius * random.random()**power (power > 1 for center bias)
             # A simple random.random()^2 gives more bias towards the center than uniform.
             r = max_radius * (random.random()**2) # More likely smaller radius values
        else:
             warnings.warn(f"Unknown distribution type '{distribution_type}'. Using uniform for point generation.")
             r = max_radius * math.sqrt(random.random()) # Fallback to uniform

        theta = random.uniform(0, 2 * math.pi) # Angle in radians

        # Convert polar (r, theta) to cartesian-like offsets (dx, dy)
        # dx is offset in East-West direction (approx km)
        # dy is offset in North-South direction (approx km)
        dx = r * math.cos(theta)
        dy = r * math.sin(theta)

        # Convert km offsets to degrees offset
        # 1 degree of latitude is approximately 111 km
        # 1 degree of longitude is approximately 111 km * cos(latitude)
        # We use the center latitude for the longitude conversion factor as a simplification.
        # This approximation is less accurate over large areas.
        delta_lat_deg = dy / 111.0

        # Handle potential division by zero or near-zero at poles
        cos_center_lat = math.cos(math.radians(center_lat))
        if abs(cos_center_lat) < 1e-6: # Check if close to poles
             if dx != 0:
                  warnings.warn("Attempted to generate longitude offset at or near poles. Longitude calculation is problematic.")
                  # Depending on requirements, might need to handle this differently (e.g., special case for polar regions)
                  # For VRP, center is unlikely to be at the pole.
             delta_lon_deg = 0.0 # At pole, any east-west movement is just rotation
        else:
             delta_lon_deg = dx / (111.0 * cos_center_lat)

        new_lat = center_lat + delta_lat_deg
        new_lon = center_lon + delta_lon_deg

        # Wrap longitude around -180 to 180 degrees
        new_lon = (new_lon + 180) % 360 - 180
         # Clamp latitude to valid range [-90, 90]
        new_lat = max(-90.0, min(90.0, new_lat))

        if math.isnan(new_lat) or math.isnan(new_lon) or math.isinf(new_lat) or math.isinf(new_lon):
             warnings.warn(f"Generated NaN or Inf coordinates: ({new_lat}, {new_lon}). Skipping point.")
             return None # Return None for invalid generated point

        return (new_lat, new_lon)

    # --- Generate Locations for each entity type ---
    try:
        print(f"Generating {num_logistics_centers} logistics centers within {radius_km} km radius...")
        for _ in range(num_logistics_centers):
            # Logistics centers are often positioned strategically or uniformly within the overall area.
            # Using uniform distribution here within the given radius.
            point = _generate_point_in_circle(center_latitude, center_longitude, radius_km, distribution_type='uniform')
            if point:
                 generated_points['logistics_centers'].append(point)
            else:
                 warnings.warn("Failed to generate a valid location for a logistics center.")


        print(f"Generating {num_sales_outlets} sales outlets within {radius_km} km radius...")
        for _ in range(num_sales_outlets):
            # Sales outlets could also have various distributions. Using uniform distribution here.
            point = _generate_point_in_circle(center_latitude, center_longitude, radius_km, distribution_type='uniform')
            if point:
                generated_points['sales_outlets'].append(point)
            else:
                 warnings.warn("Failed to generate a valid location for a sales outlet.")

        print(f"Generating {num_customers} customers within {radius_km} km radius...")
        # Customer distribution follows the specified pattern
        customer_distribution = 'solomon-like' if use_solomon_like_distribution else 'uniform'
        for _ in range(num_customers):
             point = _generate_point_in_circle(center_latitude, center_longitude, radius_km, distribution_type=customer_distribution)
             if point:
                generated_points['customers'].append(point)
             else:
                  warnings.warn("Failed to generate a valid location for a customer.")


    except Exception as e:
        print(f"An unexpected error occurred during location generation: {e}")
        traceback.print_exc()
        return None

    print("Location generation complete.")
    # Check if the number of generated points matches the requested number for non-zero requests
    if num_logistics_centers > 0 and len(generated_points['logistics_centers']) != num_logistics_centers:
         warnings.warn(f"Requested {num_logistics_centers} logistics centers but only generated {len(generated_points['logistics_centers'])}.")
    if num_sales_outlets > 0 and len(generated_points['sales_outlets']) != num_sales_outlets:
         warnings.warn(f"Requested {num_sales_outlets} sales outlets but only generated {len(generated_points['sales_outlets'])}.")
    if num_customers > 0 and len(generated_points['customers']) != num_customers:
         warnings.warn(f"Requested {num_customers} customers but only generated {len(generated_points['customers'])}.")


    return generated_points

def generate_demand(num_customers: int, min_demand: float, max_demand: float) -> list | None:
    """
    Generates random demands for customers.

    Demands are generated as random floating-point numbers (or integers if min/max are integers)
    within a specified range, sampled from a uniform distribution.

    Args:
        num_customers (int): The number of customers for whom to generate demands. Must be non-negative.
        min_demand (float): The minimum possible demand value. Must be non-negative.
        max_demand (float): The maximum possible demand value. Must be >= min_demand.

    Returns:
        list | None: A list of generated demand values for each customer.
                     Returns None if input validation fails.
    """
    # --- Input Validation ---
    if not isinstance(num_customers, int) or num_customers < 0:
        warnings.warn("Invalid input: Number of customers for demand generation must be a non-negative integer.")
        return None
    if not isinstance(min_demand, (int, float)) or not isinstance(max_demand, (int, float)):
        warnings.warn("Invalid input: Minimum and maximum demand must be numeric.")
        return None
    if min_demand < 0:
         warnings.warn(f"Input Warning: Minimum demand ({min_demand}) is negative. Clamping to 0.0.")
         min_demand = 0.0 # Clamp min_demand to 0
    if max_demand < min_demand:
        warnings.warn(f"Invalid input: Maximum demand ({max_demand}) is less than minimum demand ({min_demand}).")
        return None

    generated_demands = []
    try:
        print(f"Generating {num_customers} demands between {min_demand} and {max_demand}...")
        if num_customers > 0:
            # Generate demands using numpy's uniform distribution for efficiency
            generated_demands = np.random.uniform(low=min_demand, high=max_demand, size=num_customers).tolist()

            # Optional: Round demands to integers if min/max were integers
            # This check ensures if users provide integer bounds, the output is also integers.
            if isinstance(min_demand, int) and isinstance(max_demand, int):
                 generated_demands = [round(d) for d in generated_demands]

    except Exception as e:
        print(f"An unexpected error occurred during demand generation: {e}")
        traceback.print_exc()
        return None

    print("Demand generation complete.")
    return generated_demands


# --- Optional Main Execution Block for Standalone Testing ---
if __name__ == '__main__':
    """
    Standalone execution block for testing the data generation functions.
    Generates sample data and optionally saves it to a CSV file.
    """
    print("Running data_generator.py in standalone test mode.")

    # --- Example Usage Parameters ---
    test_num_centers = 3 # Example for multiple centers
    test_num_outlets = 10
    test_num_customers = 100
    test_center_lat = 34.0522 # Los Angeles approx
    test_center_lon = -118.2437
    test_radius_km = 100.0
    test_min_demand = 5.0
    test_max_demand = 50.0
    test_use_solomon_like = True # Test both distributions

    print("\n--- Generating Locations ---")
    test_locations = generate_locations(
        num_logistics_centers=test_num_centers,
        num_sales_outlets=test_num_outlets,
        num_customers=test_num_customers,
        center_latitude=test_center_lat,
        center_longitude=test_center_lon,
        radius_km=test_radius_km,
        use_solomon_like_distribution=test_use_solomon_like
    )

    if test_locations:
        print("\n--- Location Generation Summary ---")
        print(f"Generated Logistics Centers: {len(test_locations.get('logistics_centers', []))} points")
        # print(test_locations['logistics_centers']) # Uncomment to see coordinates
        print(f"Generated Sales Outlets: {len(test_locations.get('sales_outlets', []))} points")
        # print(test_locations['sales_outlets']) # Uncomment to see coordinates
        print(f"Generated Customers: {len(test_locations.get('customers', []))} points")
        # print(test_locations['customers']) # Uncomment to see coordinates
    else:
        print("\nLocation generation failed. Cannot proceed with demand generation or saving.")
        sys.exit(1)


    print("\n--- Generating Demands ---")
    test_demands = generate_demand(
        num_customers=len(test_locations.get('customers', [])), # Ensure demand count matches actual generated customers
        min_demand=test_min_demand,
        max_demand=test_max_demand
    )

    if test_demands:
        print(f"\nGenerated {len(test_demands)} demands.")
        # print(test_demands) # Uncomment to see demands
    else:
        print("\nDemand generation failed.")
        # Continue if locations generated, but demand failed (may not be runnable for optimization)


    # --- Optional: Save generated data to CSV ---
    # This part structures the data into a format similar to standard VRP datasets for potential loading.
    if test_locations and test_demands is not None and len(test_locations.get('customers', [])) == len(test_demands):
        print("\n--- Saving data to CSV ---")
        try:
             data_list = []
             # Add logistics centers (type 'depot' for compatibility with some parsers, ID starts from 0 or 1)
             # Using 'logistics_center' type explicitly and ID starts from 1
             for i, loc in enumerate(test_locations.get('logistics_centers',[])):
                 data_list.append(['logistics_center', i + 1, loc[0], loc[1], 0, 0, 0, 0, 0]) # Add dummy 0s for demand, tw, service

             # Add sales outlets (type 'outlet', ID continues)
             for i, loc in enumerate(test_locations.get('sales_outlets',[])):
                 data_list.append(['sales_outlet', len(test_locations.get('logistics_centers',[])) + i + 1, loc[0], loc[1], 0, 0, 0, 0, 0]) # Add dummy 0s

             # Add customers (type 'customer', ID continues)
             current_id_counter = len(test_locations.get('logistics_centers',[])) + len(test_locations.get('sales_outlets',[]))
             for i, loc in enumerate(test_locations.get('customers',[])):
                 demand = test_demands[i] if i < len(test_demands) else 0.0 # Safely get demand
                 data_list.append(['customer', current_id_counter + i + 1, loc[0], loc[1], demand, 0, 0, 0, 0]) # Add dummy 0s for tw, service

             # Create DataFrame
             df = pd.DataFrame(data_list, columns=['type', 'id', 'latitude', 'longitude', 'demand', 'ready_time', 'due_time', 'service_duration', 'assigned_depot_id'])
             # Added dummy columns for compatibility and future expansion (TW, Service, Assigned Depot)

             output_dir = 'output/generated_data' # Save to a dedicated subdirectory
             output_filename = os.path.join(output_dir, 'generated_md_2evrpsd_instance.csv')
             os.makedirs(output_dir, exist_ok=True) # Ensure output dir exists

             df.to_csv(output_filename, index=False)
             print(f"Successfully saved generated data to {output_filename}")

        except ImportError:
            print("\nWarning: pandas library not installed. Skipping CSV export.")
            print("Install pandas with: pip install pandas")
        except Exception as e:
            print(f"\nError saving generated data to CSV: {e}")
            traceback.print_exc()
    elif test_locations and test_demands is None:
         print("\nSkipping CSV export because demand generation failed.")
    elif test_locations and test_demands is not None and len(test_locations.get('customers', [])) != len(test_demands):
         print(f"\nWarning: Mismatch between number of generated customers ({len(test_locations.get('customers', []))}) and demands ({len(test_demands)}). Skipping CSV export.")
    else:
         print("\nSkipping CSV export due to location generation errors.")


    print("\nStandalone test finished.")