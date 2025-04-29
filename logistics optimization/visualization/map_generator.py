# visualization/map_generator.py
# -*- coding: utf-8 -*-
"""
Generates interactive Folium maps to visualize logistics routes.
"""
import folium
import os
import webbrowser
import warnings # Using warnings for non-critical issues

# Try importing core utilities, handle potential path issues
try:
    # Assumes core package is accessible
    from core.distance_calculator import haversine
except ImportError:
    # Fallback if run standalone or path issues
    try:
         import sys
         # Get the directory containing the current file (visualization)
         current_dir = os.path.dirname(os.path.abspath(__file__))
         # Get the parent directory (project root)
         project_root = os.path.dirname(current_dir)
         if project_root not in sys.path:
              sys.path.insert(0, project_root)
         from core.distance_calculator import haversine
         print("MapGen: Successfully imported haversine via relative path logic.")
    except ImportError:
        warnings.warn("Warning (map_generator): Could not import haversine from core. Using dummy distance.")
        # Provide a dummy function to avoid NameError, but results will be inaccurate
        def haversine(coord1, coord2):
            # This dummy function returns a simple Manhattan distance for testing purposes
            # It's NOT geographically accurate. Replace with real haversine if possible.
            if not coord1 or not coord2 or len(coord1) != 2 or len(coord2) != 2:
                 return float('inf')
            try:
                return abs(coord1[0] - coord2[0]) + abs(coord1[1] - coord2[1]) * 100 # Arbitrary scaling
            except:
                return float('inf')

# Define colors and icons consistently using FontAwesome
# Ensure these match colors used elsewhere if consistency is desired
COLOR_CENTER = 'green'
COLOR_SALES = 'blue'
COLOR_CUSTOMER = 'red'
COLOR_VEHICLE_S1 = 'black'    # Stage 1: Center to Sales
COLOR_VEHICLE_S2 = 'darkblue' # Stage 2: Sales to Customer (Vehicle)
COLOR_DRONE_S2 = 'purple'   # Stage 2: Sales to Customer (Drone)

ICON_CENTER = 'home'
ICON_SALES = 'store'
ICON_CUSTOMER = 'user'
ICON_PREFIX = 'fa' # Use FontAwesome prefix


def add_markers(m, points, demands):
    """
    Adds styled markers for centers, sales outlets, and customers to the Folium map.

    Args:
        m (folium.Map): The Folium map object.
        points (dict): Dictionary containing location data ('logistics_centers', 'sales_outlets', 'customers').
        demands (list): List of customer demands.
    """
    # Logistics Centers
    logistics_centers = points.get("logistics_centers", [])
    for i, loc in enumerate(logistics_centers):
        if loc and len(loc) == 2:
            folium.Marker(
                location=loc,
                popup=folium.Popup(f"Logistics Center {i+1}", max_width="100%"),
                tooltip=f"Logistics Center {i+1}",
                icon=folium.Icon(color=COLOR_CENTER, icon=ICON_CENTER, prefix=ICON_PREFIX)
            ).add_to(m)
        else:
            warnings.warn(f"Skipping invalid logistics center location at index {i}: {loc}")

    # Sales Outlets
    sales_outlets = points.get("sales_outlets", [])
    for i, loc in enumerate(sales_outlets):
        if loc and len(loc) == 2:
            folium.Marker(
                location=loc,
                popup=folium.Popup(f"Sales Outlet {i+1}", max_width="100%"),
                tooltip=f"Sales Outlet {i+1}",
                icon=folium.Icon(color=COLOR_SALES, icon=ICON_SALES, prefix=ICON_PREFIX)
            ).add_to(m)
        else:
            warnings.warn(f"Skipping invalid sales outlet location at index {i}: {loc}")

    # Customers
    customers = points.get("customers", [])
    num_demands = len(demands) if demands else 0
    for i, loc in enumerate(customers):
        if loc and len(loc) == 2:
            demand = demands[i] if demands and 0 <= i < num_demands else 'N/A'
            folium.Marker(
                location=loc,
                popup=folium.Popup(f"Customer {i+1}<br>Demand: {demand}", max_width="100%"),
                tooltip=f"Customer {i+1} (Demand: {demand})",
                icon=folium.Icon(color=COLOR_CUSTOMER, icon=ICON_CUSTOMER, prefix=ICON_PREFIX)
            ).add_to(m)
        else:
            warnings.warn(f"Skipping invalid customer location at index {i}: {loc}")

def add_routes(m, solution_structure, points_data, vehicle_params, drone_params):
    """
    Adds route polylines to the Folium map based on the solution structure.
    Handles multi-depot Stage 1 routes and distinguishes Stage 2 vehicle/drone routes.

    Args:
        m (folium.Map): The Folium map object.
        solution_structure (dict | object): A dictionary or object containing the solution details.
                                            Expected keys/attributes: 'stage1_routes', 'stage2_trips'.
                                            If an object (like SolutionCandidate), assumes these attributes exist.
        points_data (dict): Dictionary containing location data ('logistics_centers', 'sales_outlets', 'customers').
        vehicle_params (dict): Vehicle parameters.
        drone_params (dict): Drone parameters.
    """
    depot_coords = points_data.get('logistics_centers', [])
    outlet_coords = points_data.get('sales_outlets', [])
    customer_coords = points_data.get('customers', [])
    num_outlets = len(outlet_coords)
    num_customers = len(customer_coords)

    # Determine how to access solution data (dict vs object attribute)
    stage1_routes = {}
    stage2_trips = {}
    if isinstance(solution_structure, dict):
        stage1_routes = solution_structure.get('stage1_routes', {})
        stage2_trips = solution_structure.get('stage2_trips', {})
    elif hasattr(solution_structure, 'stage1_routes') and hasattr(solution_structure, 'stage2_trips'):
        stage1_routes = getattr(solution_structure, 'stage1_routes', {})
        stage2_trips = getattr(solution_structure, 'stage2_trips', {})
    else:
        warnings.warn("MapGen Warning: Invalid solution_structure type or missing route data. Cannot draw routes.")
        return

    # --- Stage 1 Routes ---
    if stage1_routes and isinstance(stage1_routes, dict):
        route_colors_s1 = ['black', 'darkred', 'darkgreen', 'gray', 'darkpurple'] # Cycle through colors for different depots
        for depot_idx, outlet_indices in stage1_routes.items():
            if not (0 <= depot_idx < len(depot_coords)):
                warnings.warn(f"MapGen Warning: Invalid depot index {depot_idx} in Stage 1 routes.")
                continue
            if not isinstance(outlet_indices, list):
                warnings.warn(f"MapGen Warning: Invalid route sequence type for depot {depot_idx}.")
                continue

            if not outlet_indices: continue # Skip empty routes

            route_coords = [depot_coords[depot_idx]]
            valid_route = True
            for outlet_idx in outlet_indices:
                if 0 <= outlet_idx < num_outlets:
                    route_coords.append(outlet_coords[outlet_idx])
                else:
                    warnings.warn(f"MapGen Warning: Invalid outlet index {outlet_idx} in Stage 1 route for depot {depot_idx}.")
                    valid_route = False
                    break # Stop processing this invalid route

            if valid_route and len(route_coords) > 1:
                 route_coords.append(depot_coords[depot_idx]) # Return to depot
                 route_color = route_colors_s1[depot_idx % len(route_colors_s1)] # Cycle through colors
                 folium.PolyLine(
                     locations=route_coords,
                     color=route_color, # Use cycled color
                     weight=2.5,
                     opacity=0.8,
                     tooltip=f"Depot {depot_idx+1} Stage 1 Route"
                 ).add_to(m)


    # --- Stage 2 Trips ---
    if stage2_trips and isinstance(stage2_trips, dict):
        for outlet_idx, trips_from_outlet in stage2_trips.items():
             if not (0 <= outlet_idx < num_outlets):
                 warnings.warn(f"MapGen Warning: Invalid outlet index {outlet_idx} in Stage 2 trips.")
                 continue
             if not isinstance(trips_from_outlet, list):
                  warnings.warn(f"MapGen Warning: Invalid trip data type for outlet {outlet_idx}.")
                  continue

             outlet_loc = outlet_coords[outlet_idx]

             for trip_num, trip_info in enumerate(trips_from_outlet):
                 if not isinstance(trip_info, dict):
                     warnings.warn(f"MapGen Warning: Invalid trip info format for outlet {outlet_idx}, trip {trip_num+1}.")
                     continue

                 trip_type = trip_info.get('type', 'unknown').lower()
                 cust_indices_in_trip = trip_info.get('route', []) # Expects a list of customer indices

                 if not cust_indices_in_trip or not isinstance(cust_indices_in_trip, list): continue

                 # Filter valid customer indices and get coords
                 valid_cust_coords = []
                 valid_cust_indices = []
                 for c_idx in cust_indices_in_trip:
                      if 0 <= c_idx < num_customers and customer_coords[c_idx]:
                           valid_cust_coords.append(customer_coords[c_idx])
                           valid_cust_indices.append(c_idx)
                      else:
                           warnings.warn(f"MapGen Warning: Invalid customer index {c_idx} or missing location for trip {trip_num+1} from outlet {outlet_idx}.")

                 if not valid_cust_coords: continue # Skip if no valid customers in trip

                 # Draw based on type
                 if trip_type == 'vehicle':
                     trip_path_coords = [outlet_loc] + valid_cust_coords + [outlet_loc] # O -> C1 -> C2 -> ... -> O
                     folium.PolyLine(
                         locations=trip_path_coords,
                         color=COLOR_VEHICLE_S2,
                         weight=2,
                         opacity=0.8,
                         tooltip=f"O{outlet_idx+1} Vehicle Trip {trip_num+1}: -> {' -> '.join([f'C{c+1}' for c in valid_cust_indices])} ->"
                     ).add_to(m)
                 elif trip_type == 'drone':
                     # Draw individual dashed lines from outlet to each customer in the drone trip
                     for i, c_idx in enumerate(valid_cust_indices):
                          cust_loc = valid_cust_coords[i]
                          folium.PolyLine(
                              locations=[outlet_loc, cust_loc], # Direct flight O -> C
                              color=COLOR_DRONE_S2,
                              weight=1.5,
                              opacity=0.9,
                              dash_array='5, 5',
                              tooltip=f"O{outlet_idx+1} Drone Trip {trip_num+1}: -> C{c_idx+1}"
                          ).add_to(m)
                 else:
                      warnings.warn(f"MapGen Warning: Unknown trip type '{trip_type}' for outlet {outlet_idx}, trip {trip_num+1}. Skipping drawing.")


def add_legend(m):
    """Adds a custom HTML legend to the Folium map."""
    legend_html = f"""
     <div style="position: fixed;
                 bottom: 10px; left: 10px; width: 180px; height: auto;
                 background-color: white; border:2px solid grey; z-index:9999;
                 font-size:12px; padding: 10px; border-radius: 5px; opacity: 0.85;">
         <p style="margin-top:0; margin-bottom:5px; font-weight: bold;">Legend:</p>
         <div style="margin-bottom: 3px;"><i class="fa fa-{ICON_CENTER}" style="color:{COLOR_CENTER}"></i> Logistics Center</div>
         <div style="margin-bottom: 3px;"><i class="fa fa-{ICON_SALES}" style="color:{COLOR_SALES}"></i> Sales Outlet</div>
         <div style="margin-bottom: 5px;"><i class="fa fa-{ICON_CUSTOMER}" style="color:{COLOR_CUSTOMER}"></i> Customer</div>
         <div style="margin-bottom: 3px;"><hr style="border: 1px solid {COLOR_VEHICLE_S1}; margin: 2px 0;"> Stage 1 Route (Vehicle)</div>
         <div style="margin-bottom: 3px;"><hr style="border: 1px solid {COLOR_VEHICLE_S2}; margin: 2px 0;"> Stage 2 Route (Vehicle)</div>
         <div><hr style="border: 1px dashed {COLOR_DRONE_S2}; margin: 2px 0;"> Stage 2 Route (Drone)</div>
     </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))


def generate_folium_map(problem_data, solution_structure, vehicle_params, drone_params,
                         output_path="output/maps/route_map.html", map_title="Logistics Route Map"):
    """
    Generates and saves a Folium map visualizing the solution's routes, including a legend.
    Handles multi-depot structure.

    Args:
        problem_data (dict): Dictionary containing 'locations' and 'demands'.
        solution_structure (dict | object | None): Dictionary or object containing the solution details
                                                   (e.g., 'stage1_routes', 'stage2_trips').
                                                   If None, only points will be plotted.
        vehicle_params (dict): Vehicle parameters.
        drone_params (dict): Drone parameters.
        output_path (str): Path to save the generated HTML map file.
        map_title (str): Title for the map (currently not used in the HTML directly, but good practice).

    Returns:
        str | None: The path to the saved map file if successful, None otherwise.
    """
    points = problem_data.get('locations')
    demands = problem_data.get('demands')

    if not points or not points.get("logistics_centers"):
        warnings.warn("MapGen Error: No logistics centers defined in points data.")
        return None

    # Calculate map center (e.g., average of all points or first depot)
    all_coords = points.get("logistics_centers", []) + points.get("sales_outlets", []) + points.get("customers", [])
    valid_coords = [c for c in all_coords if c and len(c)==2]
    if valid_coords:
        avg_lat = sum(c[0] for c in valid_coords) / len(valid_coords)
        avg_lon = sum(c[1] for c in valid_coords) / len(valid_coords)
        center_loc = (avg_lat, avg_lon)
    else:
        center_loc = (0, 0) # Fallback center
        warnings.warn("MapGen Warning: No valid coordinates found to center map. Using (0,0).")


    # Create the map centered appropriately
    m = folium.Map(location=center_loc, zoom_start=12, control_scale=True)

    # Add Tile Layers
    folium.TileLayer('openstreetmap', name='OpenStreetMap').add_to(m)
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    # folium.TileLayer('Stamen Terrain', attr='Map tiles by Stamen Design', name='Stamen Terrain').add_to(m) # Optional

    # Add markers for all points
    add_markers(m, points, demands)

    # Add routes from the solution if provided and valid
    if solution_structure:
        add_routes(m, solution_structure, points, vehicle_params, drone_params)
    else:
         warnings.warn("MapGen Info: No solution structure provided. Plotting points only.")

    # Add the custom HTML legend
    add_legend(m)

    # Add Layer control to toggle base maps or future overlay layers
    folium.LayerControl().add_to(m)

    # Ensure output directory exists
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
         try:
             os.makedirs(output_dir, exist_ok=True)
         except OSError as e:
             warnings.warn(f"MapGen Error: Could not create output directory '{output_dir}': {e}")
             return None

    # Save the map
    try:
        m.save(output_path)
        print(f"Interactive map saved to: {output_path}")
        return output_path # <-- Return the path string on success
    except Exception as e:
        warnings.warn(f"MapGen Error: Failed to save Folium map to '{output_path}': {e}")
        return None # <-- Return None on failure

def open_map_in_browser(filename):
    """Opens the generated HTML map file in the default web browser."""
    if not filename or not isinstance(filename, str):
         warnings.warn(f"Error opening map: Invalid filename type ({type(filename)}). Expected string.")
         return False
    if not os.path.exists(filename):
        warnings.warn(f"Error: Cannot open map - file not found at {filename}")
        return False
    try:
        # Get absolute path and format as file URL
        abs_path = os.path.realpath(filename)
        # Ensure the path is formatted correctly for a URL
        # Replace backslashes for Windows paths if necessary, though webbrowser usually handles this
        url = f"file:///{abs_path.replace(os.sep, '/')}"
        print(f"Attempting to open map URL: {url}")
        success = webbrowser.open(url, new=2) # new=2: open in new tab if possible
        if not success:
            warnings.warn(f"webbrowser.open returned False for {url}. Browser might not have opened.")
            # Try without the triple slash for file://, sometimes needed on certain systems
            url_alt = f"file://{abs_path.replace(os.sep, '/')}"
            print(f"Retrying with alternative URL format: {url_alt}")
            success = webbrowser.open(url_alt, new=2)
        return success
    except Exception as e:
        warnings.warn(f"Error opening map file '{filename}' in browser: {e}")
        return False

# Example usage block (useful for testing this module independently)
if __name__ == "__main__":
     print("Running map_generator example...")
     # Example dummy data (Multi-Depot)
     dummy_problem_data = {
         "locations": {
             "logistics_centers": [(39.95, 116.35), (39.85, 116.50)], # Depot 0, Depot 1
             "sales_outlets": [(39.92, 116.38), (39.90, 116.45), (39.88, 116.52), (39.96, 116.30)], # Outlets 0, 1, 2, 3
             "customers": [(39.93, 116.39), (39.91, 116.46), (39.89, 116.53), (39.97, 116.31), (39.87, 116.51)] # Customers 0, 1, 2, 3, 4
         },
         "demands": [10, 15, 5, 20, 8]
     }
     # Example solution structure (can be dict or object)
     dummy_solution_structure = {
         'stage1_routes': {
             0: [0, 3], # Depot 0 -> Outlet 0 -> Outlet 3 -> Depot 0
             1: [1, 2]  # Depot 1 -> Outlet 1 -> Outlet 2 -> Depot 1
         },
         'stage2_trips': {
             0: [{'type': 'vehicle', 'route': [0]}],          # Outlet 0 -> Cust 0 (Vehicle)
             1: [{'type': 'drone', 'route': [1]}],            # Outlet 1 -> Cust 1 (Drone)
             2: [{'type': 'vehicle', 'route': [2, 4]}],      # Outlet 2 -> Cust 2 -> Cust 4 (Vehicle)
             3: [{'type': 'drone', 'route': [3]}]             # Outlet 3 -> Cust 3 (Drone)
         }
         # Add other attributes if your SolutionCandidate has them (not strictly needed by map generator)
     }
     dummy_vehicle_params = {'payload': 100, 'cost_per_km': 1, 'speed_kmph': 50}
     dummy_drone_params = {'payload': 10, 'max_flight_distance_km': 5, 'cost_per_km': 0.5, 'speed_kmph': 80}

     output_filename = os.path.join("..", "output", "maps", "example_map_generator_md_test.html") # Relative path for testing
     print(f"Attempting to save example map to: {output_filename}")

     # Call the generator
     generated_path = generate_folium_map(
         problem_data=dummy_problem_data,
         solution_structure=dummy_solution_structure,
         vehicle_params=dummy_vehicle_params,
         drone_params=dummy_drone_params,
         output_path=output_filename
     )

     if generated_path:
          print(f"Example map generated successfully: {generated_path}")
          # Automatically open only if generation was successful
          open_map_in_browser(generated_path)
     else:
          print("Example map generation failed.")

     # Test plotting only points
     output_points_filename = os.path.join("..", "output", "maps", "example_map_generator_points_only.html")
     print(f"\nAttempting to save points-only map to: {output_points_filename}")
     generated_points_path = generate_folium_map(
         problem_data=dummy_problem_data,
         solution_structure=None, # Pass None for solution
         vehicle_params={}, # Pass empty dicts
         drone_params={},
         output_path=output_points_filename
     )
     if generated_points_path:
          print(f"Points-only map generated successfully: {generated_points_path}")
          open_map_in_browser(generated_points_path)
     else:
          print("Points-only map generation failed.")