# algorithm/pso_optimizer.py
# -*- coding: utf-8 -*-\
"""
Particle Swarm Optimization (PSO) algorithm adapted for Multi-Depot,
Two-Echelon VRP with Drones and Split Deliveries (MD-2E-VRPSD).

Applies PSO optimization to Stage 1 routes per depot using a permutation-based
representation and adapted velocity/position update mechanisms (based on swap sequences).
Stage 2 trips (allowing splits and using vehicles/drones) are generated
heuristically during fitness evaluation by the core cost function.

Incorporates feasibility priority in personal best (pbest) and global best (gbest)
updates to guide the swarm towards feasible solutions (no unmet demand).

Relies on updated core utility functions and the cost function for initialization
and evaluation.
"""

import random
import copy
import time
import numpy as np
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
    project_root_pso = os.path.dirname(current_dir)
    if project_root_pso not in sys.path:
        sys.path.insert(0, project_root_pso)
        # print(f"PSO: Added project root to sys.path: {project_root_pso}") # Optional debug print

    # Import necessary components from core modules
    # Need SolutionCandidate, initial solution generator
    from core.problem_utils import (
        SolutionCandidate,
        create_initial_solution_mdsd, # Potentially used if initial_solution_candidate is not passed fully formed
        create_heuristic_trips_split_delivery # Needed by SolutionCandidate.evaluate indirectly
    )
    # Need distance and cost function references, but they are passed to SolutionCandidate.evaluate
    from core.distance_calculator import haversine # Needed by SolutionCandidate.evaluate indirectly
    from core.cost_function import calculate_total_cost_and_evaluate # Needed by SolutionCandidate.evaluate

except ImportError as e:
    print(f"CRITICAL ERROR in algorithm.pso_optimizer: Failed during initial import block: {e}")
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
        all_locations = {} # Dummy

    def create_initial_solution_mdsd(*args, **kwargs):
        print("DUMMY create_initial_solution_mdsd called in PSO due to import error!")
        return None
    def haversine(*args): return float('inf')
    def calculate_total_cost_and_evaluate(*args, **kwargs): return float('inf'), float('inf'), float('inf'), float('inf'), {}, True, True, {}
    def create_heuristic_trips_split_delivery(*args, **kwargs): return [] # Dummy

    print("Particle Swarm Optimization will use dummy functions/classes due to critical import failure.")

except Exception as e:
    print(f"An unexpected error occurred during PSO import block: {e}")
    traceback.print_exc()
    # Define a dummy run function that indicates error
    def run_pso_optimizer(*args, **kwargs):
        print("CRITICAL ERROR: Particle Swarm Optimization initialization failed.")
        return {'run_error': f"Initialization failed: {e}"}
    print("Particle Swarm Optimization will not run due to unexpected import error.")


# Define a small tolerance for floating-point comparisons
FLOAT_TOLERANCE_PSO = 1e-6


# --- Helper Functions for Permutation PSO Operations ---

def get_swap_sequence(perm1: list, perm2: list) -> list[tuple[int, int]]:
    """
    Calculates a sequence of swaps to transform permutation perm1 into perm2.
    This is a common way to represent velocity in permutation-based PSO.

    Args:
        perm1 (list): The starting permutation (list of elements).
        perm2 (list): The target permutation (list of elements).

    Returns:
        list[tuple[int, int]]: A list of swap operations (index_a, index_b)
                                to apply to perm1 to get perm2. Returns empty list
                                if inputs are invalid or identical.
    """
    if not isinstance(perm1, list) or not isinstance(perm2, list) or len(perm1) != len(perm2):
        warnings.warn("Invalid input for get_swap_sequence: inputs must be lists of same length.")
        return []

    size = len(perm1)
    if size == 0:
        return []

    # Create a working copy of perm1 that we will transform
    working_perm = list(perm1)
    swaps = []

    # Use a mapping from element value to its current index in working_perm for efficient lookup
    # This assumes element values are unique and hashable
    try:
        element_to_index = {element: i for i, element in enumerate(working_perm)}

        # Iterate through the target permutation (perm2)
        for i in range(size):
            target_element = perm2[i]
            current_element_at_i = working_perm[i]

            # If the element at position i in working_perm is not the desired element (from perm2)
            if current_element_at_i != target_element:
                # Find where the target element currently is in working_perm
                current_pos_of_target_element = element_to_index[target_element]

                # Perform the swap in working_perm
                working_perm[i], working_perm[current_pos_of_target_element] = working_perm[current_pos_of_target_element], working_perm[i]
                swaps.append((i, current_pos_of_target_element)) # Record the swap

                # Update the index mapping for the two elements that were swapped
                element_to_index[current_element_at_i] = current_pos_of_target_element # The element that was at i is now at current_pos_of_target_element
                element_to_index[target_element] = i # The target element is now at i

            # If working_perm[i] is already the target_element, do nothing and move to the next position.

    except (ValueError, KeyError) as e:
        warnings.warn(f"Error in get_swap_sequence during permutation processing (likely non-unique or missing elements): {e}")
        # Return empty swaps as sequence is invalid
        return []
    except Exception as e:
        warnings.warn(f"An unexpected error occurred in get_swap_sequence: {e}")
        traceback.print_exc()
        return []


    # Verification (Optional): Ensure applying swaps to perm1 results in perm2
    # This can be computationally expensive for long permutations
    # verification_perm = list(perm1)
    # for idx1, idx2 in swaps:
    #     verification_perm[idx1], verification_perm[idx2] = verification_perm[idx2], verification_perm[idx1]
    # if verification_perm != perm2:
    #     warnings.warn("Verification failed: Applying generated swaps did not result in the target permutation.")
    #     # This could indicate an issue with the swap generation logic
    #     return [] # Return empty swaps if verification fails


    return swaps

def apply_swap_sequence(perm: list, swaps: list[tuple[int, int]]) -> list:
    """
    Applies a sequence of swap operations to a permutation.

    Args:
        perm (list): The starting permutation.
        swaps (list[tuple[int, int]]): A list of swap operations (index_a, index_b).

    Returns:
        list: The resulting permutation after applying the swaps.
              Returns a copy of the original list if inputs are invalid.
    """
    if not isinstance(perm, list) or not isinstance(swaps, list):
        warnings.warn("Invalid input for apply_swap_sequence.")
        return list(perm) # Return copy if inputs are invalid

    mutated_perm = list(perm) # Work on a copy
    size = len(mutated_perm)

    try:
        for idx1, idx2 in swaps:
            # Validate indices
            if 0 <= idx1 < size and 0 <= idx2 < size:
                mutated_perm[idx1], mutated_perm[idx2] = mutated_perm[idx2], mutated_perm[idx1]
            else:
                warnings.warn(f"Invalid swap indices ({idx1}, {idx2}) for permutation size {size}. Skipping swap.")
                # Continue applying other valid swaps

    except Exception as e:
        warnings.warn(f"An unexpected error occurred in apply_swap_sequence: {e}")
        traceback.print_exc()
        # Return the partially mutated or original copy in case of error
        return list(perm)


    return mutated_perm

def compose_swap_sequences(seq1: list[tuple[int, int]], seq2: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """
    Composes two swap sequences. Applying the resulting sequence is equivalent
    to applying seq1 then seq2. This is permutation addition (seq1 + seq2).

    Note: Composition of swap sequences is not straightforward concatenation.
    Applying a swap changes indices for subsequent swaps. A common approach
    is to represent swaps as transpositions and compose them, or simulate
    the application.

    A simpler approach for permutation PSO velocity update (v = v + c1*r1*(pbest-x) + c2*r2*(gbest-x))
    is to calculate (pbest-x) and (gbest-x) as swap sequences, scale them
    (e.g., apply a fraction of swaps randomly), and then apply them sequentially
    to the current position. The inertia term (w*v) means applying a fraction of the current velocity.

    This function `compose_swap_sequences` might not be directly needed if we
    implement the velocity update by applying scaled swap sequences sequentially.
    Let's use the application-based approach for velocity update.

    However, if a composition function is needed, a correct implementation is complex.
    For this problem context, it's more practical to represent velocity as a set of swaps
    and define operations (scaling, addition) in terms of applying these swaps.

    Let's define the operations needed for PSO velocity update directly.
    """
    # For this implementation, we will represent velocity as a list of swaps.
    # Velocity update: v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))
    # - w*v(t): Apply a fraction 'w' of swaps from v(t) randomly.
    # - (pbest - x(t)): Get swaps to go from x(t) to pbest.
    # - c1*r1*(pbest - x(t)): Apply a fraction 'c1*r1' of swaps from (pbest - x(t)) randomly.
    # - (gbest - x(t)): Get swaps to go from x(t) to gbest.
    # - c2*r2*(gbest - x(t)): Apply a fraction 'c2*r2' of swaps from (gbest - x(t)) randomly.
    # - v(t+1): A new sequence of swaps resulting from applying the scaled components sequentially.
    # This is not a simple concatenation or composition. The `Particle.update_velocity` and `Particle.update_position`
    # methods below will implement this application-based update.

    # Returning a dummy list for now if this function were ever called (it shouldn't be directly in this implementation)
    warnings.warn("compose_swap_sequences is not intended for direct use in this PSO implementation.")
    return []

# --- Particle Class ---
class Particle:
    """
    Represents a particle in the PSO swarm.

    Each particle has:
    - Position: A SolutionCandidate object representing the current solution.
    - Velocity: A dictionary where keys are depot indices and values are lists
                of swap operations representing the velocity for that depot's Stage 1 route.
    - Personal Best (pbest): The best SolutionCandidate found by this particle so far.
    - Personal Best Cost: The weighted cost of the pbest solution.
    """
    def __init__(self, initial_solution: SolutionCandidate, num_depots: int):
        """
        Initializes a Particle.

        Args:
            initial_solution (SolutionCandidate): The starting solution candidate
                                                  for this particle's initial position.
                                                  Should be a copy of the base initial solution.
            num_depots (int): The number of depots in the problem. Used to initialize
                              the velocity structure.
        """
        if not isinstance(initial_solution, SolutionCandidate):
             warnings.warn("Particle initialized without a valid SolutionCandidate.")
             # Initialize with an invalid dummy solution
             self.position = SolutionCandidate(problem_data={}, vehicle_params={}, drone_params={}, unmet_demand_penalty=float('inf'))
             self.pbest = copy.deepcopy(self.position)
             self.pbest_cost = float('inf')
             self.velocity = {} # Empty velocity for invalid state
             return

        # Position is the current solution candidate
        self.position = initial_solution # This should be a copy

        # Personal Best (pbest) - Initially set to the starting position
        self.pbest = copy.deepcopy(self.position)
        self.pbest_cost = self.position.weighted_cost # Use weighted cost for pbest comparison

        # Velocity - Represented as a dictionary of swap sequences, one for each depot
        # Initial velocity is typically zero (empty swap sequences)
        self.velocity = {depot_idx: [] for depot_idx in range(num_depots)} # Key: depot index, Value: list of swaps

    def update_pbest(self):
        """
        Updates the particle's personal best (pbest) solution if the current
        position is better than the current pbest (using feasibility-first comparison).
        """
        # Use SolutionCandidate.__lt__ for comparison
        if self.position < self.pbest:
            self.pbest = copy.deepcopy(self.position)
            self.pbest_cost = self.position.weighted_cost
            # print(f" Particle updated pbest. New Cost: {format_float(self.pbest_cost, 4)}, Feasible: {self.pbest.is_feasible}")


    def update_velocity(self, gbest_solution: SolutionCandidate, w: float, c1: float, c2: float):
        """
        Updates the particle's velocity based on the standard PSO velocity equation,
        adapted for permutation representation using swap sequences.

        v(t+1) = w*v(t) + c1*r1*(pbest - x(t)) + c2*r2*(gbest - x(t))

        This is applied *independently* to the Stage 1 route of each depot.

        Args:
            gbest_solution (SolutionCandidate): The global best solution found by the swarm.
            w (float): Inertia weight.
            c1 (float): Cognitive weight (influence of pbest).
            c2 (float): Social weight (influence of gbest).
        """
        r1 = random.random() # Random numbers for cognitive component
        r2 = random.random() # Random numbers for social component

        new_velocity = {}

        depot_indices = list(self.position.stage1_routes.keys())
        if not depot_indices:
             # Cannot update velocity if no depots/routes
             # Keep velocity as empty or previous state
             self.velocity = {depot_idx: [] for depot_idx in self.velocity.keys()} # Clear velocity
             # warnings.warn("Cannot update particle velocity: No depots/routes in position.")
             return


        for depot_idx in depot_indices:
            current_route = self.position.stage1_routes.get(depot_idx, [])
            pbest_route = self.pbest.stage1_routes.get(depot_idx, [])
            gbest_route = gbest_solution.stage1_routes.get(depot_idx, [])
            current_depot_velocity = self.velocity.get(depot_idx, [])

            # Only update velocity if the route is not empty and has >= 2 elements for swaps
            if len(current_route) >= 2 and len(pbest_route) >= 2 and len(gbest_route) >= 2 and current_route == pbest_route == gbest_route:
                 # If all routes are the same, velocity tends towards zero
                 new_velocity[depot_idx] = []
                 continue
            elif len(current_route) >= 2 and len(pbest_route) >= 2 and len(gbest_route) >= 2 and len(current_depot_velocity) == 0 and current_route == pbest_route and current_route == gbest_route:
                 # Edge case: If velocity is empty and routes are identical, keep velocity empty
                 new_velocity[depot_idx] = []
                 continue # Skip calculation for this depot


            # --- Calculate Velocity Components (as swap sequences) ---

            # Inertia component: w * v(t)
            # Apply a fraction 'w' of the current velocity's swaps
            # A simple way: randomly select w * len(current_depot_velocity) swaps
            num_inertia_swaps = int(round(w * len(current_depot_velocity)))
            inertia_swaps = random.sample(current_depot_velocity, min(num_inertia_swaps, len(current_depot_velocity)))


            # Cognitive component: c1 * r1 * (pbest - x(t))
            # Get swaps from current position to pbest
            pbest_diff_swaps = get_swap_sequence(current_route, pbest_route)
            # Apply a fraction 'c1 * r1' of these swaps
            num_cognitive_swaps = int(round(c1 * r1 * len(pbest_diff_swaps)))
            cognitive_swaps = random.sample(pbest_diff_swaps, min(num_cognitive_swaps, len(pbest_diff_swaps)))


            # Social component: c2 * r2 * (gbest - x(t))
            # Get swaps from current position to gbest
            gbest_diff_swaps = get_swap_sequence(current_route, gbest_route)
            # Apply a fraction 'c2 * r2' of these swaps
            num_social_swaps = int(round(c2 * r2 * len(gbest_diff_swaps)))
            social_swaps = random.sample(gbest_diff_swaps, min(num_social_swaps, len(gbest_diff_swaps)))


            # --- Compose New Velocity ---
            # The new velocity is the *sequence of applications* of the scaled components.
            # Apply inertia swaps first, then cognitive swaps, then social swaps.
            # The resulting velocity is the total transformation represented as a sequence of swaps.
            # This is the complex part. A practical approach: The new velocity is a list
            # combining the scaled swaps from the three components. The order in which
            # these combined swaps are applied will determine the new position.
            # Let's simply concatenate the scaled swap lists to form the new velocity list.
            # The order within each component's swaps is important, but the order *between*
            # components when concatenated might not be perfectly theoretically sound
            # but is a common heuristic in permutation PSO implementations.

            new_velocity[depot_idx] = inertia_swaps + cognitive_swaps + social_swaps

            # Optional: Limit velocity magnitude (number of swaps) if needed
            # max_velocity = ... # Define a max number of swaps per depot route velocity
            # if len(new_velocity[depot_idx]) > max_velocity:
            #     new_velocity[depot_idx] = random.sample(new_velocity[depot_idx], max_velocity)


        self.velocity = new_velocity # Update the particle's velocity dictionary


    def update_position(self):
        """
        Updates the particle's position by applying its current velocity
        (sequence of swaps) to its current Stage 1 routes for each depot.

        x(t+1) = x(t) + v(t+1)
        """
        # print(" Applying velocity to update position...")
        # Create a new SolutionCandidate for the new position
        # Deep copy the current position as the base
        new_position = copy.deepcopy(self.position)

        depot_indices = list(new_position.stage1_routes.keys())

        for depot_idx in depot_indices:
            current_route = new_position.stage1_routes.get(depot_idx, [])
            depot_velocity = self.velocity.get(depot_idx, [])

            # Apply the velocity's swap sequence to this depot's route
            if current_route and depot_velocity: # Only apply if route and velocity are non-empty
                 try:
                     # Use the apply_swap_sequence helper function
                     new_route = apply_swap_sequence(current_route, depot_velocity)
                     new_position.stage1_routes[depot_idx] = new_route
                     # print(f"  Updated position for depot {depot_idx}.")
                 except Exception as e:
                      warnings.warn(f"Error applying velocity to update position for depot {depot_idx}: {e}")
                      traceback.print_exc()
                      # If applying velocity fails for a depot, maybe keep the old route for that depot?
                      # Or mark the new position as invalid/inf cost during evaluation.
                      # Keeping the old route for that depot is simpler:
                      new_position.stage1_routes[depot_idx] = current_route # Revert to old route for this depot
                      # This might still lead to an invalid overall solution, which evaluation will catch.


        # Mark the new position (SolutionCandidate) as unevaluated
        new_position.is_feasible = False
        new_position.weighted_cost = float('inf')
        new_position.evaluated_cost = float('inf')
        new_position.evaluated_time = float('inf')
        new_position.evaluated_unmet = float('inf')
        new_position.served_customer_details = {}
        new_position.evaluation_stage1_error = False
        new_position.evaluation_stage2_error = False
        new_position.stage2_trips = {}


        # Update the particle's position to the newly generated one
        self.position = new_position


def run_pso_optimizer(problem_data: dict, vehicle_params: dict, drone_params: dict,
                      unmet_demand_penalty: float, cost_weight: float, time_weight: float,
                      initial_solution_candidate: SolutionCandidate,
                      algo_specific_params: dict) -> dict | None:
    """
    Runs the Particle Swarm Optimization (PSO) algorithm to find a good solution for the
    Multi-Depot, Two-Echelon VRP with Drones and Split Deliveries.

    Applies a permutation-based PSO variant, focusing on optimizing Stage 1 routes
    for each depot.

    Args:
        problem_data (dict): Dictionary containing problem instance data.
        vehicle_params (dict): Dictionary of vehicle parameters.
        drone_params (dict): Dictionary of drone parameters.
        unmet_demand_penalty (float): Penalty cost per unit of unmet demand.
        cost_weight (float): Weight for raw cost in objective.
        time_weight (float): Weight for time/makespan in objective.
        initial_solution_candidate (SolutionCandidate): A base solution candidate
                                                        containing initial assignments
                                                        and a starting Stage 1 route structure
                                                        (e.g., from a greedy heuristic).
                                                        The swarm will be initialized based
                                                        on copies of this candidate.
        algo_specific_params (dict): Dictionary of PSO-specific parameters:
                                     'num_particles' (int), 'max_iterations' (int),
                                     'inertia_weight' (float), 'cognitive_weight' (float),
                                     'social_weight' (float).

    Returns:
        dict | None: A dictionary containing the best solution found (as a SolutionCandidate
                     object or a dictionary representing its state), its evaluation results,
                     and the global best cost history per iteration. Returns None or error dict
                     if the algorithm fails to run or find a valid solution.
        Example: {
            'best_solution': SolutionCandidate_object,
            'weighted_cost': float, 'evaluated_cost': float, 'evaluated_time': float,
            'evaluated_unmet': float, 'is_feasible': bool,
            'evaluation_stage1_error': bool, 'evaluation_stage2_error': bool,
            'stage1_routes': dict, 'stage2_trips': dict, 'served_customer_details': dict,
            'cost_history': list, # List of global best weighted cost per iteration
            'total_computation_time': float,
            'algorithm_name': 'pso_optimizer',
            'algorithm_params': dict # Parameters used
        }
    """
    print("\n--- Starting Particle Swarm Optimization (MD-SD) ---")
    start_time_pso = time.time()

    # --- Default PSO Parameters ---
    default_pso_params = {
        'num_particles': 50,
        'max_iterations': 200,
        'inertia_weight': 0.8,     # w
        'cognitive_weight': 1.5,   # c1
        'social_weight': 1.5       # c2
    }

    # --- Parameter Validation and Extraction ---
    try:
        if not isinstance(initial_solution_candidate, SolutionCandidate):
            raise TypeError("'initial_solution_candidate' must be a SolutionCandidate object.")
        if not initial_solution_candidate.all_locations or not initial_solution_candidate.demands:
            raise ValueError("'initial_solution_candidate' does not contain valid problem data.")
        if not initial_solution_candidate.stage1_routes:
            warnings.warn("Initial solution candidate has no Stage 1 routes. PSO may not be effective.")
        else:
             num_depots = len(initial_solution_candidate.all_locations.get('logistics_centers', []))
             if num_depots == 0:
                  warnings.warn("Initial solution candidate has no depots. PSO may not be meaningful.")


        # Merge default and provided parameters
        pso_params = default_pso_params.copy()
        if isinstance(algo_specific_params, dict):
             pso_params.update(algo_specific_params)
        else:
             warnings.warn("'algo_specific_params' is not a dictionary. Using default PSO parameters.")


        num_particles = pso_params.get('num_particles')
        max_iterations = pso_params.get('max_iterations')
        inertia_weight = pso_params.get('inertia_weight')
        cognitive_weight = pso_params.get('cognitive_weight')
        social_weight = pso_params.get('social_weight')

        # Validate PSO parameters
        if not isinstance(num_particles, int) or num_particles <= 0: raise ValueError("num_particles must be a positive integer.")
        if not isinstance(max_iterations, int) or max_iterations < 0: raise ValueError("max_iterations must be a non-negative integer.")
        if not isinstance(inertia_weight, (int, float)) or not (0.0 <= inertia_weight <= 1.0): raise ValueError("inertia_weight must be between 0.0 and 1.0.")
        if not isinstance(cognitive_weight, (int, float)) or cognitive_weight < 0: raise ValueError("cognitive_weight must be a non-negative number.")
        if not isinstance(social_weight, (int, float)) or social_weight < 0: raise ValueError("social_weight must be a non-negative number.")


        print(f"PSO Parameters: Particles={num_particles}, Iterations={max_iterations}, w={inertia_weight}, c1={cognitive_weight}, c2={social_weight}")

    except Exception as e:
        print(f"Error validating PSO parameters or initial solution: {e}")
        traceback.print_exc()
        return {'run_error': f"Parameter or initial solution validation failed: {e}"}


    # --- Initialization ---
    swarm = []
    gbest_solution = None # Global best solution candidate
    gbest_cost = float('inf') # Weighted cost of the global best

    num_depots_in_problem = len(initial_solution_candidate.all_locations.get('logistics_centers', []))


    print("Initializing PSO swarm...")
    try:
        for i in range(num_particles):
            # Create an initial position for the particle
            # A common way: deep copy the base initial solution and potentially add slight random perturbation
            particle_initial_position = copy.deepcopy(initial_solution_candidate)

            # Optional: Add slight random perturbation to the initial position (e.g., a few swaps)
            # This adds diversity to the initial swarm
            if particle_initial_position.stage1_routes:
                 num_perturbation_swaps = random.randint(0, 5) # Example: 0 to 5 random swaps per depot route
                 for depot_idx, route in particle_initial_position.stage1_routes.items():
                      if len(route) >= 2:
                           for _ in range(num_perturbation_swaps):
                                # Perform a random swap on this depot's route
                                idx1, idx2 = random.sample(range(len(route)), 2)
                                route[idx1], route[idx2] = route[idx2], route[idx1]
                           particle_initial_position.stage1_routes[depot_idx] = route # Update the route


            # Evaluate the initial position
            particle_initial_position.evaluate(
                distance_func=haversine,
                stage2_trip_generator_func=create_heuristic_trips_split_delivery,
                cost_weight=cost_weight,
                time_weight=time_weight,
                unmet_demand_penalty=unmet_demand_penalty
            )

            # Create the particle
            particle = Particle(initial_solution=particle_initial_position, num_depots=num_depots_in_problem)
            swarm.append(particle)

            # Update global best if this particle's initial position is better
            if gbest_solution is None or particle.position < gbest_solution:
                 gbest_solution = copy.deepcopy(particle.position)
                 gbest_cost = gbest_solution.weighted_cost
                 # print(f" Initial particle {i} is the first global best. Cost: {format_float(gbest_cost, 4)}, Feasible: {gbest_solution.is_feasible}")

    except Exception as e:
        print(f"Error during PSO swarm initialization: {e}")
        traceback.print_exc()
        return {'run_error': f"Swarm initialization failed: {e}"}


    # --- PSO Main Loop ---
    gbest_cost_history = [] # To store the global best weighted cost per iteration

    print("Starting PSO optimization loop...")
    for iteration in range(max_iterations):
        # print(f"\n--- Iteration {iteration + 1}/{max_iterations} ---")

        # Store current global best cost for history
        gbest_cost_history.append(gbest_cost)
        # print(f" Iteration {iteration + 1}: Gbest Cost: {format_float(gbest_cost, 4)}, Feasible: {gbest_solution.is_feasible if gbest_solution else False}")


        # Update velocity and position for each particle
        for i, particle in enumerate(swarm):
            try:
                # Update velocity based on pbest and gbest
                # This updates particle.velocity
                particle.update_velocity(gbest_solution, inertia_weight, cognitive_weight, social_weight)

                # Update position by applying the new velocity
                # This updates particle.position
                particle.update_position()

                # Evaluate the new position
                # This updates particle.position's evaluation results
                particle.position.evaluate(
                    distance_func=haversine,
                    stage2_trip_generator_func=create_heuristic_trips_split_delivery,
                    cost_weight=cost_weight, # Ensure evaluation uses correct weights
                    time_weight=time_weight,
                    unmet_demand_penalty=unmet_demand_penalty
                )

                # Update personal best (pbest)
                particle.update_pbest()

                # Update global best (gbest) if this particle's current position or pbest is better
                # Check current position first
                if gbest_solution is None or particle.position < gbest_solution:
                     gbest_solution = copy.deepcopy(particle.position)
                     gbest_cost = gbest_solution.weighted_cost
                     # print(f"  Iteration {iteration}: Found new global best from particle {i} position. Cost: {format_float(gbest_cost, 4)}, Feasible: {gbest_solution.is_feasible}")

                # Then check the particle's pbest
                if gbest_solution is None or particle.pbest < gbest_solution:
                     gbest_solution = copy.deepcopy(particle.pbest)
                     gbest_cost = gbest_solution.weighted_cost
                     # print(f"  Iteration {iteration}: Found new global best from particle {i} pbest. Cost: {format_float(gbest_cost, 4)}, Feasible: {gbest_solution.is_feasible}")

            except Exception as e:
                warnings.warn(f"Error processing particle {i} in iteration {iteration}: {e}")
                traceback.print_exc()
                # If particle update fails, it might be stuck or invalid.
                # Its position/velocity might not update correctly, affecting future iterations.
                # The evaluation error within the particle's evaluate method should mark its cost as inf.
                # The pbest/gbest updates will then ignore this particle if its cost is inf.
                # Continuing might be okay, but it's good to log the error.


        # Optional: Print progress periodically
        if (iteration + 1) % (max_iterations // 10 or 1) == 0: # Print roughly 10 times
            print(f"  Iteration {iteration + 1}/{max_iterations}: Gbest Cost={format_float(gbest_cost, 4)}, Gbest Feasible={gbest_solution.is_feasible if gbest_solution else False}")


    # --- PSO Finished ---
    end_time_pso = time.time()
    total_time_pso = end_time_pso - start_time_pso
    print(f"\nParticle Swarm Optimization (MD-SD) finished after {max_iterations} iterations in {total_time_pso:.4f} seconds.")

    # Final evaluation of the overall best solution found (gbest)
    if gbest_solution:
        print("Re-evaluating overall best solution found by PSO (gbest)...")
        try:
            gbest_solution.evaluate(
                distance_func=haversine,
                stage2_trip_generator_func=create_heuristic_trips_split_delivery,
                cost_weight=cost_weight, # Ensure final evaluation uses the requested weights
                time_weight=time_weight,
                unmet_demand_penalty=unmet_demand_penalty
            )
            print(f"PSO Final Best Evaluation: Feasible: {gbest_solution.is_feasible}, Weighted Cost: {format_float(gbest_solution.weighted_cost, 4)}")

            # Prepare the result dictionary
            pso_results = {
                'best_solution': gbest_solution, # Return the SolutionCandidate object
                'weighted_cost': gbest_solution.weighted_cost,
                'evaluated_cost': gbest_solution.evaluated_cost,
                'evaluated_time': gbest_solution.evaluated_time,
                'evaluated_unmet': gbest_solution.evaluated_unmet,
                'is_feasible': gbest_solution.is_feasible,
                'evaluation_stage1_error': gbest_solution.evaluation_stage1_error,
                'evaluation_stage2_error': gbest_solution.evaluation_stage2_error,
                'stage1_routes': gbest_solution.stage1_routes, # Include final routes
                'stage2_trips': gbest_solution.stage2_trips, # Include final trips
                'served_customer_details': gbest_solution.served_customer_details, # Include customer details
                'cost_history': gbest_cost_history, # Return the global best cost history
                'total_computation_time': total_time_pso,
                'algorithm_name': 'pso_optimizer',
                'algorithm_params': pso_params # Store parameters used
            }
            return pso_results

        except Exception as e:
            print(f"Error during final evaluation of PSO gbest solution: {e}")
            traceback.print_exc()
            # Return partial results with error indicated
            return {
                'best_solution': gbest_solution, # Return the object even if final eval failed
                'weighted_cost': float('inf'), # Indicate final evaluation failure
                'evaluated_cost': float('inf'),
                'evaluated_time': float('inf'),
                'evaluated_unmet': float('inf'),
                'is_feasible': False,
                'evaluation_stage1_error': True, # Assume error in final eval
                'evaluation_stage2_error': True, # Assume error in final eval
                'stage1_routes': gbest_solution.stage1_routes if gbest_solution else {}, # Return the routes found
                'stage2_trips': {}, # Stage 2 trips were not generated in final eval or failed
                'served_customer_details': {},
                'cost_history': gbest_cost_history,
                'total_computation_time': total_time_pso,
                'algorithm_name': 'pso_optimizer',
                'algorithm_params': pso_params,
                'run_error': f"Final evaluation failed: {e}"
            }

    else:
        print("Particle Swarm Optimization did not find a valid global best solution.")
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
             'cost_history': gbest_cost_history,
             'total_computation_time': total_time_pso,
             'algorithm_name': 'pso_optimizer',
             'algorithm_params': pso_params,
             'run_error': "No valid solution found by PSO."
        }


# --- Helper functions (Placeholder for format_float) ---
# Assuming format_float is available (e.g., defined in problem_utils or report_generator)
# Copying it here for standalone PSO execution testing, but prefer central definition.
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
     print("Using DUMMY create_initial_solution_mdsd for standalone PSO test.")
     def create_initial_solution_mdsd(problem_data, vehicle_params, drone_params, unmet_demand_penalty, cost_weight, time_weight):
          print("DUMMY create_initial_solution_mdsd called in PSO.")
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
               print(f"Error evaluating DUMMY initial solution in PSO: {eval_e}")
               dummy_candidate.is_feasible = False
               dummy_candidate.weighted_cost = float('inf')
               dummy_candidate.evaluation_stage1_error = True
               dummy_candidate.evaluation_stage2_error = True

          return dummy_candidate


# --- Optional Main Execution Block for Standalone Testing ---
if __name__ == '__main__':
    """
    Standalone execution block for testing the Particle Swarm Optimization algorithm.
    Requires dummy problem data and uses the dummy create_initial_solution_mdsd
    (or real if imported).
    """
    print("Running algorithm/pso_optimizer.py in standalone test mode.")

    # --- Create Dummy Problem Data ---
    # Needs to be sufficient for create_initial_solution_mdsd and subsequent evaluation
    try:
        print("\n--- Creating Dummy Problem Data for PSO Test ---")
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

        dummy_problem_data_pso = {
            'locations': dummy_locations,
            'demands': dummy_demands
        }

        dummy_vehicle_params_pso = {'payload': 200.0, 'cost_per_km': 1.5, 'speed_kmph': 60.0}
        dummy_drone_params_pso = {'payload': 30.0, 'max_flight_distance_km': 15.0, 'cost_per_km': 0.8, 'speed_kmph': 100.0}
        dummy_unmet_penalty_pso = 500.0
        dummy_cost_weight_pso = 1.0
        dummy_time_weight_pso = 0.1

        # Dummy PSO Parameters
        dummy_pso_params = {
            'num_particles': 10,  # Smaller swarm for faster test
            'max_iterations': 50, # Fewer iterations for faster test
            'inertia_weight': 0.8,
            'cognitive_weight': 1.5,
            'social_weight': 1.5
        }

        print("Dummy data and PSO parameters created.")

    except Exception as e:
        print(f"Error creating dummy data for PSO test: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Create a Dummy Initial Solution Candidate ---
    # Use the potentially dummy or real create_initial_solution_mdsd
    print("\n--- Creating Dummy Initial Solution Candidate ---")
    dummy_initial_solution_pso = create_initial_solution_mdsd(
        problem_data=dummy_problem_data_pso,
        vehicle_params=dummy_vehicle_params_pso,
        drone_params=dummy_drone_params_pso,
        unmet_demand_penalty=dummy_unmet_penalty_pso,
        cost_weight=dummy_cost_weight_pso,
        time_weight=dummy_time_weight_pso
    )

    if dummy_initial_solution_pso is None:
        print("Failed to create dummy initial solution. Cannot run PSO test.")
        sys.exit(1)
    else:
        print(f"Dummy initial solution created: Feasible={dummy_initial_solution_pso.is_feasible}, Weighted Cost={format_float(dummy_initial_solution_pso.weighted_cost, 4)}")
        print("Stage 1 Routes in initial solution:", dummy_initial_solution_pso.stage1_routes)


    # --- Run the PSO ---
    print("\n--- Running Particle Swarm Optimization (dummy data) ---")
    try:
        pso_results = run_pso_optimizer(
            problem_data=dummy_problem_data_pso,
            vehicle_params=dummy_vehicle_params_pso,
            drone_params=dummy_drone_params_pso,
            unmet_demand_penalty=dummy_unmet_penalty_pso,
            cost_weight=dummy_cost_weight_pso,
            time_weight=dummy_time_weight_pso,
            initial_solution_candidate=dummy_initial_solution_pso,
            algo_specific_params=dummy_pso_params
        )

        print("\n--- PSO Results Summary ---")
        if pso_results:
             print(f"Algorithm Name: {pso_results.get('algorithm_name')}")
             print(f"Run Time: {format_float(pso_results.get('total_computation_time'), 4)} seconds")
             if pso_results.get('run_error'):
                  print(f"Run Error: {pso_results.get('run_error')}")
             else:
                  best_solution = pso_results.get('best_solution')
                  if best_solution:
                       print("\nBest Solution Found (gbest):")
                       print(f"  Feasible: {best_solution.is_feasible}")
                       print(f"  Weighted Cost: {format_float(best_solution.weighted_cost, 4)}")
                       print(f"  Raw Cost: {format_float(best_solution.evaluated_cost, 2)}")
                       print(f"  Time (Makespan): {format_float(best_solution.evaluated_time, 2)}")
                       print(f"  Unmet Demand: {format_float(best_solution.evaluated_unmet, 2)}")
                       print("  Final Stage 1 Routes:", best_solution.stage1_routes)
                       # print("  Final Stage 2 Trips:", best_solution.stage2_trips) # Can be verbose
                       # print("  Served Customer Details:", best_solution.served_customer_details) # Can be verbose

                  print("\nCost History (Global Best per iteration):")
                  history = pso_results.get('cost_history', [])
                  # Print first few, mid few, and last few
                  if len(history) > 20: # Adjust number based on max_iterations
                       print([format_float(c, 4) for c in history[:5]] + ['...'] + [format_float(c, 4) for c in history[len(history)//2-2 : len(history)//2+3]] + ['...'] + [format_float(c, 4) for c in history[-5:]])
                  else:
                       print([format_float(c, 4) for c in history])


        else:
             print("Particle Swarm Optimization run failed or returned no results.")

    except Exception as e:
        print(f"An unexpected error occurred during PSO test execution: {e}")
        traceback.print_exc()

    print("\nStandalone test finished.")