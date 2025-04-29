# algorithm/genetic_algorithm.py
# -*- coding: utf-8 -*-
"""
Genetic Algorithm (GA) adapted for Multi-Depot, Two-Echelon VRP
with Drones and Split Deliveries (MD-2E-VRPSD).

Focuses on optimizing Stage 1 routes per depot using a permutation-based
representation. Each individual in the population is a SolutionCandidate object
holding the multi-depot Stage 1 route structure. Stage 2 trips (allowing splits
and using vehicles/drones) are generated heuristically during fitness evaluation
by the core cost function.

Incorporates feasibility priority in selection, elite strategy, and best
solution tracking using the SolutionCandidate's built-in comparison logic.
Uses adapted genetic operators (crossover and mutation) designed to work
with the multi-depot Stage 1 route representation.

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
    project_root_ga = os.path.dirname(current_dir)
    if project_root_ga not in sys.path:
        sys.path.insert(0, project_root_ga)
        # print(f"GA: Added project root to sys.path: {project_root_ga}") # Optional debug print

    # Import necessary components from core modules
    # Need SolutionCandidate, initial solution generator, neighbor generator (for mutation), permutation mutations
    from core.problem_utils import (
        SolutionCandidate,
        create_initial_solution_mdsd, # Potentially used if initial_solution_candidate is not passed fully formed
        generate_neighbor_solution_mdsd, # Needed for mutation-like operation
        swap_mutation,
        scramble_mutation,
        inversion_mutation,
        create_heuristic_trips_split_delivery # Needed by SolutionCandidate.evaluate indirectly
    )
    # Need distance and cost function references, but they are passed to SolutionCandidate.evaluate
    from core.distance_calculator import haversine # Needed by SolutionCandidate.evaluate indirectly
    from core.cost_function import calculate_total_cost_and_evaluate # Needed by SolutionCandidate.evaluate

except ImportError as e:
    print(f"CRITICAL ERROR in algorithm.genetic_algorithm: Failed during initial import block: {e}")
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
        print("DUMMY create_initial_solution_mdsd called in GA due to import error!")
        return None
    def generate_neighbor_solution_mdsd(solution):
        print("DUMMY generate_neighbor_solution_mdsd called in GA due to import error!")
        return None
    def swap_mutation(route): return list(route)
    def scramble_mutation(route): return list(route)
    def inversion_mutation(route): return list(route)
    def haversine(*args): return float('inf')
    def calculate_total_cost_and_evaluate(*args, **kwargs): return float('inf'), float('inf'), float('inf'), float('inf'), {}, True, True, {}
    def create_heuristic_trips_split_delivery(*args, **kwargs): return [] # Dummy

    print("Genetic Algorithm will use dummy functions/classes due to critical import failure.")

except Exception as e:
    print(f"An unexpected error occurred during GA import block: {e}")
    traceback.print_exc()
    # Define a dummy run function that indicates error
    def run_genetic_algorithm(*args, **kwargs):
        print("CRITICAL ERROR: Genetic Algorithm initialization failed.")
        return {'run_error': f"Initialization failed: {e}"}
    print("Genetic Algorithm will not run due to unexpected import error.")


# Define a small tolerance for floating-point comparisons (e.g., checking if demand is zero)
FLOAT_TOLERANCE_GA = 1e-6


def run_genetic_algorithm(problem_data: dict, vehicle_params: dict, drone_params: dict,
                          unmet_demand_penalty: float, cost_weight: float, time_weight: float,
                          initial_solution_candidate: SolutionCandidate,
                          algo_specific_params: dict) -> dict | None:
    """
    Runs the Genetic Algorithm (GA) to find a good solution for the
    Multi-Depot, Two-Echelon VRP with Drones and Split Deliveries.

    Optimizes the Stage 1 routes (depot to outlet sequence) for each depot
    within the SolutionCandidate structure.

    Args:
        problem_data (dict): Dictionary containing problem instance data.
        vehicle_params (dict): Dictionary of vehicle parameters.
        drone_params (dict): Dictionary of drone parameters.
        unmet_demand_penalty (float): Penalty cost per unit of unmet demand.
        cost_weight (float): Weight for raw cost in objective.
        time_weight (float): Weight for time/makespan in objective.
        initial_solution_candidate (SolutionCandidate): A base solution candidate
                                                        containing initial assignments
                                                        (outlets to depots, customers to outlets)
                                                        and a starting Stage 1 route structure
                                                        (e.g., from a greedy heuristic).
                                                        The GA will optimize the Stage 1 routes
                                                        within copies of this candidate.
        algo_specific_params (dict): Dictionary of GA-specific parameters:
                                     'population_size' (int), 'num_generations' (int),
                                     'mutation_rate' (float), 'crossover_rate' (float),
                                     'elite_count' (int), 'tournament_size' (int).

    Returns:
        dict | None: A dictionary containing the best solution found (as a SolutionCandidate
                     object or a dictionary representing its state), its evaluation results,
                     and the cost history per generation. Returns None or error dict
                     if the algorithm fails to run or find a valid solution.
        Example: {
            'best_solution': SolutionCandidate_object,
            'weighted_cost': float, 'evaluated_cost': float, 'evaluated_time': float,
            'evaluated_unmet': float, 'is_feasible': bool,
            'evaluation_stage1_error': bool, 'evaluation_stage2_error': bool,
            'stage1_routes': dict, 'stage2_trips': dict, 'served_customer_details': dict,
            'cost_history': list, # List of best weighted cost per generation
            'avg_cost_history': list, # Optional: List of average weighted cost per generation
            'total_computation_time': float,
            'algorithm_name': 'genetic_algorithm',
            'algorithm_params': dict # Parameters used
        }
    """
    print("\n--- Starting Genetic Algorithm (MD-SD) ---")
    start_time_ga = time.time()

    # --- Default GA Parameters ---
    # These can be overridden by algo_specific_params
    default_ga_params = {
        'population_size': 100,
        'num_generations': 500,
        'mutation_rate': 0.1, # Probability of mutation per individual
        'crossover_rate': 0.8, # Probability of crossover between two parents
        'elite_count': 5, # Number of best individuals to carry over directly
        'tournament_size': 5 # Size for tournament selection
    }

    # --- Parameter Validation and Extraction ---
    try:
        if not isinstance(initial_solution_candidate, SolutionCandidate):
            raise TypeError("'initial_solution_candidate' must be a SolutionCandidate object.")
        if not initial_solution_candidate.all_locations or not initial_solution_candidate.demands:
            raise ValueError("'initial_solution_candidate' does not contain valid problem data.")
        if not initial_solution_candidate.stage1_routes:
            warnings.warn("Initial solution candidate has no Stage 1 routes. GA may not be effective.")


        # Merge default and provided parameters
        ga_params = default_ga_params.copy()
        if isinstance(algo_specific_params, dict):
             ga_params.update(algo_specific_params)
        else:
             warnings.warn("'algo_specific_params' is not a dictionary. Using default GA parameters.")


        pop_size = ga_params.get('population_size')
        num_generations = ga_params.get('num_generations')
        mutation_rate = ga_params.get('mutation_rate')
        crossover_rate = ga_params.get('crossover_rate')
        elite_count = ga_params.get('elite_count')
        tournament_size = ga_params.get('tournament_size')

        # Validate GA parameters
        if not isinstance(pop_size, int) or pop_size <= 0: raise ValueError("population_size must be a positive integer.")
        if not isinstance(num_generations, int) or num_generations < 0: raise ValueError("num_generations must be a non-negative integer.")
        if not isinstance(mutation_rate, (int, float)) or not (0.0 <= mutation_rate <= 1.0): raise ValueError("mutation_rate must be between 0.0 and 1.0.")
        if not isinstance(crossover_rate, (int, float)) or not (0.0 <= crossover_rate <= 1.0): raise ValueError("crossover_rate must be between 0.0 and 1.0.")
        if not isinstance(elite_count, int) or elite_count < 0 or elite_count > pop_size: raise ValueError("elite_count must be between 0 and population_size.")
        if not isinstance(tournament_size, int) or tournament_size <= 0 or tournament_size > pop_size: raise ValueError("tournament_size must be between 1 and population_size.")


        print(f"GA Parameters: Population Size={pop_size}, Generations={num_generations}, Mutation Rate={mutation_rate}, Crossover Rate={crossover_rate}, Elite Count={elite_count}, Tournament Size={tournament_size}")

    except Exception as e:
        print(f"Error validating GA parameters or initial solution: {e}")
        traceback.print_exc()
        return {'run_error': f"Parameter or initial solution validation failed: {e}"}


    # --- GA Components ---
    # The core GA logic is implemented here, using the functions/classes from core.problem_utils

    def create_initial_population(base_solution: SolutionCandidate, size: int) -> list[SolutionCandidate]:
        """
        Creates the initial population by perturbing the Stage 1 routes
        of the base initial solution.

        Each individual is a deep copy of the base solution with its
        Stage 1 routes randomly shuffled or mutated.

        Args:
            base_solution (SolutionCandidate): The base solution structure
                                               with assignments and initial routes.
            size (int): The desired population size.

        Returns:
            list[SolutionCandidate]: A list of SolutionCandidate objects
                                     forming the initial population.
        """
        population = []
        print(f"Creating initial population of size {size}...")
        for i in range(size):
            try:
                # Create a deep copy of the base solution
                individual = copy.deepcopy(base_solution)

                # Perturb the Stage 1 routes of this individual
                # Randomly shuffle each depot's route
                for depot_idx in individual.stage1_routes:
                    if individual.stage1_routes[depot_idx]: # Only shuffle non-empty routes
                        random.shuffle(individual.stage1_routes[depot_idx])

                # The newly created individual needs to be evaluated
                # Evaluation will be done in the main GA loop after population creation
                # Initialize evaluation results to reflect it hasn't been evaluated yet
                individual.is_feasible = False
                individual.weighted_cost = float('inf')
                individual.evaluated_cost = float('inf')
                individual.evaluated_time = float('inf')
                individual.evaluated_unmet = float('inf')
                individual.served_customer_details = {}
                individual.evaluation_stage1_error = False
                individual.evaluation_stage2_error = False
                individual.stage2_trips = {}

                population.append(individual)
            except Exception as e:
                 warnings.warn(f"Error creating individual {i} in initial population: {e}")
                 traceback.print_exc()
                 # Optionally add a dummy invalid solution or just continue

        return population


    def evaluate_population(population: list[SolutionCandidate],
                            distance_func: callable,
                            stage2_trip_generator_func: callable,
                            cost_weight: float, time_weight: float, unmet_demand_penalty: float):
        """
        Evaluates the fitness of each individual in the population.

        Calls the evaluate method for each SolutionCandidate, which in turn
        uses the core cost function and Stage 2 trip generator.

        Args:
            population (list[SolutionCandidate]): The population to evaluate.
            distance_func (callable): Function for distance calculation.
            stage2_trip_generator_func (callable): Function for Stage 2 trip generation.
            cost_weight (float): Weight for raw cost.
            time_weight (float): Weight for time.
            unmet_demand_penalty (float): Penalty for unmet demand.
        """
        # print("Evaluating population...")
        for individual in population:
            try:
                 individual.evaluate(
                     distance_func=distance_func,
                     stage2_trip_generator_func=stage2_trip_generator_func,
                     cost_weight=cost_weight, # Ensure evaluation uses correct weights
                     time_weight=time_weight,
                     unmet_demand_penalty=unmet_demand_penalty
                 )
            except Exception as e:
                 warnings.warn(f"Error evaluating individual: {e}")
                 traceback.print_exc()
                 # If evaluation fails, mark the individual as unfeasible with infinite cost
                 individual.is_feasible = False
                 individual.weighted_cost = float('inf')
                 individual.evaluated_cost = float('inf')
                 individual.evaluated_time = float('inf')
                 individual.evaluated_unmet = float('inf')
                 individual.evaluation_stage1_error = True
                 individual.evaluation_stage2_error = True


    def select_parents(population: list[SolutionCandidate], num_parents: int, tournament_size: int) -> list[SolutionCandidate]:
        """
        Selects parents from the population using Tournament Selection.

        Args:
            population (list[SolutionCandidate]): The current population.
            num_parents (int): The number of parents to select.
            tournament_size (int): The size of the tournament (number of individuals competing).

        Returns:
            list[SolutionCandidate]: A list of selected parent SolutionCandidate objects.
        """
        selected = []
        pop_size = len(population)
        if pop_size == 0:
             warnings.warn("Cannot select parents from empty population.")
             return []
        if tournament_size > pop_size:
             warnings.warn(f"Tournament size ({tournament_size}) is larger than population size ({pop_size}). Setting tournament size to population size.")
             tournament_size = pop_size
        if tournament_size == 0: # Prevent infinite loop if tournament_size becomes 0
            tournament_size = 1 # Minimum size 1

        # print(f"Selecting {num_parents} parents using tournament selection (size {tournament_size})...")
        for _ in range(num_parents):
            # Select 'tournament_size' random individuals for the tournament
            tournament_competitors = random.sample(population, tournament_size)
            # The winner is the best individual among the competitors (using SolutionCandidate.__lt__)
            winner = min(tournament_competitors) # min() uses the __lt__ method
            selected.append(winner) # Append the actual winner object

        return selected


    def crossover(parent1: SolutionCandidate, parent2: SolutionCandidate, crossover_rate: float) -> tuple[SolutionCandidate, SolutionCandidate]:
        """
        Applies crossover to two parent SolutionCandidates to produce two offspring.

        Uses a multi-depot adapted Ordered Crossover (OX1) operator.
        Randomly selects one depot's Stage 1 route in the parents and applies
        OX1 to those two specific routes. The other depot routes are inherited
        from the parents.

        Args:
            parent1 (SolutionCandidate): The first parent.
            parent2 (SolutionCandidate): The second parent.
            crossover_rate (float): The probability of applying crossover.

        Returns:
            tuple[SolutionCandidate, SolutionCandidate]: Two offspring SolutionCandidate objects.
                                                        If crossover does not occur, returns copies of parents.
        """
        # Create offspring as copies of parents by default
        offspring1 = copy.deepcopy(parent1)
        offspring2 = copy.deepcopy(parent2)

        if random.random() < crossover_rate:
            # print(" Applying crossover...")
            # Select a random depot index to perform crossover on its route
            depot_indices = list(parent1.stage1_routes.keys())
            if not depot_indices:
                # print(" No depots available for crossover.")
                return offspring1, offspring2 # Cannot perform crossover if no depots

            selected_depot_index = random.choice(depot_indices)

            parent1_route = parent1.stage1_routes.get(selected_depot_index, [])
            parent2_route = parent2.stage1_routes.get(selected_depot_index, [])

            # Only perform crossover if both routes for the selected depot are non-empty and have length >= 2
            if len(parent1_route) >= 2 and len(parent2_route) >= 2:
                 try:
                     # Apply OX1 crossover to the selected routes
                     # The ox1_crossover function needs to be defined below or imported
                     child1_route_partial, child2_route_partial = ox1_crossover(parent1_route, parent2_route)

                     # Replace the corresponding routes in the offspring solutions
                     offspring1.stage1_routes[selected_depot_index] = child1_route_partial
                     offspring2.stage1_routes[selected_depot_index] = child2_route_partial

                     # Mark offspring as unevaluated
                     for off in [offspring1, offspring2]:
                         off.is_feasible = False
                         off.weighted_cost = float('inf')
                         off.evaluated_cost = float('inf')
                         off.evaluated_time = float('inf')
                         off.evaluated_unmet = float('inf')
                         off.served_customer_details = {}
                         off.evaluation_stage1_error = False
                         off.evaluation_stage2_error = False
                         off.stage2_trips = {}


                 except Exception as e:
                     warnings.warn(f"Error during OX1 crossover for depot {selected_depot_index}: {e}")
                     traceback.print_exc()
                     # If crossover fails, return copies of parents as offspring
                     offspring1 = copy.deepcopy(parent1)
                     offspring2 = copy.deepcopy(parent2)
                     # Ensure they are marked as unevaluated (unless they were already)
                     for off in [offspring1, offspring2]:
                          if not off.weighted_cost == float('inf'): # Only reset if previously evaluated
                               off.is_feasible = False
                               off.weighted_cost = float('inf')
                               off.evaluated_cost = float('inf')
                               off.evaluated_time = float('inf')
                               off.evaluated_unmet = float('inf')
                               off.served_customer_details = {}
                               off.evaluation_stage1_error = False
                               off.evaluation_stage2_error = False
                               off.stage2_trips = {}


            # else:
                # print(f" Skipping crossover for depot {selected_depot_index} (route too short or empty).")


        return offspring1, offspring2


    def mutate(individual: SolutionCandidate, mutation_rate: float) -> SolutionCandidate:
        """
        Applies mutation to an individual SolutionCandidate.

        Randomly selects one depot's Stage 1 route and applies a randomly chosen
        permutation mutation operator from core.problem_utils (swap, scramble, inversion)
        with a certain probability.

        Args:
            individual (SolutionCandidate): The individual to mutate.
            mutation_rate (float): The probability of applying mutation.

        Returns:
            SolutionCandidate: The mutated individual (a new object).
        """
        mutated_individual = copy.deepcopy(individual) # Work on a copy

        if random.random() < mutation_rate:
            # print(" Applying mutation...")
            # Select a random depot index to mutate its route
            depot_indices = list(mutated_individual.stage1_routes.keys())
            if not depot_indices:
                 # print(" No depots available for mutation.")
                 return mutated_individual # Cannot mutate if no depots

            selected_depot_index = random.choice(depot_indices)
            route_to_mutate = mutated_individual.stage1_routes.get(selected_depot_index, [])

            if route_to_mutate: # Only mutate non-empty routes
                 # Select a random mutation operator from core.problem_utils
                 mutation_operators = [swap_mutation, scramble_mutation, inversion_mutation]
                 selected_operator = random.choice(mutation_operators)

                 try:
                     # Apply the selected mutation operator to the route
                     mutated_route = selected_operator(route_to_mutate)
                     mutated_individual.stage1_routes[selected_depot_index] = mutated_route

                     # Mark the mutated individual as unevaluated
                     mutated_individual.is_feasible = False
                     mutated_individual.weighted_cost = float('inf')
                     mutated_individual.evaluated_cost = float('inf')
                     mutated_individual.evaluated_time = float('inf')
                     mutated_individual.evaluated_unmet = float('inf')
                     mutated_individual.served_customer_details = {}
                     mutated_individual.evaluation_stage1_error = False
                     mutated_individual.evaluation_stage2_error = False
                     mutated_individual.stage2_trips = {}

                     # print(f" Mutated depot {selected_depot_index} route using {selected_operator.__name__}.")


                 except Exception as e:
                     warnings.warn(f"Error during mutation for depot {selected_depot_index} using {selected_operator.__name__}: {e}")
                     traceback.print_exc()
                     # If mutation fails, return the unmutated copy, but mark as unevaluated
                     if not mutated_individual.weighted_cost == float('inf'): # Only reset if previously evaluated
                          mutated_individual.is_feasible = False
                          mutated_individual.weighted_cost = float('inf')
                          mutated_individual.evaluated_cost = float('inf')
                          mutated_individual.evaluated_time = float('inf')
                          mutated_individual.evaluated_unmet = float('inf')
                          mutated_individual.served_customer_details = {}
                          mutated_individual.evaluation_stage1_error = False
                          mutated_individual.evaluation_stage2_error = False
                          mutated_individual.stage2_trips = {}


            # else:
                 # print(f" Skipping mutation for depot {selected_depot_index} (route is empty).")


        return mutated_individual


    # --- OX1 Crossover Implementation (Adapted for single route) ---
    # This function is called by the 'crossover' function defined above.
    def ox1_crossover(parent1_route: list, parent2_route: list) -> tuple[list, list]:
        """
        Applies Ordered Crossover (OX1) to two parent lists (permutations)
        representing Stage 1 routes.

        Selects a random segment from parent1 and copies it to child1.
        The remaining elements in child1 are filled in the order they appear
        in parent2, starting after the copied segment, skipping elements
        already in the segment. Child2 is generated symmetrically.

        Args:
            parent1_route (list): The list representing the first parent's route.
            parent2_route (list): The list representing the second parent's route.

        Returns:
            tuple[list, list]: Two new lists representing the offspring routes.
                               Returns copies of input routes if they are too short.
        """
        size = len(parent1_route) # Assume parent routes have the same size (number of assigned outlets)
        if size < 2: # Need at least 2 elements to select a segment
             return list(parent1_route), list(parent2_route) # Return copies if too short


        # Select two random cut points
        cut1 = random.randint(0, size - 1)
        cut2 = random.randint(0, size - 1)

        # Ensure cut1 is less than cut2 by swapping if necessary
        if cut1 > cut2:
            cut1, cut2 = cut2, cut1

        # If cut points are the same, adjust one to ensure a segment of size >= 1
        if cut1 == cut2:
             if cut2 < size - 1:
                  cut2 += 1
             elif cut1 > 0:
                  cut1 -= 1
             # If size is 1, cut1=cut2=0, cannot form a segment >=1, handled by size < 2 check


        # Create offspring lists initialized with placeholder (e.g., None)
        # Offspring will have the same structure as parents (lists of outlet indices)
        child1_route = [None] * size
        child2_route = [None] * size

        # Copy the segment from parent1 to child1, and parent2 to child2
        child1_route[cut1 : cut2 + 1] = parent1_route[cut1 : cut2 + 1]
        child2_route[cut1 : cut2 + 1] = parent2_route[cut1 : cut2 + 1]

        # Fill the remaining positions in child1
        # Iterate through parent2 starting from cut2 + 1 (wrapping around)
        # Add elements not already in the segment in child1
        current_parent2_pos = (cut2 + 1) % size
        current_child1_pos = (cut2 + 1) % size
        elements_in_segment1 = set(child1_route[cut1 : cut2 + 1]) # Elements already in child 1's segment

        while None in child1_route:
            element_from_parent2 = parent2_route[current_parent2_pos]

            # Check if this element from parent2 is already in child1's copied segment
            if element_from_parent2 not in elements_in_segment1:
                # If not, add it to the next available position in child1
                child1_route[current_child1_pos] = element_from_parent2
                current_child1_pos = (current_child1_pos + 1) % size # Move to the next fill position

            # Move to the next element in parent2
            current_parent2_pos = (current_parent2_pos + 1) % size


        # Fill the remaining positions in child2 (symmetric to child1)
        # Iterate through parent1 starting from cut2 + 1 (wrapping around)
        # Add elements not already in the segment in child2
        current_parent1_pos = (cut2 + 1) % size
        current_child2_pos = (cut2 + 1) % size
        elements_in_segment2 = set(child2_route[cut1 : cut2 + 1]) # Elements already in child 2's segment

        while None in child2_route:
            element_from_parent1 = parent1_route[current_parent1_pos]

            # Check if this element from parent1 is already in child2's copied segment
            if element_from_parent1 not in elements_in_segment2:
                # If not, add it to the next available position in child2
                child2_route[current_child2_pos] = element_from_parent1
                current_child2_pos = (current_child2_pos + 1) % size # Move to the next fill position

            # Move to the next element in parent1
            current_parent1_pos = (current_parent1_pos + 1) % size

        return child1_route, child2_route


    # --- Main GA Loop ---
    cost_history = [] # To store the best weighted cost per generation
    avg_cost_history = [] # To store the average weighted cost per generation (optional)
    best_solution_overall = None # To track the best SolutionCandidate found across all generations

    # 1. Create initial population
    try:
        population = create_initial_population(base_solution=initial_solution_candidate, size=pop_size)
        if not population:
             raise RuntimeError("Initial population creation failed.")
    except Exception as e:
        print(f"Error creating initial population: {e}")
        traceback.print_exc()
        return {'run_error': f"Initial population creation failed: {e}"}


    # 2. Main evolution loop
    print("Starting GA evolution loop...")
    for generation in range(num_generations):
        # print(f"\n--- Generation {generation + 1}/{num_generations} ---")

        # Evaluate the current population
        try:
            evaluate_population(
                population=population,
                distance_func=haversine, # Pass the real distance function
                stage2_trip_generator_func=create_heuristic_trips_split_delivery, # Pass the real Stage 2 generator
                cost_weight=cost_weight,
                time_weight=time_weight,
                unmet_demand_penalty=unmet_demand_penalty
            )
            # Sort population by fitness (ascending weighted cost, feasible first)
            population.sort() # Uses the __lt__ method of SolutionCandidate

        except Exception as e:
            warnings.warn(f"Error during evaluation or sorting in generation {generation}: {e}")
            traceback.print_exc()
            # Continue to next generation, but population might be in a bad state


        # Track best and average cost for history
        current_best_solution = population[0] if population else None # Best is the first after sorting
        if current_best_solution:
             cost_history.append(current_best_solution.weighted_cost)
             # Update overall best solution found so far
             if best_solution_overall is None or current_best_solution < best_solution_overall:
                  best_solution_overall = copy.deepcopy(current_best_solution)
                  # print(f" New best solution found in generation {generation + 1}: {best_solution_overall}")
        else:
             # Handle case where population is empty or evaluation failed for all
             cost_history.append(float('inf'))


        # Calculate average cost (only for valid solutions)
        valid_costs = [ind.weighted_cost for ind in population if ind.weighted_cost is not None and not math.isinf(ind.weighted_cost) and not math.isnan(ind.weighted_cost)]
        avg_cost_history.append(sum(valid_costs) / len(valid_costs) if valid_costs else float('inf'))

        # print(f" Gen {generation + 1}: Best Cost: {cost_history[-1]:.4f}, Avg Cost: {avg_cost_history[-1]:.4f}, Feasible: {current_best_solution.is_feasible if current_best_solution else False}")


        # Create the next generation
        next_population = []

        # Add elites to the next generation (elitism)
        # Ensure we don't add more elites than available individuals
        num_elites_to_transfer = min(elite_count, len(population))
        # Deep copy elites to prevent modification in subsequent steps affecting the next generation's starting point
        elites = [copy.deepcopy(population[i]) for i in range(num_elites_to_transfer)]
        next_population.extend(elites)
        # print(f" Added {num_elites_to_transfer} elites.")


        # Fill the rest of the population through selection, crossover, and mutation
        num_individuals_to_generate = pop_size - num_elites_to_transfer
        if num_individuals_to_generate < 0: num_individuals_to_generate = 0 # Should not happen if elite_count <= pop_size

        num_pairs_for_crossover = num_individuals_to_generate // 2 # Generate offspring in pairs

        if num_pairs_for_crossover > 0:
             # Select parents for crossover (select 2 * num_pairs_for_crossover parents)
             parents = select_parents(population, 2 * num_pairs_for_crossover, tournament_size)
             if len(parents) < 2 * num_pairs_for_crossover:
                 warnings.warn(f"Could not select enough parents ({len(parents)}/{2 * num_pairs_for_crossover}). Generating fewer offspring.")
                 num_pairs_for_crossover = len(parents) // 2 # Adjust if not enough parents


             # Perform crossover and mutation
             for i in range(num_pairs_for_crossover):
                 try:
                     parent1 = parents[2 * i]
                     parent2 = parents[2 * i + 1]

                     # Apply crossover
                     offspring1, offspring2 = crossover(parent1, parent2, crossover_rate)

                     # Apply mutation to each offspring individually
                     mutated_offspring1 = mutate(offspring1, mutation_rate)
                     mutated_offspring2 = mutate(offspring2, mutation_rate)

                     next_population.extend([mutated_offspring1, mutated_offspring2])

                 except Exception as e:
                      warnings.warn(f"Error during crossover/mutation for parent pair {i} in generation {generation}: {e}")
                      traceback.print_exc()
                      # Add copies of parents as fallback if genetic operators fail
                      # Ensure they are marked as unevaluated
                      p1_copy = copy.deepcopy(parent1)
                      p2_copy = copy.deepcopy(parent2)
                      for p in [p1_copy, p2_copy]:
                           p.is_feasible = False; p.weighted_cost = float('inf'); p.evaluated_cost = float('inf'); p.evaluated_time = float('inf'); p.evaluated_unmet = float('inf'); p.served_customer_details = {}; p.evaluation_stage1_error = False; p.evaluation_stage2_error = False; p.stage2_trips = {}
                      next_population.extend([p1_copy, p2_copy])


             # If num_individuals_to_generate is odd, we might have one parent left
             # or need to generate one more individual. A simple way is to select
             # one more parent and add a mutated copy of it.
             if len(next_population) < pop_size:
                  # Select one more parent and mutate it
                  extra_parent = select_parents(population, 1, tournament_size)[0]
                  mutated_extra_offspring = mutate(copy.deepcopy(extra_parent), mutation_rate)
                  next_population.append(mutated_extra_offspring)
                  # print(" Added one extra mutated offspring.")


        # Ensure the next population size is exactly pop_size
        # If more individuals were generated due to error handling or odd counts, truncate
        # If fewer were generated (e.g., due to selection/operator failure), fill with copies of best individual
        while len(next_population) < pop_size:
             warnings.warn(f"Next population size ({len(next_population)}) less than target ({pop_size}). Filling with copies of best individual.")
             best_individual_copy = copy.deepcopy(population[0]) if population else None
             if best_individual_copy:
                  # Ensure copied best individual is marked for re-evaluation in the next generation's loop
                  best_individual_copy.is_feasible = False
                  best_individual_copy.weighted_cost = float('inf')
                  best_individual_copy.evaluated_cost = float('inf')
                  best_individual_copy.evaluated_time = float('inf')
                  best_individual_copy.evaluated_unmet = float('inf')
                  best_individual_copy.served_customer_details = {}
                  best_individual_copy.evaluation_stage1_error = False
                  best_individual_copy.evaluation_stage2_error = False
                  best_individual_copy.stage2_trips = {}
                  next_population.append(best_individual_copy)
             else:
                 # If even the best is None, add a dummy invalid solution
                 warnings.warn("Population empty, cannot copy best individual. Adding dummy invalid solution.")
                 dummy_invalid = SolutionCandidate(problem_data={}, vehicle_params={}, drone_params={}, unmet_demand_penalty=float('inf'))
                 dummy_invalid.is_feasible = False; dummy_invalid.weighted_cost = float('inf'); dummy_invalid.evaluation_stage1_error = True; dummy_invalid.evaluation_stage2_error = True
                 next_population.append(dummy_invalid)


        if len(next_population) > pop_size:
            # This should theoretically not happen with num_individuals_to_generate logic,
            # but as a safeguard, truncate if somehow overfilled.
            next_population = next_population[:pop_size]
            warnings.warn(f"Next population size ({len(next_population)}) exceeded target ({pop_size}). Truncating.")


        # Replace the current population with the new generation
        population = next_population

    # --- GA Finished ---
    end_time_ga = time.time()
    total_time_ga = end_time_ga - start_time_ga
    print(f"\nGenetic Algorithm (MD-SD) finished after {num_generations} generations in {total_time_ga:.4f} seconds.")

    # Final evaluation of the overall best solution found (if any)
    if best_solution_overall:
        print("Re-evaluating overall best solution found by GA...")
        try:
            best_solution_overall.evaluate(
                distance_func=haversine,
                stage2_trip_generator_func=create_heuristic_trips_split_delivery,
                cost_weight=cost_weight, # Ensure final evaluation uses the requested weights
                time_weight=time_weight,
                unmet_demand_penalty=unmet_demand_penalty
            )
            print(f"GA Final Best Evaluation: Feasible: {best_solution_overall.is_feasible}, Weighted Cost: {best_solution_overall.weighted_cost:.4f}")

            # Prepare the result dictionary
            ga_results = {
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
                'cost_history': cost_history, # Return the cost history
                'avg_cost_history': avg_cost_history, # Return average cost history
                'total_computation_time': total_time_ga,
                'algorithm_name': 'genetic_algorithm',
                'algorithm_params': ga_params # Store parameters used
            }
            return ga_results

        except Exception as e:
            print(f"Error during final evaluation of GA best solution: {e}")
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
                'avg_cost_history': avg_cost_history,
                'total_computation_time': total_time_ga,
                'algorithm_name': 'genetic_algorithm',
                'algorithm_params': ga_params,
                'run_error': f"Final evaluation failed: {e}"
            }

    else:
        print("Genetic Algorithm did not find a valid best solution.")
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
             'avg_cost_history': avg_cost_history,
             'total_computation_time': total_time_ga,
             'algorithm_name': 'genetic_algorithm',
             'algorithm_params': ga_params,
             'run_error': "No valid solution found by GA."
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
     print("Using DUMMY create_initial_solution_mdsd for standalone GA test.")
     def create_initial_solution_mdsd(problem_data, vehicle_params, drone_params, unmet_demand_penalty, cost_weight, time_weight):
          print("DUMMY create_initial_solution_mdsd called in GA.")
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
               dummy_candidate.evaluate(haversine, create_heuristic_trips_split_delivery, cost_weight, time_weight, unmet_demand_penalty)
          except Exception as eval_e:
               print(f"Error evaluating DUMMY initial solution: {eval_e}")
               dummy_candidate.is_feasible = False
               dummy_candidate.weighted_cost = float('inf')
               dummy_candidate.evaluation_stage1_error = True
               dummy_candidate.evaluation_stage2_error = True

          return dummy_candidate


# --- Optional Main Execution Block for Standalone Testing ---
if __name__ == '__main__':
    """
    Standalone execution block for testing the Genetic Algorithm.
    Requires dummy problem data and uses the dummy create_initial_solution_mdsd
    and the dummy Stage 2 generator (or real if imported).
    """
    print("Running algorithm/genetic_algorithm.py in standalone test mode.")

    # --- Create Dummy Problem Data ---
    # Needs to be sufficient for create_initial_solution_mdsd and subsequent evaluation
    try:
        print("\n--- Creating Dummy Problem Data for GA Test ---")
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

        dummy_problem_data_ga = {
            'locations': dummy_locations,
            'demands': dummy_demands
        }

        dummy_vehicle_params_ga = {'payload': 200.0, 'cost_per_km': 1.5, 'speed_kmph': 60.0}
        dummy_drone_params_ga = {'payload': 30.0, 'max_flight_distance_km': 15.0, 'cost_per_km': 0.8, 'speed_kmph': 100.0}
        dummy_unmet_penalty_ga = 500.0
        dummy_cost_weight_ga = 1.0
        dummy_time_weight_ga = 0.1

        # Dummy GA Parameters
        dummy_ga_params = {
            'population_size': 20,  # Smaller population for faster test
            'num_generations': 10, # Fewer generations for faster test
            'mutation_rate': 0.2,
            'crossover_rate': 0.8,
            'elite_count': 2,
            'tournament_size': 4
        }

        print("Dummy data and GA parameters created.")

    except Exception as e:
        print(f"Error creating dummy data for GA test: {e}")
        traceback.print_exc()
        sys.exit(1)

    # --- Create a Dummy Initial Solution Candidate ---
    # Use the potentially dummy or real create_initial_solution_mdsd
    print("\n--- Creating Dummy Initial Solution Candidate ---")
    dummy_initial_solution = create_initial_solution_mdsd(
        problem_data=dummy_problem_data_ga,
        vehicle_params=dummy_vehicle_params_ga,
        drone_params=dummy_drone_params_ga,
        unmet_demand_penalty=dummy_unmet_penalty_ga,
        cost_weight=dummy_cost_weight_ga,
        time_weight=dummy_time_weight_ga
    )

    if dummy_initial_solution is None:
        print("Failed to create dummy initial solution. Cannot run GA test.")
        sys.exit(1)
    else:
        print(f"Dummy initial solution created: Feasible={dummy_initial_solution.is_feasible}, Weighted Cost={format_float(dummy_initial_solution.weighted_cost, 4)}")
        print("Stage 1 Routes in initial solution:", dummy_initial_solution.stage1_routes)


    # --- Run the GA ---
    print("\n--- Running Genetic Algorithm (dummy data) ---")
    try:
        ga_results = run_genetic_algorithm(
            problem_data=dummy_problem_data_ga,
            vehicle_params=dummy_vehicle_params_ga,
            drone_params=dummy_drone_params_ga,
            unmet_demand_penalty=dummy_unmet_penalty_ga,
            cost_weight=dummy_cost_weight_ga,
            time_weight=dummy_time_weight_ga,
            initial_solution_candidate=dummy_initial_solution,
            algo_specific_params=dummy_ga_params
        )

        print("\n--- GA Results Summary ---")
        if ga_results:
             print(f"Algorithm Name: {ga_results.get('algorithm_name')}")
             print(f"Run Time: {format_float(ga_results.get('total_computation_time'), 4)} seconds")
             if ga_results.get('run_error'):
                  print(f"Run Error: {ga_results.get('run_error')}")
             else:
                  best_solution = ga_results.get('best_solution')
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

                  print("\nCost History (Best per generation):")
                  # print(ga_results.get('cost_history', [])) # Print full list
                  # Print first few and last few
                  history = ga_results.get('cost_history', [])
                  if len(history) > 10:
                       print(history[:5] + ['...'] + history[-5:])
                  else:
                       print(history)

                  # Optional: Print average cost history
                  # print("\nAverage Cost History per generation:")
                  # avg_history = ga_results.get('avg_cost_history', [])
                  # if len(avg_history) > 10:
                  #      print([format_float(c, 4) for c in avg_history[:5]] + ['...'] + [format_float(c, 4) for c in avg_history[-5:]])
                  # else:
                  #      print([format_float(c, 4) for c in avg_history])

        else:
             print("Genetic Algorithm run failed or returned no results.")

    except Exception as e:
        print(f"An unexpected error occurred during GA test execution: {e}")
        traceback.print_exc()

    print("\nStandalone test finished.")