# visualization/plot_generator.py
# -*- coding: utf-8 -*-
"""
Generates Matplotlib plots for visualizing optimization algorithm results,
designed for embedding within a Tkinter GUI canvas or saving standalone.

Provides methods to plot:
- Cost convergence curves over iterations/generations.
- Bar charts comparing final weighted cost and Total Delivery Time (Makespan)
  across algorithms.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
import math
import warnings
from typing import Dict, Any, List, Tuple, Optional, Union

# Configure logging for this module
import logging
logger = logging.getLogger(__name__)
# Basic config if run standalone or not configured by main app
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


# --- Type Hinting Setup ---
# Define a placeholder or import the actual SolutionCandidate class if possible
# This helps with type checking and code readability.
try:
    # Adjust the import path based on your actual project structure
    from core.problem_utils import SolutionCandidate as ActualSolutionCandidate
    SolutionCandidateType = ActualSolutionCandidate
    logger.debug("Successfully imported SolutionCandidate for type hinting.")
except ImportError:
    logger.warning("Could not import SolutionCandidate from core.problem_utils. Using generic type hints.")
    # Define a simple placeholder class if the real one can't be imported
    class _DummySolutionCandidate:
        total_cost: Optional[float] = None
        total_time: Optional[float] = None # Represents Makespan
        weighted_cost: Optional[float] = None
        is_feasible: Optional[bool] = None
        cost_history: Optional[List[float]] = None
        # Add other potential attributes if needed for type hinting elsewhere
        evaluation_stage1_error: Optional[bool] = None
        evaluation_stage2_error: Optional[bool] = None
        served_customer_details: Optional[Dict] = None
        stage1_routes: Optional[Dict] = None
        stage2_trips: Optional[Dict] = None

    SolutionCandidateType = _DummySolutionCandidate # Use the dummy for type hints


# Define consistent styling for algorithms
# Ensure these keys match the algorithm keys used in route_optimizer.py
DEFAULT_ALGO_STYLES = {
    'genetic_algorithm': {'color': 'blue', 'marker': 'o', 'name': 'Genetic Algorithm'},
    'greedy_heuristic': {'color': 'grey', 'marker': 's', 'name': 'Greedy Heuristic'},
    'simulated_annealing': {'color': 'orange', 'marker': '^', 'name': 'Simulated Annealing'},
    'pso_optimizer': {'color': 'purple', 'marker': 'p', 'name': 'PSO'},
    # Add styles for other potential algorithms here
    'default': {'color': 'red', 'marker': 'x', 'name': 'Unknown Algorithm'}
}

# Define a small tolerance for floating-point comparisons
FLOAT_TOLERANCE_PLOT = 1e-9


class PlotGenerator:
    """
    Handles the generation and customization of Matplotlib plots for visualizing
    logistics optimization algorithm performance and results.
    """

    def __init__(self, algorithm_styles: Optional[Dict[str, Dict[str, str]]] = None):
        """
        Initializes the PlotGenerator.

        Args:
            algorithm_styles: An optional dictionary defining custom styles
                              (color, marker, name) for algorithm keys.
                              If None, uses DEFAULT_ALGO_STYLES.
        """
        self.algo_styles = algorithm_styles if algorithm_styles else DEFAULT_ALGO_STYLES
        logger.debug("PlotGenerator initialized.")

    def _get_algo_style(self, algo_key: str) -> Tuple[str, str, str]:
        """Safely retrieves color, marker, and display name for a given algorithm key."""
        style = self.algo_styles.get(algo_key, self.algo_styles['default'])
        color = style.get('color', self.algo_styles['default']['color'])
        marker = style.get('marker', self.algo_styles['default']['marker'])
        name = style.get('name', self.algo_styles['default']['name'])
        return color, marker, name

    def _plot_no_data_message(self, ax: plt.Axes, message: str):
        """Helper function to display a message on an Axes object when no data is available."""
        if ax:
            ax.clear()
            ax.text(0.5, 0.5, message,
                    horizontalalignment='center', verticalalignment='center',
                    transform=ax.transAxes, color='grey', fontsize='large', wrap=True)
            try:
                 # Redraw the canvas if possible (GUI context)
                 ax.figure.canvas.draw_idle()
            except AttributeError:
                 # Ignore if not in a canvas environment (e.g., saving directly)
                 pass
            except Exception as e:
                 logger.warning(f"Error redrawing canvas for no data message: {e}")

    # ========================================================================
    # plot_iteration_curves - Plots algorithm convergence history
    # ========================================================================
    def plot_iteration_curves(self,
                              results_by_algorithm: Dict[str, Dict[str, Any]],
                              ax: plt.Axes):
        """
        Plots the cost convergence history (typically best weighted cost found so far)
        for multiple algorithms on a given Axes object.

        Handles cases where algorithms provide cost history lists or only a final value.
        Scales the x-axis (iterations/generations) as a percentage of the maximum length found.
        Uses a log scale for the y-axis if appropriate.

        Args:
            results_by_algorithm: The results dictionary from route_optimizer. Expected structure:
                                  {'algo_key': {'result_data': SolutionCandidateType_or_dict, 'run_error': Optional[str], ...}, ...}
                                  The 'result_data' should ideally contain a 'cost_history' list.
                                  If not, it falls back to using the final 'weighted_cost'.
            ax: The matplotlib.axes.Axes object to plot on.
        """
        if not ax:
            logger.error("Plotting Error: No Matplotlib Axes provided for iteration curves.")
            return

        logger.info("Generating iteration curve plot...")
        ax.clear()
        ax.set_title('Algorithm Cost Convergence')
        ax.set_xlabel('Iteration / Generation (Scaled %)')
        ax.set_ylabel('Best Weighted Cost Found') # Initial label, may change based on scale
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)

        max_len_effective = 0 # Max number of *valid* data points across algorithms
        plot_data_available = False
        plotted_lines = [] # Keep track of lines plotted for the legend

        # --- First Pass: Check data availability and determine max length for scaling ---
        for algo_key, result_summary in results_by_algorithm.items():
            result_data = result_summary.get('result_data')
            if not result_data or result_summary.get('run_error'):
                 logger.debug(f"Skipping iteration plot for '{algo_key}' (no result_data or run error).")
                 continue

            cost_history = None
            if isinstance(result_data, dict): # Handle dict case
                cost_history = result_data.get('cost_history')
            elif hasattr(result_data, 'cost_history'): # Handle object case
                cost_history = getattr(result_data, 'cost_history', None)

            if cost_history and isinstance(cost_history, list):
                # Filter out None, NaN, Inf values from history for length calculation
                valid_history_points = [c for c in cost_history if c is not None and math.isfinite(c)]
                if not valid_history_points:
                    logger.debug(f"Skipping iteration plot for '{algo_key}' (cost history contains only invalid values).")
                    continue

                plot_data_available = True
                # Find index of the first valid point (to handle initial inf/None values)
                first_valid_idx = -1
                for i, c in enumerate(cost_history):
                    if c is not None and math.isfinite(c):
                        first_valid_idx = i
                        break
                current_effective_len = len(cost_history) - first_valid_idx if first_valid_idx != -1 else 0
                max_len_effective = max(max_len_effective, current_effective_len)

            else: # No cost history, check for a single final weighted_cost
                weighted_cost = None
                if isinstance(result_data, dict):
                    weighted_cost = result_data.get('weighted_cost')
                elif hasattr(result_data, 'weighted_cost'):
                    weighted_cost = getattr(result_data, 'weighted_cost', None)

                if weighted_cost is not None and math.isfinite(weighted_cost):
                    plot_data_available = True
                    current_effective_len = 1 # Single point counts as length 1
                    max_len_effective = max(max_len_effective, current_effective_len)
                    logger.debug(f"Using final weighted_cost for '{algo_key}' as single-point history.")
                else:
                    logger.debug(f"Skipping iteration plot for '{algo_key}' (no valid 'cost_history' or 'weighted_cost').")
                    continue

        if not plot_data_available:
            logger.warning("No valid iteration data found for any algorithm.")
            self._plot_no_data_message(ax, "No valid iteration data available.")
            return

        max_len_effective = max(1, max_len_effective) # Ensure at least 1 to avoid division by zero

        # --- Second Pass: Plot the data ---
        can_use_log_scale = True
        min_positive_cost = float('inf')

        for algo_key, result_summary in results_by_algorithm.items():
            result_data = result_summary.get('result_data')
            if not result_data or result_summary.get('run_error'):
                continue

            cost_history = None
            final_weighted_cost = None
            if isinstance(result_data, dict):
                cost_history = result_data.get('cost_history')
                final_weighted_cost = result_data.get('weighted_cost')
            elif hasattr(result_data, 'cost_history'):
                 cost_history = getattr(result_data, 'cost_history', None)
                 final_weighted_cost = getattr(result_data, 'weighted_cost', None)
            elif hasattr(result_data, 'weighted_cost'): # Fallback if only weighted_cost exists
                 final_weighted_cost = getattr(result_data, 'weighted_cost', None)

            valid_plot_points = []
            original_point_indices = []

            if cost_history and isinstance(cost_history, list):
                first_valid_idx = -1
                temp_points = []
                temp_indices = []
                for i, c in enumerate(cost_history):
                     if c is not None and math.isfinite(c):
                         if first_valid_idx == -1: first_valid_idx = i
                         temp_points.append(c)
                         temp_indices.append(i) # Store original index for potential x-axis scaling

                if temp_points:
                     valid_plot_points = temp_points
                     original_point_indices = temp_indices
                else: continue # Skip if no valid points after filtering

            elif final_weighted_cost is not None and math.isfinite(final_weighted_cost):
                 valid_plot_points = [final_weighted_cost]
                 original_point_indices = [0] # Single point at effective index 0
            else:
                 continue # Skip this algorithm if no valid data

            color, marker, algo_name = self._get_algo_style(algo_key)
            num_points = len(valid_plot_points)

            if num_points > 1:
                # Scale x-axis based on the *effective length* of this algorithm's history relative to the max effective length
                current_effective_len = len(valid_plot_points)
                # Use numpy linspace to create evenly spaced points for plotting, scaled to 100%
                # This represents the percentage of the algorithm's progress
                x_values = np.linspace(0, 100 * (current_effective_len / max_len_effective), num_points)
                label = f"{algo_name}"
                linestyle = '-'
            elif num_points == 1:
                x_values = np.array([0]) # Plot single point at the start (0%)
                label = f"{algo_name} (Final Value)"
                linestyle = 'None' # No line for single point
            else:
                 continue # Should not happen if we filter empty lists, but for safety

            # Check for non-positive values which prevent log scale
            if any(c <= FLOAT_TOLERANCE_PLOT for c in valid_plot_points):
                can_use_log_scale = False
            elif valid_plot_points: # Check if list is not empty before min()
                min_positive_cost = min(min_positive_cost, min(valid_plot_points))

            line, = ax.plot(x_values, valid_plot_points, marker=marker, linestyle=linestyle,
                            color=color, label=label, markersize=4, alpha=0.8)
            plotted_lines.append(line)

        # --- Final Plot Adjustments ---
        if can_use_log_scale and min_positive_cost != float('inf') and min_positive_cost > FLOAT_TOLERANCE_PLOT:
            try:
                ax.set_yscale('log')
                ax.set_ylabel('Best Weighted Cost Found (Log Scale)')
                # Optionally adjust grid for log scale
                ax.grid(True, which='minor', axis='y', linestyle=':', linewidth=0.4)
            except ValueError as e:
                 logger.warning(f"Could not set log scale despite positive values. Using linear scale. Error: {e}")
                 ax.set_yscale('linear')
                 ax.set_ylabel('Best Weighted Cost Found (Linear Scale)')
        else:
             ax.set_yscale('linear')
             ax.set_ylabel('Best Weighted Cost Found (Linear Scale)')

        # Dynamic Y limits for linear scale
        if ax.get_yscale() == 'linear':
             all_y_data = np.concatenate([line.get_ydata() for line in plotted_lines if line.get_ydata().size > 0]) if plotted_lines else np.array([])
             if all_y_data.size > 0:
                 min_val, max_val = np.min(all_y_data), np.max(all_y_data)
                 # Add padding, ensuring bottom limit is not below 0 if all costs are positive
                 y_range = max_val - min_val if max_val > min_val else max(1, abs(max_val * 0.2)) # Avoid zero range
                 padding = y_range * 0.05
                 bottom_lim = max(0, min_val - padding) if min_val >= 0 else min_val - padding
                 top_lim = max_val + padding
                 # Prevent excessive range if min/max are very close
                 if abs(top_lim - bottom_lim) < FLOAT_TOLERANCE_PLOT * 10: top_lim = bottom_lim + 1
                 ax.set_ylim(bottom=bottom_lim, top=top_lim)
             else:
                 ax.set_ylim(bottom=0, top=1) # Default limits if no data plotted

        if plotted_lines:
            # Create legend outside the plot area if many lines, otherwise 'best'
            num_lines = len(plotted_lines)
            legend_loc = 'upper right' # Default
            if num_lines > 4: # Adjust threshold as needed
                legend_loc = 'center left'
                bbox_anchor = (1.02, 0.5) # Position to the right of the axes
                ax.legend(handles=plotted_lines, fontsize='small', loc=legend_loc, bbox_to_anchor=bbox_anchor)
            else:
                 ax.legend(handles=plotted_lines, fontsize='small', loc='best')
        else:
            # This case should be handled by plot_data_available check earlier, but add safety message
            self._plot_no_data_message(ax, "No valid iteration data available to plot.")

        try:
             # Use constrained layout for better automatic spacing
             if ax.figure.get_layout_engine().name != 'constrained':
                 ax.figure.set_layout_engine('constrained')
             else: # Trigger re-layout if already constrained
                 ax.figure.execute_constrained_layout()
             ax.figure.canvas.draw_idle()
             logger.info("Iteration curve plot generated successfully.")
        except AttributeError:
            logger.debug("Iteration plot generated (no canvas to draw).")
        except Exception as e:
             logger.error(f"Error during final iteration plot adjustments or drawing: {e}", exc_info=True)


    # ========================================================================
    # plot_comparison_bars - Plots final Weighted Cost and Makespan comparison
    # *** CORRECTED to use 'evaluated_time' for Makespan ***
    # ========================================================================
    def plot_comparison_bars(self,
                             results_by_algorithm: Dict[str, Dict[str, Any]],
                             ax_cost: plt.Axes,
                             ax_time: plt.Axes):
        """
        Plots bar charts comparing final weighted cost and Total Delivery Time (Makespan)
        for algorithms on two separate Axes objects.

        Highlights feasibility status on both plots using color intensity and hatching.

        Args:
            results_by_algorithm: The results dictionary from route_optimizer.
                                  Expected structure: {'algo_key': {'result_data': {...}, 'run_error': str|None, ...}, ...}
                                  The 'result_data' object/dict should contain:
                                  - 'weighted_cost' (float | None)
                                  - 'evaluated_time' (float | None) <-- **CORRECTED KEY** for Makespan
                                  - 'is_feasible' (bool | None)
            ax_cost: The matplotlib.axes.Axes object for the weighted cost comparison plot.
            ax_time: The matplotlib.axes.Axes object for the Makespan comparison plot.
        """
        if not ax_cost or not ax_time:
            logger.error("Plotting Error: Axes for comparison plots not provided.")
            return

        logger.info("Generating comparison bar charts (Weighted Cost vs. Makespan)...")
        ax_cost.clear()
        ax_time.clear()

        plot_data = []
        # --- Extract and Validate Data for Plotting ---
        for algo_key, result_summary in results_by_algorithm.items():
            result_data = result_summary.get('result_data')
            run_error = result_summary.get('run_error')

            if run_error:
                 logger.warning(f"Skipping comparison plot for '{algo_key}' due to run error: {run_error}")
                 continue
            if not result_data:
                 logger.warning(f"Skipping comparison plot for '{algo_key}' (no result_data found).")
                 continue

            # --- Robust Extraction of Metrics ---
            w_cost: Optional[float] = None
            makespan: Optional[float] = None # Initialize makespan as None
            is_feasible: Optional[bool] = None

            if isinstance(result_data, dict): # If result_data is a dict
                 w_cost = result_data.get('weighted_cost')
                 makespan = result_data.get('evaluated_time') # *** CORRECTED KEY ***
                 is_feasible = result_data.get('is_feasible')
            # Check if it's an object (duck typing using expected attributes)
            elif hasattr(result_data, 'weighted_cost') and hasattr(result_data, 'evaluated_time'):
                 w_cost = getattr(result_data, 'weighted_cost', None)
                 makespan = getattr(result_data, 'evaluated_time', None) # *** CORRECTED ATTRIBUTE NAME ***
                 is_feasible = getattr(result_data, 'is_feasible', None)
            else:
                logger.warning(f"Cannot extract data from result_data for '{algo_key}' (unknown format). Skipping.")
                continue

            # Validate extracted metrics
            valid_cost = w_cost is not None and math.isfinite(w_cost)
            valid_makespan = makespan is not None and math.isfinite(makespan)
            feasible_flag = is_feasible if isinstance(is_feasible, bool) else False # Default to False if None/invalid

            # Handle invalid/missing values for plotting (use placeholders)
            # Plot invalid costs as Inf, invalid makespans as None (handled later)
            plot_cost = float('inf') if not valid_cost else w_cost
            plot_makespan = makespan if valid_makespan else None

            if not valid_cost: logger.warning(f"Invalid weighted_cost for '{algo_key}'. Will plot as INF.")
            if not valid_makespan: logger.warning(f"Invalid or missing 'evaluated_time' (Makespan) for '{algo_key}'. Will plot as N/A.")

            plot_data.append({
                'key': algo_key,
                'cost': plot_cost,
                'makespan': plot_makespan, # Store None if invalid
                'feasible': feasible_flag
            })

        if not plot_data:
            logger.warning("No valid result data found for comparison plots.")
            self._plot_no_data_message(ax_cost, "No results to compare (Cost).")
            self._plot_no_data_message(ax_time, "No results to compare (Makespan).")
            return

        # --- Prepare Data for Bar Charts ---
        # Sort data by algorithm display name for consistent order
        plot_data.sort(key=lambda x: self._get_algo_style(x['key'])[2])
        sorted_keys = [item['key'] for item in plot_data]
        algo_names = [self._get_algo_style(key)[2] for key in sorted_keys]
        costs = [item['cost'] for item in plot_data]
        makespans = [item['makespan'] for item in plot_data] # List may contain None
        feasibility_flags = [item['feasible'] for item in plot_data]
        colors = [self._get_algo_style(key)[0] for key in sorted_keys]

        x_pos = np.arange(len(algo_names))
        bar_width = 0.6

        # Determine max valid values for axis scaling, ignoring Inf/None
        finite_costs = [c for c in costs if math.isfinite(c)]
        max_cost_plot = max(finite_costs, default=0)

        finite_makespans = [m for m in makespans if m is not None and math.isfinite(m)]
        max_makespan_plot = max(finite_makespans, default=0)

        # Prepare values for plotting bars: replace Inf/None with a value slightly above the max valid value
        plot_cost_values = []
        cost_inf_flags = []
        for c in costs:
             if math.isfinite(c):
                 plot_cost_values.append(c)
                 cost_inf_flags.append(False)
             else:
                 # Plot Inf bar slightly above the max finite value for visibility
                 plot_cost_values.append(max_cost_plot * 1.1 if max_cost_plot > 0 else 1)
                 cost_inf_flags.append(True)

        plot_makespan_values = []
        makespan_na_flags = []
        for m in makespans:
            if m is not None and math.isfinite(m):
                plot_makespan_values.append(m)
                makespan_na_flags.append(False)
            else:
                # Plot NA bar slightly above the max finite value
                plot_makespan_values.append(max_makespan_plot * 1.1 if max_makespan_plot > 0 else 1)
                makespan_na_flags.append(True)


        # --- Plot Weighted Costs ---
        ax_cost.set_title('Algorithm Final Solution Comparison', pad=20) # Add padding
        ax_cost.set_ylabel('Weighted Cost')
        ax_cost.grid(True, axis='y', linestyle='--', linewidth=0.5, zorder=0)

        bars_cost = ax_cost.bar(x_pos, plot_cost_values, width=bar_width, color=colors, alpha=1.0, zorder=3)

        # Add annotations and feasibility styling for cost bars
        for i, bar in enumerate(bars_cost):
             is_inf = cost_inf_flags[i]
             is_feasible = feasibility_flags[i]

             # Style bar based on feasibility
             bar_alpha = 0.8 if is_feasible else 0.45 # More distinct alpha
             bar_hatch = '' if is_feasible else '///' # Use denser hatch
             bar.set_alpha(bar_alpha)
             if bar_hatch: bar.set_hatch(bar_hatch); bar.set_edgecolor('grey') # Add edge color for hatch

             # Annotation text
             cost_text = 'INF' if is_inf else f'{costs[i]:.2f}' # Use original cost for label
             feasibility_text = 'Feasible' if is_feasible else 'Infeasible'
             label_text = f"{cost_text}\n({feasibility_text})"

             # Annotation position
             plot_height = bar.get_height()
             current_max_y = max(plot_cost_values) if plot_cost_values else 1
             y_offset = current_max_y * 0.02 if current_max_y > 0 else 0.1
             text_y_pos = plot_height + y_offset

             # Add background box for Inf or Infeasible labels for better visibility
             text_bbox = dict(boxstyle='round,pad=0.2', fc='wheat', alpha=0.7, ec='none') if (is_inf or not is_feasible) else None
             ax_cost.text(bar.get_x() + bar.get_width() / 2., text_y_pos, label_text,
                          ha='center', va='bottom', fontsize='x-small', zorder=5, bbox=text_bbox)

        # Configure x-axis for cost plot
        ax_cost.set_xticks(x_pos)
        ax_cost.set_xticklabels(algo_names, rotation=30, ha='right', fontsize='small')
        ax_cost.tick_params(axis='x', which='major', length=0) # Hide ticks

        # Set y-limit dynamically for cost plot
        max_plot_y_cost = max(plot_cost_values) if plot_cost_values else 1
        if max_plot_y_cost > 0:
            # Ensure padding accounts for annotations
            cost_padding = max_plot_y_cost * 0.18 # Increased padding
            ax_cost.set_ylim(bottom=0, top=max_plot_y_cost + cost_padding)
        else:
            ax_cost.set_ylim(bottom=0, top=1)


        # --- Plot Total Delivery Time (Makespan) ---
        ax_time.set_title('') # Keep title minimal as main title is above cost plot
        ax_time.set_ylabel('Total Delivery Time (Makespan, hours)') # More descriptive label
        ax_time.set_xlabel('Algorithm')
        ax_time.grid(True, axis='y', linestyle='--', linewidth=0.5, zorder=0)

        # Use plot_makespan_values calculated earlier
        bars_time = ax_time.bar(x_pos, plot_makespan_values, width=bar_width, color=colors, alpha=1.0, zorder=3)

        # Add annotations and feasibility styling for Makespan bars
        for i, bar in enumerate(bars_time):
            is_na = makespan_na_flags[i]
            is_feasible = feasibility_flags[i]

            # Apply styling based on feasibility
            bar_alpha = 0.8 if is_feasible else 0.45
            bar_hatch = '' if is_feasible else '///'
            bar.set_alpha(bar_alpha)
            if bar_hatch: bar.set_hatch(bar_hatch); bar.set_edgecolor('grey')

            # Annotation text using Makespan
            makespan_text = 'N/A' if is_na else f'{makespans[i]:.2f}' # Use original makespan for label
            feasibility_text = 'Feasible' if is_feasible else 'Infeasible'
            label_text = f"{makespan_text}\n({feasibility_text})"

            # Position annotation
            plot_height = bar.get_height()
            current_max_y = max(plot_makespan_values) if plot_makespan_values else 1
            y_offset = current_max_y * 0.02 if current_max_y > 0 else 0.1
            text_y_pos = plot_height + y_offset

            # Add background box for NA or Infeasible labels
            text_bbox = dict(boxstyle='round,pad=0.2', fc='wheat', alpha=0.7, ec='none') if (is_na or not is_feasible) else None
            ax_time.text(bar.get_x() + bar.get_width() / 2., text_y_pos, label_text,
                         ha='center', va='bottom', fontsize='x-small', zorder=5, bbox=text_bbox)


        # Configure x-axis for time plot
        ax_time.set_xticks(x_pos)
        ax_time.set_xticklabels(algo_names, rotation=30, ha='right', fontsize='small')
        ax_time.tick_params(axis='x', which='major', length=0) # Hide ticks

        # Set y-limit dynamically for time plot
        max_plot_y_time = max(plot_makespan_values) if plot_makespan_values else 1
        if max_plot_y_time > 0:
            # Ensure padding accounts for annotations
            time_padding = max_plot_y_time * 0.18 # Increased padding
            ax_time.set_ylim(bottom=0, top=max_plot_y_time + time_padding)
        else:
            ax_time.set_ylim(bottom=0, top=1)


        # --- Final Adjustments and Redraw ---
        try:
            # Use constrained layout for better spacing between subplots and titles
            fig = ax_cost.figure
            if fig.get_layout_engine().name != 'constrained':
                 fig.set_layout_engine('constrained')
            else: # Trigger re-layout if already constrained
                 fig.execute_constrained_layout()

            # Explicitly draw the canvas
            fig.canvas.draw_idle()
            logger.info("Comparison plots (Weighted Cost vs. Makespan) generated successfully.")
        except AttributeError:
            logger.debug("Comparison plots generated (no canvas to draw).")
        except Exception as e:
             logger.error(f"Error during final comparison plot adjustments or drawing: {e}", exc_info=True)


# --- Standalone Testing Block (Optional) ---
# Updated to reflect the correction and test robustly
if __name__ == '__main__':
    """
    Example usage for testing the PlotGenerator class independently.
    Tests the corrected Makespan comparison plot logic.
    """
    logger.info("Running plot_generator.py in standalone test mode (Corrected Makespan Test).")

    # --- Create Dummy Results Data ---
    # Structure mimicking output from route_optimizer, ensuring 'evaluated_time' is used

    class DummyResultDataObject: # Simulates object case
        def __init__(self, cost, time, feasible, history=None, weighted=None):
            self.total_cost = cost # Raw cost (can be None)
            self.evaluated_time = time # Makespan (can be None) <-- Use correct name
            self.is_feasible = feasible
            self.weighted_cost = weighted if weighted is not None else (cost + time if cost is not None and time is not None else float('inf'))
            self.cost_history = history if history else ([self.weighted_cost] * 5 if self.weighted_cost != float('inf') else [2000, 1800, 1600, 1500, 1400]) # Example history

    def create_dummy_result_dict(cost, time, feasible, history=None, weighted=None): # Simulates dict case
        w = weighted if weighted is not None else (cost + time if cost is not None and time is not None else float('inf'))
        h = history if history else ([w] * 5 if w != float('inf') else [3000, 2800, 2500, 2300, 2200]) # Example history
        return {
            'total_cost': cost,
            'evaluated_time': time, # Makespan <-- Use correct name
            'is_feasible': feasible,
            'weighted_cost': w,
            'cost_history': h
        }


    dummy_results_corrected = {
        'genetic_algorithm': { # Using Object
            'result_data': DummyResultDataObject(cost=1450.7, time=7.5, feasible=True, weighted=1500, history=[2200, 2000, 1800, 1600, 1550, 1500]),
            'run_error': None
        },
        'simulated_annealing': { # Using Dict
             'result_data': create_dummy_result_dict(cost=1850.0, time=8.8, feasible=False, weighted=1900, history=[2700, 2600, 2400, 2200, 2000, 1950, 1900]),
             'run_error': None
        },
        'pso_optimizer': { # Using Object
             'result_data': DummyResultDataObject(cost=1600.8, time=7.2, feasible=True, weighted=1650, history=[float('inf'), 2300, 2000, 1800, 1700, 1650, 1620]),
             'run_error': None
        },
        'greedy_heuristic': { # Using Dict
             'result_data': create_dummy_result_dict(cost=2500.0, time=6.8, feasible=True, weighted=3180, history=[3180]), # Only final cost as history
             'run_error': None
        },
         'makespan_missing_algo': { # Using Object with None time
             'result_data': DummyResultDataObject(cost=2200.0, time=None, feasible=True, weighted=2200, history=[2200]),
             'run_error': None
         },
        'failed_algorithm': { # Failed run
             'result_data': None,
             'run_error': 'Timeout during execution'
        }
    }

    # --- Create Plots ---
    plot_gen = PlotGenerator()
    output_dir = "output_test/charts_corrected_makespan"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Iteration Curve Plot (Should work as before)
    fig1, ax1 = plt.subplots(figsize=(8, 5))
    plot_gen.plot_iteration_curves(dummy_results_corrected, ax1)
    # Set suptitle after plotting might require redraw/layout adjustment
    fig1.suptitle("Test Iteration Curve Plot (Corrected Data)")
    if fig1.get_layout_engine().name == 'constrained': fig1.execute_constrained_layout()
    try:
        iter_filename = os.path.join(output_dir, "test_iteration_curves_corrected.png")
        fig1.savefig(iter_filename, dpi=150, bbox_inches='tight')
        logger.info(f"Test iteration plot saved to {iter_filename}")
    except Exception as e:
        logger.error(f"Failed to save test iteration plot: {e}", exc_info=True)
    plt.close(fig1)


    # 2. Comparison Plot (Weighted Cost and Makespan) - Should now work correctly
    fig2, (ax_cost_test, ax_time_test) = plt.subplots(2, 1, figsize=(8, 9), sharex=False)
    plot_gen.plot_comparison_bars(dummy_results_corrected, ax_cost_test, ax_time_test)
    # Set suptitle after plotting might require redraw/layout adjustment
    fig2.suptitle("Test Comparison Plot: Weighted Cost vs. Makespan (Corrected Access)")
    if fig2.get_layout_engine().name == 'constrained': fig2.execute_constrained_layout()
    try:
        comp_filename = os.path.join(output_dir, "test_comparison_bars_cost_makespan_corrected.png")
        fig2.savefig(comp_filename, dpi=150, bbox_inches='tight')
        logger.info(f"Test comparison plot saved to {comp_filename}")
    except Exception as e:
        logger.error(f"Failed to save test comparison plot: {e}", exc_info=True)
    plt.close(fig2)

    logger.info("Standalone plot generation test (Corrected Makespan) finished.")