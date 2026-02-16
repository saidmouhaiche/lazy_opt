"""
Example: Using langevitour with LazyOpt for interactive high-dimensional visualization.

This creates an interactive HTML visualization that lets you explore your optimization
search space with smooth animated projections.
"""

import numpy as np
from lazy_opt import LazyOpt
from lazy_plot_langevitour import plot_langevitour_from_lazy_opt


def example_function(x):
    """
    Example objective function - modify this to match your problem.
    Returns: (feasible, (objective_values,))
    """
    # Simple sphere function with constraints
    x = x.flatten()

    # Objective: minimize sum of squares
    f1 = np.sum(x**2)

    # Constraint: x values should be > 0.2
    feasible = np.all(x > 0.2)

    return feasible, (f1,)


if __name__ == '__main__':

    print("="*60)
    print("LazyOpt with Langevitour Visualization Example")
    print("="*60)

    # Define problem dimensions
    dims = 5

    # Define bounds: [lower1, upper1, lower2, upper2, ...]
    bounds = []
    for i in range(dims):
        bounds.extend([0.0, 1.0])  # Each dimension: [0, 1]

    # Hyperparameters
    hyper_params = {
        'surrogate': 'KNN',
        'epsilon': 1,
        'number_of_DOE_samples': 3500,
        'number_of_iterations': 100,
        'number_of_psuedo_candidates': 10000,
    }

    # Options
    options = {
        'live_plot_draw': False,  # Disable live plotting for speed
        'live_plot_skip': 5,
        'draw_latent_plot': False,  # Disable TSNE plot
        'number_of_processes': 4,
        'assume_feasible_seeds': False,
        'use_langevitour': True  # Enable langevitour
    }

    # Optional: provide seed points
    seed = None
    # seed = np.array([
    #     [0.5, 0.5, 0.5, 0.5, 0.5],
    #     [0.3, 0.3, 0.3, 0.3, 0.3],
    # ])

    print("\nRunning optimization...")
    print(f"Dimensions: {dims}")
    print(f"Max iterations: {hyper_params['number_of_iterations']}")

    # Run optimization
    lazy = LazyOpt(
        solver_function=example_function,
        bounds=bounds,
        hyper_params=hyper_params,
        seed=seed,
        options=options
    )

    print("\n" + "="*60)
    print("Optimization complete!")
    print("="*60)

    # Print results summary
    feasible_indices = [i for i, f in enumerate(lazy.feasible) if f]
    if feasible_indices:
        best_idx = feasible_indices[0]
        best_obj = lazy.objectives[0][0]

        for idx in feasible_indices:
            if lazy.objectives[idx][0] < best_obj:
                best_idx = idx
                best_obj = lazy.objectives[idx][0]

        print(f"\nBest feasible solution:")
        print(f"  Index: {best_idx}")
        print(f"  Objective: {best_obj:.6f}")
        print(f"  Design variables: {lazy.x[best_idx]}")
        print(f"\nTotal feasible points found: {len(feasible_indices)}/{len(lazy.feasible)}")
    else:
        print("\nNo feasible solutions found!")

    # Create interactive langevitour visualization
    print("\n" + "="*60)
    print("Creating Interactive Visualization")
    print("="*60)

    plot_langevitour_from_lazy_opt(
        lazy,
        output_file="optimization_tour.html"
    )

    print("\n✓ Done! Open 'optimization_tour.html' in your browser.")
    print("\nTips for using langevitour:")
    print("  1. The plot will animate through different 2D projections")
    print("  2. Drag with mouse to rotate and explore")
    print("  3. Use checkboxes to show/hide different groups:")
    print("     - Infeasible points (typically red)")
    print("     - Feasible points colored by objective value quartiles")
    print("  4. Click 'Guide' to enable projection pursuit")
    print("  5. Click on axis names to fix specific dimensions")
