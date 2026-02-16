"""
Interactive high-dimensional visualization using langevitour.
Provides an animated touring view of the optimization search space.
"""

import numpy as np

def plot_langevitour(x, f, objectives, feasible, bounds=None, output_file="optimization_tour.html"):
    """
    Create an interactive langevitour visualization of the optimization process.

    Parameters:
    -----------
    x : np.ndarray
        Design variables (N, dims) - the actual search points
    f : np.ndarray
        Feasibility values (N, 1) - binary feasibility
    objectives : list of tuples
        List of objective values for each point, e.g., [(f1,), (f2,), ...]
    feasible : list of bool
        Feasibility status for each point
    bounds : list, optional
        Bounds for each dimension [lower1, upper1, lower2, upper2, ...]
    output_file : str
        Output HTML filename

    Returns:
    --------
    tour : Langevitour object
        The tour visualization object
    """
    try:
        from langevitour import Langevitour
    except ImportError:
        print("\n" + "="*60)
        print("ERROR: langevitour is not installed!")
        print("Install it with: pip install langevitour")
        print("="*60 + "\n")
        return None

    # Convert data to proper format
    X = x.copy()
    n_points = X.shape[0]
    dims = X.shape[1]

    # Create groups based on feasibility and objective value quartiles
    # Group 0: infeasible (red)
    # Groups 1-4: feasible, colored by objective value quartile (green to yellow)
    f1_values = np.array([obj[0] for obj in objectives])

    # Convert feasible to boolean numpy array
    feasible_mask = np.array([bool(f) for f in feasible])

    # Get objective values for feasible points only
    feasible_f1 = f1_values[feasible_mask]

    # Compute quartile thresholds once (if we have feasible points)
    if len(feasible_f1) > 0:
        quartile_thresholds = np.percentile(feasible_f1, [25, 50, 75])
    else:
        quartile_thresholds = None

    # Define group names
    group_names = [
        "Infeasible",
        "Best 25% (feasible)",
        "25-50% (feasible)",
        "50-75% (feasible)",
        "Worst 25% (feasible)"
    ]

    groups = []

    for i in range(n_points):
        if not feasible[i]:
            groups.append(group_names[0])  # Infeasible
        else:
            # Divide feasible points into quartiles based on objective value
            if quartile_thresholds is not None:
                quartile = np.searchsorted(quartile_thresholds, f1_values[i])
                groups.append(group_names[quartile + 1])  # Groups 1-4
            else:
                groups.append(group_names[1])

    # Create axis labels
    if bounds is not None:
        axis_labels = []
        for i in range(dims):
            lower = bounds[2*i]
            upper = bounds[2*i+1]
            axis_labels.append(f"x{i+1} [{lower:.2f}, {upper:.2f}]")
    else:
        axis_labels = [f"x{i+1}" for i in range(dims)]

    # Create the tour
    print("\nCreating langevitour visualization...")
    print(f"  Points: {n_points}")
    print(f"  Dimensions: {dims}")
    print(f"  Groups: {len(set(groups))}")
    print(f"  Feasible points: {sum(feasible)}")

    tour = Langevitour(
        data=X,
        group=groups,
        column_names=axis_labels,
        point_size=2.5
    )

    # Save to HTML
    tour.write_html(output_file)
    print(f"\n✓ Interactive visualization saved to: {output_file}")
    print(f"  Open this file in a web browser to explore your optimization space!")
    print(f"\n  Features:")
    print(f"    - Drag to rotate the projection")
    print(f"    - Use GUI to show/hide groups")
    print(f"    - Enable 'Guide' mode for projection pursuit")
    print(f"    - Select specific axes to focus on")

    return tour


def plot_langevitour_from_lazy_opt(lazy_opt, output_file="optimization_tour.html"):
    """
    Convenience function to create langevitour directly from a LazyOpt instance.

    Parameters:
    -----------
    lazy_opt : LazyOpt
        The LazyOpt instance after optimization
    output_file : str
        Output HTML filename

    Returns:
    --------
    tour : Langevitour object
    """
    return plot_langevitour(
        x=lazy_opt.x,
        f=lazy_opt.f,
        objectives=lazy_opt.objectives,
        feasible=lazy_opt.feasible,
        bounds=lazy_opt.bounds,
        output_file=output_file
    )


if __name__ == "__main__":
    print("langevitour plotting utilities for LazyOpt")
    print("\nUsage:")
    print("  from lazy_plot_langevitour import plot_langevitour_from_lazy_opt")
    print("  plot_langevitour_from_lazy_opt(lazy_opt, 'my_optimization.html')")
