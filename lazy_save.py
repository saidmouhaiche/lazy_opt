"""
lazy_save.py - CSV save/load utilities for lazy_opt

Supports saving and loading optimization results with:
- Variable number of input dimensions
- Variable number of objectives
- Feasibility status
- Incremental append mode for live updates
"""

import pandas as pd
import numpy as np
import pickle
import warnings
from pathlib import Path


def save_to_csv(x, feasible, objectives, filepath, mode='w', include_header=True):
    """
    Save optimization results to CSV format.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_dims)
        Input design points
    feasible : list of bool, length n_samples
        Feasibility status for each point
    objectives : list of tuples, length n_samples
        Objective values for each point. Each tuple can have variable length.
        Example: [(10.5,), (15.2,), (20.1, 5.3)]
    filepath : str or Path
        Output CSV file path
    mode : str, default='w'
        Write mode: 'w' for overwrite, 'a' for append
    include_header : bool, default=True
        Whether to write column headers (set False when appending)

    Returns
    -------
    None

    Examples
    --------
    # Single objective
    x = np.array([[0.1, 0.2], [0.3, 0.4]])
    feasible = [True, False]
    objectives = [(10.5,), (15.2,)]
    save_to_csv(x, feasible, objectives, 'results.csv')

    # Multiple objectives
    objectives = [(10.5, 20.0), (15.2, 18.5)]
    save_to_csv(x, feasible, objectives, 'results.csv')

    # Append mode (for live updates)
    save_to_csv(x_new, feasible_new, objectives_new, 'results.csv',
                mode='a', include_header=False)
    """

    n_samples, n_dims = x.shape

    # Determine max number of objectives
    max_n_objectives = max(len(obj) for obj in objectives)

    # Build column names
    col_names = [f'x{i}' for i in range(n_dims)]
    col_names.append('feasible')
    col_names.extend([f'f{i}' for i in range(max_n_objectives)])

    # Build data rows
    rows = []
    for i in range(n_samples):
        row = list(x[i, :])  # input dimensions
        row.append(int(feasible[i]))  # feasible as 0/1

        # Add objectives, pad with NaN if fewer objectives
        obj_vals = list(objectives[i])
        obj_vals.extend([np.nan] * (max_n_objectives - len(obj_vals)))
        row.extend(obj_vals)

        rows.append(row)

    # Create DataFrame
    df = pd.DataFrame(rows, columns=col_names)

    # Save to CSV
    df.to_csv(filepath, mode=mode, header=include_header, index=False)

    if mode == 'w':
        print(f"Saved {n_samples} samples to {filepath}")
    else:
        print(f"Appended {n_samples} samples to {filepath}")


def load_from_csv(filepath):
    """
    Load optimization results from CSV format.

    Parameters
    ----------
    filepath : str or Path
        Input CSV file path

    Returns
    -------
    x : np.ndarray, shape (n_samples, n_dims)
        Input design points
    feasible : list of bool
        Feasibility status for each point
    objectives : list of tuples
        Objective values for each point

    Examples
    --------
    x, feasible, objectives = load_from_csv('results.csv')

    # Can be used to seed a new optimization
    lazy = LazyOpt(
        solver_function=my_func,
        bounds=[(0, 1), (0, 1)],
        hyper_params={...},
        seed={'x': x, 'objectives': objectives, 'feasible': feasible}
    )
    """

    df = pd.read_csv(filepath)

    # Identify column types
    x_cols = [col for col in df.columns if col.startswith('x')]
    f_cols = [col for col in df.columns if col.startswith('f')]

    n_dims = len(x_cols)
    n_samples = len(df)

    # Extract x values
    x = df[x_cols].values

    # Extract feasible
    feasible = df['feasible'].astype(bool).tolist()

    # Extract objectives (handle variable-length tuples)
    objectives = []
    for idx in range(n_samples):
        obj_vals = df.loc[idx, f_cols].values
        # Remove NaN values and convert to tuple
        obj_vals_clean = tuple(obj_vals[~np.isnan(obj_vals)])
        objectives.append(obj_vals_clean)

    print(f"Loaded {n_samples} samples from {filepath}")
    print(f"  Dimensions: {n_dims}")
    print(f"  Objectives: {len(f_cols)}")
    print(f"  Feasible: {sum(feasible)}/{n_samples}")

    return x, feasible, objectives


def save_incremental(lazy_opt, filepath, iteration):
    """
    Save the latest point from a LazyOpt instance incrementally.
    Useful for live updates during optimization.

    Parameters
    ----------
    lazy_opt : LazyOpt instance
        The optimization object
    filepath : str or Path
        Output CSV file path
    iteration : int
        Current iteration number (used to determine if this is first save)

    Examples
    --------
    # Inside optimization loop
    for i in range(max_iterations):
        # ... optimization step ...
        save_incremental(lazy_opt, 'live_results.csv', iteration=i)
    """

    # Get the latest point
    x_latest = lazy_opt.x[-1:, :]  # last row
    feasible_latest = [lazy_opt.feasible[-1]]
    objectives_latest = [lazy_opt.objectives[-1]]

    # First iteration: write with header
    # Subsequent: append without header
    if iteration == 0:
        mode = 'w'
        include_header = True
    else:
        mode = 'a'
        include_header = False

    save_to_csv(x_latest, feasible_latest, objectives_latest,
                filepath, mode=mode, include_header=include_header)


def save_full(lazy_opt, filepath):
    """
    Save all results from a LazyOpt instance.

    Parameters
    ----------
    lazy_opt : LazyOpt instance
        The optimization object
    filepath : str or Path
        Output CSV file path

    Examples
    --------
    # After optimization completes
    save_full(lazy_opt, 'final_results.csv')
    """

    save_to_csv(lazy_opt.x, lazy_opt.feasible, lazy_opt.objectives, filepath)


# Example usage
if __name__ == "__main__":
    # Example: creating sample data
    print("=== Example: Save and Load ===\n")

    # Single objective case
    x = np.array([[0.1, 0.2, 0.3],
                  [0.4, 0.5, 0.6],
                  [0.7, 0.8, 0.9]])
    feasible = [True, False, True]
    objectives = [(10.5,), (15.2,), (8.3,)]

    save_to_csv(x, feasible, objectives, 'test_results.csv')

    # Load it back
    x_loaded, feasible_loaded, objectives_loaded = load_from_csv('test_results.csv')

    print("\nOriginal:")
    print(f"x:\n{x}")
    print(f"feasible: {feasible}")
    print(f"objectives: {objectives}")

    print("\nLoaded:")
    print(f"x:\n{x_loaded}")
    print(f"feasible: {feasible_loaded}")
    print(f"objectives: {objectives_loaded}")

    # Multi-objective case
    print("\n=== Example: Multi-objective ===\n")
    objectives_multi = [(10.5, 20.0, 5.0), (15.2, 18.5, 6.2), (8.3, 22.1, 4.8)]
    save_to_csv(x, feasible, objectives_multi, 'test_results_multi.csv')

    x_m, f_m, obj_m = load_from_csv('test_results_multi.csv')
    print(f"Multi-objective loaded: {obj_m}")

    # Append mode example
    print("\n=== Example: Append mode ===\n")
    x_new = np.array([[0.2, 0.3, 0.4]])
    feasible_new = [True]
    objectives_new = [(12.0,)]

    save_to_csv(x_new, feasible_new, objectives_new, 'test_results.csv',
                mode='a', include_header=False)

    x_appended, f_appended, obj_appended = load_from_csv('test_results.csv')
    print(f"After append: {len(x_appended)} samples")
