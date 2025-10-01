import os
import pickle
import numpy as np
import tensorstore as ts
from scipy.stats import zscore
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import networkx as nx
import multiprocessing as mp
from functools import partial
from numba import jit

from zapbench import constants, data_utils

# --- Configuration ---
cfg = {
    'traces_path': 'file:///home/v/proj/zebra/data/traces',
    'max_neurons': None,
    'max_lag': 2,
    'n_shuffles': 50,
    'p_value_threshold': 0.01,
    'output_file': 'combined_connectivity_graph.pkl',
    'plot_dir': 'connectivity_plots',
    'use_gpu': False,
    
    'use_windowed_analysis': True,
    'window_size_fraction': 0.2,
    'window_step_fraction': 0.1,
    'min_significant_windows': 5,
    
    'n_workers': 0,  # None = use all CPUs
}


def compute_lagged_correlations_vectorized(source_windows, target_windows, max_lag):
    """
    Compute correlations at specific lags for all windows and targets simultaneously.
    
    Args:
        source_windows: (n_windows, window_size)
        target_windows: (n_windows, window_size, n_targets)
        max_lag: maximum lag to compute
    
    Returns:
        correlations: (n_windows, n_targets, 2*max_lag+1) - correlation at each lag
    """
    n_windows, window_size = source_windows.shape
    n_targets = target_windows.shape[2]
    n_lags = 2 * max_lag + 1
    
    correlations = np.zeros((n_windows, n_targets, n_lags))
    
    # For each lag, compute correlation
    for lag_idx, lag in enumerate(range(-max_lag, max_lag + 1)):
        if lag == 0:
            # No shift needed
            correlations[:, :, lag_idx] = np.sum(
                source_windows[:, :, np.newaxis] * target_windows,
                axis=1
            )
        elif lag > 0:
            # Positive lag: source leads target
            # correlate(source, target) with positive lag means source[t] * target[t+lag]
            valid_len = window_size - lag
            correlations[:, :, lag_idx] = np.sum(
                source_windows[:, :valid_len, np.newaxis] * target_windows[:, lag:, :],
                axis=1
            )
        else:  # lag < 0
            # Negative lag: target leads source
            # source[t] * target[t+lag] = source[t] * target[t-|lag|]
            valid_len = window_size + lag  # lag is negative
            correlations[:, :, lag_idx] = np.sum(
                source_windows[:, -lag:, np.newaxis] * target_windows[:, :valid_len, :],
                axis=1
            )
    
    return correlations


def compute_significance_thresholds_vectorized(source_windows, target_windows, max_lag, n_shuffles):
    """
    Compute significance thresholds by shuffling window indices.
    Tests if temporal alignment between source and target windows matters.
    
    Args:
        source_windows: (n_windows, window_size)
        target_windows: (n_windows, window_size, n_targets)
        max_lag: maximum lag
        n_shuffles: number of shuffles
    
    Returns:
        thresholds: (n_targets,) - max correlation threshold for each target
    """
    n_windows, window_size = source_windows.shape
    n_targets = target_windows.shape[2]
    
    max_corr_shuffled = np.zeros((n_targets, n_shuffles))
    
    for shuffle_idx in range(n_shuffles):
        # Shuffle window indices to break temporal alignment
        shuffled_indices = np.random.permutation(n_windows)
        shuffled_targets = target_windows[shuffled_indices, :, :]
        
        # Compute correlations with misaligned windows
        shuffled_corr = compute_lagged_correlations_vectorized(source_windows, shuffled_targets, max_lag)
        
        # Store max absolute correlation across lags and windows
        max_corr_shuffled[:, shuffle_idx] = np.max(np.max(np.abs(shuffled_corr), axis=2), axis=0)
    
    # Compute percentile thresholds (one per target)
    thresholds = np.percentile(max_corr_shuffled, (1 - cfg['p_value_threshold']) * 100, axis=1)
    
    return thresholds


def analyze_source_neuron_windowed(source_idx, windowed_data, max_lag, n_shuffles, min_significant_windows):
    """
    Analyze one source neuron against all subsequent target neurons using windowed analysis.
    
    Args:
        source_idx: index of source neuron
        windowed_data: (n_windows, window_size, n_neurons)
        max_lag: maximum lag
        n_shuffles: number of shuffles for significance
        min_significant_windows: minimum windows that must be significant
    
    Returns:
        list of (pre_synaptic, post_synaptic) connections
    """
    n_windows, window_size, n_neurons = windowed_data.shape
    
    if source_idx >= n_neurons - 1:
        return []
    
    # Extract source windows: (n_windows, window_size)
    source_windows = windowed_data[:, :, source_idx]
    
    # Extract all target windows: (n_windows, window_size, n_targets)
    target_indices = np.arange(source_idx + 1, n_neurons)
    target_windows = windowed_data[:, :, target_indices]
    
    # Compute significance thresholds (one per target)
    thresholds = compute_significance_thresholds_vectorized(
        source_windows, target_windows, max_lag, n_shuffles
    )
    
    # Compute actual correlations
    actual_corr = compute_lagged_correlations_vectorized(
        source_windows, target_windows, max_lag
    )
    
    # Find peak correlation for each window-target pair
    peak_corr_vals = np.max(np.abs(actual_corr), axis=2)  # (n_windows, n_targets)
    
    # Determine which windows are significant for each target
    # thresholds is (n_targets,), broadcast to (n_windows, n_targets)
    significant_mask = peak_corr_vals > thresholds[np.newaxis, :]  # (n_windows, n_targets)
    
    # Count significant windows per target
    significant_window_counts = np.sum(significant_mask, axis=0)  # (n_targets,)
    
    # Find targets with enough significant windows
    valid_targets_mask = significant_window_counts >= min_significant_windows
    valid_target_indices = target_indices[valid_targets_mask]
    
    if len(valid_target_indices) == 0:
        return []
    
    # For valid targets, compute final correlation on full trace to determine direction
    # We'll use the peak lag from the window with maximum correlation
    connections = []
    
    for target_idx in valid_target_indices:
        local_target_idx = target_idx - (source_idx + 1)
        
        # Find the window with maximum absolute correlation for this target
        window_peak_corrs = peak_corr_vals[:, local_target_idx]
        best_window = np.argmax(window_peak_corrs)
        
        # Get the lag with maximum correlation in that window
        best_corr_across_lags = actual_corr[best_window, local_target_idx, :]
        best_lag_idx = np.argmax(np.abs(best_corr_across_lags))
        best_lag = best_lag_idx - max_lag
        
        # Determine direction based on lag
        if best_lag > 0:
            # Positive lag: source leads target -> source -> target
            connections.append((source_idx, target_idx))
        elif best_lag < 0:
            # Negative lag: target leads source -> target -> source
            connections.append((target_idx, source_idx))
        # If lag == 0, we skip (no clear direction)
    
    return connections


def create_windowed_data(data, window_size, window_step):
    """
    Create sliding windows from the data.
    
    Args:
        data: (n_timesteps, n_neurons)
        window_size: size of each window
        window_step: step between windows
    
    Returns:
        windowed_data: (n_windows, window_size, n_neurons)
    """
    n_timesteps, n_neurons = data.shape
    
    # Calculate number of windows
    n_windows = (n_timesteps - window_size) // window_step + 1
    
    # Pre-allocate array
    windowed_data = np.zeros((n_windows, window_size, n_neurons))
    
    for i in range(n_windows):
        start = i * window_step
        end = start + window_size
        if end > n_timesteps:
            break
        windowed_data[i] = data[start:end, :]
    
    return windowed_data[:i+1]  # Return only filled windows


def calculate_connectivity_windowed_cpu(data, max_lag, n_shuffles, min_significant_windows, window_size, window_step, n_workers):
    """
    Calculate connectivity using windowed analysis with CPU parallelization.
    """
    num_timesteps, num_neurons = data.shape
    
    print(f"Creating sliding windows (size={window_size}, step={window_step})...")
    windowed_data = create_windowed_data(data, window_size, window_step)
    n_windows = windowed_data.shape[0]
    print(f"Created {n_windows} windows")
    
    print(f"Calculating functional connectivity using {n_workers} CPU workers...")
    
    connectivity = {i: [] for i in range(num_neurons)}
    
    # Create worker function with fixed arguments
    worker_func = partial(
        analyze_source_neuron_windowed,
        windowed_data=windowed_data,
        max_lag=max_lag,
        n_shuffles=n_shuffles,
        min_significant_windows=min_significant_windows
    )

    if n_workers is None or n_workers > 0:
        with mp.Pool(n_workers) as pool:
            results = list(tqdm(
                pool.imap_unordered(worker_func, range(num_neurons)),
                total=num_neurons,
                desc="Analyzing neurons"
            ))
    else:
        results = []
        for i in tqdm(range(num_neurons), desc="Analyzing neurons"):
            results.append(worker_func(i))
    
    print("Aggregating results...")
    for connection_list in results:
        for pre_synaptic, post_synaptic in connection_list:
            connectivity[post_synaptic].append(pre_synaptic)
    
    return connectivity


def load_preprocessed_data(traces_path, condition_name, max_neurons=None):
    """Loads and preprocesses neural activity data."""
    print("Loading data...")
    ds_traces = ts.open({'driver': 'zarr3', 'kvstore': traces_path}).result()

    condition_idx = constants.CONDITION_NAMES.index(condition_name)
    trace_min, trace_max = data_utils.get_condition_bounds(condition_idx)

    traces = ds_traces[trace_min:trace_max, :].read().result()

    if max_neurons is not None:
        traces = traces[:, :max_neurons]

    print(f"Loaded traces with shape: {traces.shape}")

    print("Preprocessing data (z-scoring)...")
    traces = zscore(traces, axis=0)
    traces = np.nan_to_num(traces)

    return traces


def plot_connectivity_examples(data, connectivity, max_lag, plot_dir, n_examples=5):
    """Plots examples of connected and unconnected neuron pairs."""
    print(f"Plotting connectivity examples to '{plot_dir}'...")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    num_timesteps, num_neurons = data.shape

    # Get connected pairs
    connected_pairs_set = set()
    for post_synaptic, pre_synaptics in connectivity.items():
        for pre_synaptic in pre_synaptics:
            connected_pairs_set.add(tuple(sorted((pre_synaptic, post_synaptic))))
    
    connected_pairs = list(connected_pairs_set)
    random.shuffle(connected_pairs)

    # Find unconnected pairs
    unconnected_pairs = []
    attempts = 0
    max_attempts = n_examples * 1000
    while len(unconnected_pairs) < n_examples and attempts < max_attempts:
        n1 = random.randint(0, num_neurons - 1)
        n2 = random.randint(0, num_neurons - 1)
        if n1 == n2:
            continue
        
        pair = tuple(sorted((n1, n2)))
        if pair not in connected_pairs_set and pair not in unconnected_pairs:
            unconnected_pairs.append(pair)
        attempts += 1
    
    if len(unconnected_pairs) < n_examples:
        print(f"Warning: Could only find {len(unconnected_pairs)} unconnected pairs")

    # Plot connected pairs
    if len(connected_pairs) > 0:
        fig_connected, axes_connected = plt.subplots(min(n_examples, len(connected_pairs)), 2, 
                                                      figsize=(12, 4 * min(n_examples, len(connected_pairs))), 
                                                      constrained_layout=True)
        if min(n_examples, len(connected_pairs)) == 1:
            axes_connected = [axes_connected]
        fig_connected.suptitle('Examples of Significantly Connected Neuron Pairs', fontsize=16)
        for i, (neuron1, neuron2) in enumerate(connected_pairs[:n_examples]):
            plot_pair_analysis(data, neuron1, neuron2, max_lag, axes_connected[i])
        plt.savefig(os.path.join(plot_dir, "connected_pairs.png"))
        plt.close(fig_connected)

    # Plot unconnected pairs
    if len(unconnected_pairs) > 0:
        fig_unconnected, axes_unconnected = plt.subplots(min(n_examples, len(unconnected_pairs)), 2, 
                                                          figsize=(12, 4 * min(n_examples, len(unconnected_pairs))), 
                                                          constrained_layout=True)
        if min(n_examples, len(unconnected_pairs)) == 1:
            axes_unconnected = [axes_unconnected]
        fig_unconnected.suptitle('Examples of Unconnected Neuron Pairs', fontsize=16)
        for i, (neuron1, neuron2) in enumerate(unconnected_pairs[:n_examples]):
            plot_pair_analysis(data, neuron1, neuron2, max_lag, axes_unconnected[i])
        plt.savefig(os.path.join(plot_dir, "unconnected_pairs.png"))
        plt.close(fig_unconnected)

    print("Plotting complete.")


def plot_connectivity_graph(connectivity, plot_dir):
    """Creates and plots a directed graph of neuron connectivity."""
    print(f"Plotting connectivity network graph to '{plot_dir}'...")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    G = nx.DiGraph()
    for post_synaptic, pre_synaptics in connectivity.items():
        for pre_synaptic in pre_synaptics:
            G.add_edge(pre_synaptic, post_synaptic)

    if not G.nodes():
        print("No connections found, skipping network graph plot.")
        return

    plt.figure(figsize=(12, 12))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_size=50, font_size=8, arrows=True, width=0.5, alpha=0.7)
    plt.title("Neuron Connectivity Graph")
    plt.savefig(os.path.join(plot_dir, "connectivity_network.png"))
    plt.close()
    print("Network graph plotting complete.")


def plot_pair_analysis(data, neuron_i_idx, neuron_j_idx, max_lag, axes):
    """Helper function to plot analysis for a single pair of neurons."""
    from scipy.signal import correlate
    
    num_timesteps = data.shape[0]
    activity_i = data[:, neuron_i_idx]
    activity_j = data[:, neuron_j_idx]

    # Plot activities
    axes[0].plot(activity_i, label=f'Neuron {neuron_i_idx}', alpha=0.8)
    axes[0].plot(activity_j, label=f'Neuron {neuron_j_idx}', alpha=0.8)
    axes[0].set_title(f'Activity of Neurons {neuron_i_idx} and {neuron_j_idx}')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Z-scored Activity')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot cross-correlation
    cross_corr = correlate(activity_i, activity_j, mode='full')
    lags = np.arange(-num_timesteps + 1, num_timesteps)
    
    center_idx = len(cross_corr) // 2
    start = center_idx - max_lag
    end = center_idx + max_lag + 1
    cross_corr_window = cross_corr[start:end]
    lags_window = lags[start:end]

    axes[1].plot(lags_window, cross_corr_window)
    peak_lag_idx = np.argmax(np.abs(cross_corr_window))
    peak_lag = lags_window[peak_lag_idx]
    axes[1].axvline(peak_lag, color='r', linestyle='--', label=f'Peak Lag: {peak_lag}')
    axes[1].set_title('Cross-Correlation')
    axes[1].set_xlabel('Time Lag (steps)')
    axes[1].set_ylabel('Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)


if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning,ignore::FutureWarning'

    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass

    # Determine number of workers
    n_workers = cfg['n_workers'] if cfg['n_workers'] is not None else mp.cpu_count()
    print(f"Using {n_workers} CPU workers")

    all_conditions_connectivity = {}
    num_neurons = None

    for condition_name in constants.CONDITION_NAMES:
        print(f"\n--- Processing condition: {condition_name} ---")
        condition_plot_dir = os.path.join(cfg['plot_dir'], condition_name)

        # Load Data
        neural_data = load_preprocessed_data(
            cfg['traces_path'],
            condition_name,
            cfg['max_neurons']
        )

        if num_neurons is None:
            num_neurons = neural_data.shape[1]

        # Calculate window parameters
        num_timesteps_condition = neural_data.shape[0]
        window_size = int(num_timesteps_condition * cfg['window_size_fraction'])
        window_step = int(num_timesteps_condition * cfg['window_step_fraction'])

        # Calculate Connectivity
        connectivity_graph = calculate_connectivity_windowed_cpu(
            neural_data,
            cfg['max_lag'],
            cfg['n_shuffles'],
            cfg['min_significant_windows'],
            window_size,
            window_step,
            n_workers
        )
        
        all_conditions_connectivity[condition_name] = connectivity_graph

        # Save individual results
        condition_output_file = os.path.join(condition_plot_dir, 'connectivity_graph.pkl')
        print(f"Saving connectivity graph for '{condition_name}' to {condition_output_file}...")
        if not os.path.exists(condition_plot_dir):
            os.makedirs(condition_plot_dir)
        with open(condition_output_file, 'wb') as f:
            pickle.dump(connectivity_graph, f)

        print(f"Found {sum(len(v) for v in connectivity_graph.values())} significant connections in '{condition_name}'.")

        # Plot examples
        plot_connectivity_examples(neural_data, connectivity_graph, cfg['max_lag'], condition_plot_dir)

        # Plot network graph
        plot_connectivity_graph(connectivity_graph, condition_plot_dir)

    # Merge and Save Combined Results
    print("\n--- Merging results from all conditions ---")
    combined_connectivity = {i: [] for i in range(num_neurons)}
    for condition_name, connectivity in all_conditions_connectivity.items():
        for post_synaptic, pre_synaptics in connectivity.items():
            combined_connectivity[post_synaptic].extend(pre_synaptics)

    # Remove duplicates
    for post_synaptic in combined_connectivity:
        combined_connectivity[post_synaptic] = list(set(combined_connectivity[post_synaptic]))

    # Save combined dictionary
    print(f"Saving combined connectivity graph to {cfg['output_file']}...")
    with open(cfg['output_file'], 'wb') as f:
        pickle.dump(combined_connectivity, f)

    # Plot combined network graph
    plot_connectivity_graph(combined_connectivity, cfg['plot_dir'])

    print("\nAnalysis complete.")
