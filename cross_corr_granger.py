import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pickle
import numpy as np
import tensorstore as ts
from scipy.stats import zscore, f as f_dist
from scipy.signal import correlate
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import networkx as nx
import multiprocessing as mp
from functools import partial

from zapbench import constants, data_utils

# --- Configuration ---
cfg = {
    'traces_path': 'file:///home/v/proj/zebra/data/traces',
    'max_neurons': None,
    'max_order': 3,  # AR model order for Granger causality
    'p_value_threshold': 0.01,  # Will be Bonferroni corrected automatically
    'output_file': 'connectivity_graph_granger.pkl',
    'plot_dir': 'connectivity_plots_granger',
    'n_workers': 10,
}


def build_design_matrices(signal_A, signal_B, max_order):
    """
    Build design matrices for Granger causality test.
    
    Args:
        signal_A: (n_timesteps,) - source signal
        signal_B: (n_timesteps,) - target signal
        max_order: AR model order
    
    Returns:
        X_restricted: (n_samples, max_order) - lagged B only
        X_full: (n_samples, 2*max_order) - lagged B and A
        y: (n_samples,) - B[t]
    """
    p = max_order
    n = len(signal_B)
    n_samples = n - p
    
    # Target: B[t] for t = p, p+1, ..., n-1
    y = signal_B[p:]
    
    # Restricted model: [B[t-1], B[t-2], ..., B[t-p]]
    X_restricted = np.column_stack([
        signal_B[p-lag-1:n-lag-1] for lag in range(p)
    ])
    
    # Full model: [B[t-1], ..., B[t-p], A[t-1], ..., A[t-p]]
    X_full = np.column_stack([
        signal_B[p-lag-1:n-lag-1] for lag in range(p)
    ] + [
        signal_A[p-lag-1:n-lag-1] for lag in range(p)
    ])
    
    return X_restricted, X_full, y


def compute_F_statistic(X_restricted, X_full, y):
    """
    Compute F-statistic for Granger causality test.
    
    Args:
        X_restricted: (n_samples, p) - restricted model design matrix
        X_full: (n_samples, 2p) - full model design matrix
        y: (n_samples,) - target variable
    
    Returns:
        F: F-statistic
    """
    n_samples = len(y)
    p = X_restricted.shape[1]
    
    # Fit restricted model: B ~ B_lags
    beta_r = np.linalg.lstsq(X_restricted, y, rcond=None)[0]
    residuals_r = y - X_restricted @ beta_r
    RSS_r = np.sum(residuals_r ** 2)
    
    # Fit full model: B ~ B_lags + A_lags
    beta_f = np.linalg.lstsq(X_full, y, rcond=None)[0]
    residuals_f = y - X_full @ beta_f
    RSS_f = np.sum(residuals_f ** 2)
    
    # F-statistic: F = ((RSS_r - RSS_f) / p) / (RSS_f / (n - 2p - 1))
    if RSS_f < 1e-10:  # Avoid division by zero
        return 0.0
    
    F = ((RSS_r - RSS_f) / p) / (RSS_f / (n_samples - 2*p - 1))
    
    return max(0.0, F)  # F-statistic should be non-negative


def batch_granger_test(signal_A, batch_signals_B, max_order):
    """
    Test if A Granger-causes each B in batch.
    
    Args:
        signal_A: (n_timesteps,) - source signal
        batch_signals_B: (n_signals, n_timesteps) - target signals
        max_order: AR model order
    
    Returns:
        F_stats: (n_signals,) - F-statistic for each A→B test
    """
    n_signals = batch_signals_B.shape[0]
    F_stats = np.zeros(n_signals)
    
    for i in range(n_signals):
        signal_B = batch_signals_B[i]
        X_r, X_f, y = build_design_matrices(signal_A, signal_B, max_order)
        F_stats[i] = compute_F_statistic(X_r, X_f, y)
    
    return F_stats


def batch_granger_test_reverse(batch_signals_B, signal_A, max_order):
    """
    Test if each B Granger-causes A.
    
    Args:
        batch_signals_B: (n_signals, n_timesteps) - source signals
        signal_A: (n_timesteps,) - target signal
        max_order: AR model order
    
    Returns:
        F_stats: (n_signals,) - F-statistic for each B→A test
    """
    n_signals = batch_signals_B.shape[0]
    F_stats = np.zeros(n_signals)
    
    for i in range(n_signals):
        signal_B = batch_signals_B[i]
        X_r, X_f, y = build_design_matrices(signal_B, signal_A, max_order)
        F_stats[i] = compute_F_statistic(X_r, X_f, y)
    
    return F_stats


def compute_theoretical_threshold(n_timesteps, max_order, p_value, num_neurons):
    """
    Compute theoretical F-distribution threshold with Bonferroni correction.
    
    Args:
        n_timesteps: number of time points
        max_order: AR model order
        p_value: significance level (before correction)
        num_neurons: number of neurons (for Bonferroni correction)
    
    Returns:
        threshold: F-statistic threshold
    """
    # Bonferroni correction for multiple comparisons
    n_comparisons = num_neurons * (num_neurons - 1)  # Test both directions
    p_corrected = p_value / n_comparisons
    
    dfn = max_order  # degrees of freedom numerator
    dfd = n_timesteps - 2*max_order - 1  # degrees of freedom denominator
    
    threshold = f_dist.ppf(1 - p_corrected, dfn, dfd)
    
    return threshold


def analyze_source_neuron(source_idx, data, threshold, max_order):
    """
    Analyze one source neuron against all subsequent target neurons using Granger causality.
    
    Args:
        source_idx: index of source neuron
        data: (n_timesteps, n_neurons) - z-scored data
        threshold: F-statistic threshold
        max_order: AR model order
    
    Returns:
        list of (pre_synaptic, post_synaptic, F_statistic) tuples
    """

    num_timesteps, num_neurons = data.shape
    
    if source_idx >= num_neurons - 1:
        return []
    
    # Get all target indices
    target_indices = np.arange(source_idx + 1, num_neurons)
    n_targets = len(target_indices)
    
    if n_targets == 0:
        return []
    
    # Get signals
    source_signal = data[:, source_idx]
    target_signals = data[:, target_indices].T
    
    # Test both directions
    F_A_to_B = batch_granger_test(source_signal, target_signals, max_order)
    F_B_to_A = batch_granger_test_reverse(target_signals, source_signal, max_order)
    
    # Determine significant connections
    sig_A_to_B = F_A_to_B > threshold
    sig_B_to_A = F_B_to_A > threshold
    
    # Build connection list with F-statistics
    connections = []
    for i, target_idx in enumerate(target_indices):
        if sig_A_to_B[i] and not sig_B_to_A[i]:
            # Only A→B is significant
            connections.append((source_idx, target_idx, F_A_to_B[i]))
        elif sig_B_to_A[i] and not sig_A_to_B[i]:
            # Only B→A is significant
            connections.append((target_idx, source_idx, F_B_to_A[i]))
        elif sig_A_to_B[i] and sig_B_to_A[i]:
            # Both directions significant - choose stronger one
            if F_A_to_B[i] > F_B_to_A[i]:
                connections.append((source_idx, target_idx, F_A_to_B[i]))
            else:
                connections.append((target_idx, source_idx, F_B_to_A[i]))
        # If neither significant, skip
    
    # print(f"Found {len(connections)} connections for source neuron {source_idx}", flush=True)
    return connections


def calculate_connectivity(data, threshold, max_order, n_workers):
    """
    Calculate connectivity for all neuron pairs using parallel processing.
    
    Args:
        data: (n_timesteps, n_neurons) - z-scored
        threshold: F-statistic threshold
        max_order: AR model order
        n_workers: number of parallel workers
    
    Returns:
        connectivity: dict mapping post_synaptic -> list of pre_synaptic neurons
    """
    num_timesteps, num_neurons = data.shape
    connectivity = {i: [] for i in range(num_neurons)}
    
    print(f"Calculating connectivity using {n_workers} CPU workers...")
    
    # Create worker function with fixed arguments
    worker_func = partial(
        analyze_source_neuron,
        data=data,
        threshold=threshold,
        max_order=max_order
    )
    
    # Parallel processing
    with mp.Pool(n_workers) as pool:
        results = list(tqdm(
            pool.imap_unordered(worker_func, range(num_neurons)),
            total=num_neurons,
            desc="Analyzing neurons"
        ))
    
    # Aggregate results
    print("Aggregating results...")
    for connection_list in results:
        for pre_synaptic, post_synaptic, F_stat in connection_list:
            connectivity[post_synaptic].append((pre_synaptic, F_stat))
    
    return connectivity


def load_all_data(traces_path, max_neurons=None):
    """Load all neural activity data (ignoring conditions)."""
    print("Loading all data...")
    ds_traces = ts.open({'driver': 'zarr3', 'kvstore': traces_path}).result()
    
    # Load all traces
    traces = ds_traces[:, :].read().result()
    
    if max_neurons is not None:
        traces = traces[:, :max_neurons]
    
    print(f"Loaded traces with shape: {traces.shape}")
    
    print("Preprocessing data (z-scoring)...")
    traces = zscore(traces, axis=0)
    traces = np.nan_to_num(traces)
    
    print("Removing common input (subtracting population average)...")
    population_avg = np.mean(traces, axis=1, keepdims=True)
    traces = traces - population_avg
    
    print("Re-normalizing after removing common input...")
    traces = zscore(traces, axis=0)
    traces = np.nan_to_num(traces)
    
    return traces


def plot_connectivity_examples(data, connectivity, max_order, plot_dir, n_examples=5):
    """Plot examples of connected and unconnected neuron pairs."""
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
        fig_connected, axes_connected = plt.subplots(
            min(n_examples, len(connected_pairs)), 2,
            figsize=(12, 4 * min(n_examples, len(connected_pairs))),
            constrained_layout=True
        )
        if min(n_examples, len(connected_pairs)) == 1:
            axes_connected = [axes_connected]
        fig_connected.suptitle('Examples of Granger-Causal Neuron Pairs', fontsize=16)
        for i, (neuron1, neuron2) in enumerate(connected_pairs[:n_examples]):
            plot_pair_analysis(data, neuron1, neuron2, max_order, axes_connected[i])
        plt.savefig(os.path.join(plot_dir, "connected_pairs.png"))
        plt.close(fig_connected)
    
    # Plot unconnected pairs
    if len(unconnected_pairs) > 0:
        fig_unconnected, axes_unconnected = plt.subplots(
            min(n_examples, len(unconnected_pairs)), 2,
            figsize=(12, 4 * min(n_examples, len(unconnected_pairs))),
            constrained_layout=True
        )
        if min(n_examples, len(unconnected_pairs)) == 1:
            axes_unconnected = [axes_unconnected]
        fig_unconnected.suptitle('Examples of Non-Causal Neuron Pairs', fontsize=16)
        for i, (neuron1, neuron2) in enumerate(unconnected_pairs[:n_examples]):
            plot_pair_analysis(data, neuron1, neuron2, max_order, axes_unconnected[i])
        plt.savefig(os.path.join(plot_dir, "unconnected_pairs.png"))
        plt.close(fig_unconnected)
    
    print("Plotting complete.")


def plot_connectivity_graph(connectivity, plot_dir):
    """Create and plot a directed graph of neuron connectivity."""
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
    plt.title("Granger Causality Network")
    plt.savefig(os.path.join(plot_dir, "connectivity_network.png"))
    plt.close()
    print("Network graph plotting complete.")


def plot_pair_analysis(data, neuron_i_idx, neuron_j_idx, max_order, axes):
    """Helper function to plot analysis for a single pair of neurons."""
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
    
    # Compute Granger F-statistics
    X_r_ij, X_f_ij, y_j = build_design_matrices(activity_i, activity_j, max_order)
    F_i_to_j = compute_F_statistic(X_r_ij, X_f_ij, y_j)
    
    X_r_ji, X_f_ji, y_i = build_design_matrices(activity_j, activity_i, max_order)
    F_j_to_i = compute_F_statistic(X_r_ji, X_f_ji, y_i)
    
    # Plot cross-correlation for reference
    cross_corr = correlate(activity_i, activity_j, mode='full')
    lags = np.arange(-num_timesteps + 1, num_timesteps)
    
    center_idx = len(cross_corr) // 2
    max_lag_plot = min(20, num_timesteps // 2)
    start = center_idx - max_lag_plot
    end = center_idx + max_lag_plot + 1
    cross_corr_window = cross_corr[start:end]
    lags_window = lags[start:end]
    
    axes[1].plot(lags_window, cross_corr_window)
    peak_lag_idx = np.argmax(np.abs(cross_corr_window))
    peak_lag = lags_window[peak_lag_idx]
    axes[1].axvline(peak_lag, color='r', linestyle='--', label=f'Peak Lag: {peak_lag}')
    axes[1].set_title(f'Cross-Correlation\nF(i→j)={F_i_to_j:.2f}, F(j→i)={F_j_to_i:.2f}')
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
    
    # Load all data (ignoring conditions)
    neural_data = load_all_data(cfg['traces_path'], cfg['max_neurons'])
    
    num_timesteps, num_neurons = neural_data.shape
    
    # Compute theoretical threshold from F-distribution with Bonferroni correction
    threshold = compute_theoretical_threshold(
        num_timesteps,
        cfg['max_order'],
        cfg['p_value_threshold'],
        num_neurons
    )
    
    n_comparisons = num_neurons * (num_neurons - 1)
    p_corrected = cfg['p_value_threshold'] / n_comparisons
    
    print(f"\nGranger causality parameters:")
    print(f"  AR model order: {cfg['max_order']}")
    print(f"  Number of neurons: {num_neurons}")
    print(f"  Number of comparisons: {n_comparisons:,}")
    print(f"  Significance level (uncorrected): {cfg['p_value_threshold']}")
    print(f"  Significance level (Bonferroni): {p_corrected:.2e}")
    print(f"  F-statistic threshold: {threshold:.4f}")
    print(f"  Degrees of freedom: ({cfg['max_order']}, {num_timesteps - 2*cfg['max_order'] - 1})")
    
    # Calculate connectivity
    connectivity_graph = calculate_connectivity(
        neural_data,
        threshold,
        cfg['max_order'],
        n_workers
    )
    
    n_connections = sum(len(v) for v in connectivity_graph.values())
    total_possible = num_neurons * (num_neurons - 1)
    connectivity_rate = n_connections / total_possible
    
    print(f"\nFound {n_connections} significant Granger-causal connections")
    print(f"Connectivity rate: {connectivity_rate:.4%}")
    
    # Save results
    print(f"Saving connectivity graph to {cfg['output_file']}...")
    with open(cfg['output_file'], 'wb') as f:
        pickle.dump(connectivity_graph, f)
    
    # HOW TO FILTER LATER:
    # 
    # Option 1: Keep top K strongest connections per neuron
    # filtered = {post: sorted(conns, key=lambda x: x[1], reverse=True)[:50]
    #             for post, conns in connectivity_graph.items()}
    #
    # Option 2: Keep top X% globally
    # all_F_stats = [F for conns in connectivity_graph.values() for (_, F) in conns]
    # threshold = np.percentile(all_F_stats, 90)  # Keep top 10%
    # filtered = {post: [(pre, F) for (pre, F) in conns if F > threshold]
    #             for post, conns in connectivity_graph.items()}
    #
    # Option 3: Adaptive per-neuron threshold
    # filtered = {}
    # for post, conns in connectivity_graph.items():
    #     if len(conns) > 0:
    #         F_values = [F for (_, F) in conns]
    #         local_thresh = np.mean(F_values) + 2*np.std(F_values)
    #         filtered[post] = [(pre, F) for (pre, F) in conns if F > local_thresh]
    
    # Save diagnostics
    diagnostics = {
        'threshold': threshold,
        'max_order': cfg['max_order'],
        'n_connections': n_connections,
        'connectivity_rate': connectivity_rate,
        'config': cfg
    }
    diagnostics_file = cfg['output_file'].replace('.pkl', '_diagnostics.pkl')
    with open(diagnostics_file, 'wb') as f:
        pickle.dump(diagnostics, f)
    
    # Plot examples
    plot_connectivity_examples(neural_data, connectivity_graph, cfg['max_order'], cfg['plot_dir'])
    
    # Plot network graph
    plot_connectivity_graph(connectivity_graph, cfg['plot_dir'])
    
    print("\nAnalysis complete.")
