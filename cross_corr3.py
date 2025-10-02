import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

import pickle
import numpy as np
import tensorstore as ts
from scipy.stats import zscore
from scipy.signal import correlate
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import networkx as nx
import multiprocessing as mp
from functools import partial

import cupy as cp
use_gpu = True

from zapbench import constants, data_utils

# --- Configuration ---
cfg = {
    'traces_path': 'file:///home/v/proj/zebra/data/traces',
    'max_neurons': None,
    'max_lag': 3,
    'n_threshold_samples': 100000,
    'p_value_threshold': 0.05,
    'output_file': 'connectivity_graph_global_threshold.pkl',
    'plot_dir': 'connectivity_plots_global',
    'n_workers': 5,  # None = use all CPUs
}


def batch_correlate_fft(signals_a, signals_b, max_lag):
    """
    Compute cross-correlation for batches of signal pairs using FFT.
    
    Args:
        signals_a: (n_pairs, n_timesteps)
        signals_b: (n_pairs, n_timesteps)
        max_lag: maximum lag
    
    Returns:
        correlations: (n_pairs, 2*max_lag+1) - correlations at each lag
    """
    n_pairs, n_timesteps = signals_a.shape
    
    # Pad to next power of 2 for FFT efficiency
    fft_size = 2 ** int(np.ceil(np.log2(2 * n_timesteps - 1)))
    
    # FFT of all signals
    fft_a = np.fft.fft(signals_a, n=fft_size, axis=1)
    fft_b = np.fft.fft(signals_b, n=fft_size, axis=1)
    
    # Cross-correlation in frequency domain
    cross_corr_fft = np.conj(fft_a) * fft_b
    
    # Inverse FFT
    cross_corr_full = np.fft.ifft(cross_corr_fft, axis=1).real
    
    # Extract the relevant lags
    center_idx = n_timesteps - 1
    cross_corr_full = np.roll(cross_corr_full, center_idx, axis=1)
    
    # Extract window around zero lag
    start = center_idx - max_lag
    end = center_idx + max_lag + 1
    correlations = cross_corr_full[:, start:end]
    
    # Normalize by length (for z-scored data, this gives correlation coefficient)
    correlations = correlations / n_timesteps
    
    return correlations


def batch_correlate_single_to_many(signal, batch_signals, max_lag, batch_size=1000):
    """
    GPU-accelerated correlation using CuPy - direct drop-in replacement.
    10-50x faster than CPU version.
    """

    n_signals, n_timesteps = batch_signals.shape

    # Increase batch size for GPU (better memory coalescing)
    if use_gpu:
        batch_size = min(batch_size * 10, n_signals)  # GPU can handle larger batches

    # Transfer to GPU once
    if use_gpu:
        signal_gpu = cp.asarray(signal)
        batch_signals_gpu = cp.asarray(batch_signals)
    else:
        signal_gpu = signal
        batch_signals_gpu = batch_signals

    # Pad for FFT
    fft_size = 2 ** int(cp.ceil(cp.log2(2 * n_timesteps - 1)))

    # Pre-compute FFT of source signal once
    signal_fft = cp.fft.fft(signal_gpu, n=fft_size)

    # Allocate output array ON GPU
    correlations = cp.zeros((n_signals, 2 * max_lag + 1))

    # Process in batches
    for i in range(0, n_signals, batch_size):
        end_idx = min(i + batch_size, n_signals)
        batch = batch_signals_gpu[i:end_idx]

        # FFT of batch - all on GPU
        batch_fft = cp.fft.fft(batch, n=fft_size, axis=1)

        # Cross-correlation
        cross_corr_fft = cp.conj(signal_fft)[cp.newaxis, :] * batch_fft
        cross_corr_full = cp.fft.ifft(cross_corr_fft, axis=1).real

        # Extract relevant lags
        center_idx = n_timesteps - 1
        cross_corr_full = cp.roll(cross_corr_full, center_idx, axis=1)

        start = center_idx - max_lag
        end = center_idx + max_lag + 1
        correlations[i:end_idx] = cross_corr_full[:, start:end]

    # Normalize
    correlations = correlations / n_timesteps

    # Transfer back to CPU only at the end
    if use_gpu:
        return cp.asnumpy(correlations)
    else:
        return correlations


# def batch_correlate_single_to_many(signal, batch_signals, max_lag, batch_size=1000):
#     """
#     Correlate one signal with many signals (vectorized with memory management).
#
#     Args:
#         signal: (n_timesteps,)
#         batch_signals: (n_signals, n_timesteps)
#         max_lag: maximum lag
#         batch_size: process this many signals at once to manage memory
#
#     Returns:
#         correlations: (n_signals, 2*max_lag+1)
#     """
#     n_signals, n_timesteps = batch_signals.shape
#
#     # Pad for FFT
#     fft_size = 2 ** int(np.ceil(np.log2(2 * n_timesteps - 1)))
#
#     # Pre-compute FFT of source signal once
#     signal_fft = np.fft.fft(signal, n=fft_size)
#
#     # Allocate output array
#     correlations = np.zeros((n_signals, 2 * max_lag + 1))
#
#     # Process in batches to manage memory
#     for i in range(0, n_signals, batch_size):
#         end_idx = min(i + batch_size, n_signals)
#         batch = batch_signals[i:end_idx]
#
#         # FFT of batch
#         batch_fft = np.fft.fft(batch, n=fft_size, axis=1)
#
#         # Cross-correlation
#         cross_corr_fft = np.conj(signal_fft)[np.newaxis, :] * batch_fft
#         cross_corr_full = np.fft.ifft(cross_corr_fft, axis=1).real
#
#         # Extract relevant lags
#         center_idx = n_timesteps - 1
#         cross_corr_full = np.roll(cross_corr_full, center_idx, axis=1)
#
#         start = center_idx - max_lag
#         end = center_idx + max_lag + 1
#         correlations[i:end_idx] = cross_corr_full[:, start:end]
#
#     # Normalize
#     correlations = correlations / n_timesteps
#
#     return correlations


def compute_global_threshold(data, n_samples, max_lag, p_value):
    """
    Compute global threshold from random neuron pairs using vectorized operations.
    
    Args:
        data: (n_timesteps, n_neurons) - z-scored neural activity
        n_samples: number of random pairs to sample
        max_lag: maximum lag to consider
        p_value: significance level
    
    Returns:
        threshold: scalar threshold value
        null_distribution: array of null correlations for diagnostics
    """
    num_timesteps, num_neurons = data.shape
    
    print(f"Computing global threshold from {n_samples} random pairs...")
    
    # Sample random pairs (vectorized)
    pairs_i = np.random.randint(0, num_neurons, size=n_samples)
    pairs_j = np.random.randint(0, num_neurons, size=n_samples)
    
    # Ensure i != j
    mask = pairs_i == pairs_j
    while mask.any():
        pairs_j[mask] = np.random.randint(0, num_neurons, size=mask.sum())
        mask = pairs_i == pairs_j
    
    # Extract all pairs at once
    signals_i = data[:, pairs_i].T  # (n_samples, n_timesteps)
    signals_j = data[:, pairs_j].T  # (n_samples, n_timesteps)
    
    # Compute all correlations using FFT (vectorized)
    print('Computing correlations...')
    correlations = batch_correlate_fft(signals_i, signals_j, max_lag)
    
    # Get max absolute correlation for each pair
    null_correlations = np.max(np.abs(correlations), axis=1)
    
    # Compute threshold
    threshold = np.percentile(null_correlations, (1 - p_value) * 100)
    
    print(f"Global threshold: {threshold:.4f}")
    print(f"Null distribution: mean={np.mean(null_correlations):.4f}, std={np.std(null_correlations):.4f}")
    
    return threshold, null_correlations


def analyze_source_neuron(source_idx, data, threshold, max_lag):
    """
    Analyze one source neuron against all subsequent target neurons.
    
    Args:
        source_idx: index of source neuron
        data: (n_timesteps, n_neurons) - z-scored data
        threshold: significance threshold
        max_lag: maximum lag
    
    Returns:
        list of (pre_synaptic, post_synaptic, strength, lag) tuples
    """

    num_timesteps, num_neurons = data.shape
    
    if source_idx >= num_neurons - 1:
        return []
    
    # Get all target indices
    target_indices = np.arange(source_idx + 1, num_neurons)
    n_targets = len(target_indices)
    
    if n_targets == 0:
        return []
    
    # Vectorized correlation computation
    source_signal = data[:, source_idx]
    target_signals = data[:, target_indices].T
    
    # Compute all correlations at once
    correlations = batch_correlate_single_to_many(source_signal, target_signals, max_lag)
    
    # Find peaks and their lags
    peak_vals = np.max(np.abs(correlations), axis=1)
    peak_lags = np.argmax(np.abs(correlations), axis=1) - max_lag
    
    # Find significant connections
    significant_mask = peak_vals > threshold
    significant_targets = target_indices[significant_mask]
    significant_lags = peak_lags[significant_mask]
    significant_strengths = peak_vals[significant_mask]
    
    # Determine directionality
    connections = []
    for target_idx, lag, strength in zip(significant_targets, significant_lags, significant_strengths):
        if lag > 0:
            # Positive lag: source leads target
            connections.append((source_idx, target_idx, strength, lag))
        elif lag < 0:
            # Negative lag: target leads source
            connections.append((target_idx, source_idx, strength, abs(lag)))
        # lag == 0: skip (no clear direction)
    
    print(f"Found {len(connections)} connections for source neuron {source_idx}", flush=True)
    return connections


def calculate_connectivity(data, threshold, max_lag, n_workers):
    """
    Calculate connectivity for all neuron pairs using parallel processing.
    
    Args:
        data: (n_timesteps, n_neurons) - z-scored
        threshold: pre-computed significance threshold
        max_lag: maximum lag
        n_workers: number of parallel workers
    
    Returns:
        connectivity: dict mapping post_synaptic -> list of (pre_synaptic, strength, lag) tuples
    """
    num_timesteps, num_neurons = data.shape
    connectivity = {i: [] for i in range(num_neurons)}
    
    print(f"Calculating connectivity using {n_workers} CPU workers...")
    
    # Create worker function with fixed arguments
    worker_func = partial(
        analyze_source_neuron,
        data=data,
        threshold=threshold,
        max_lag=max_lag
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
        for pre_synaptic, post_synaptic, strength, lag in connection_list:
            connectivity[post_synaptic].append((pre_synaptic, strength, lag))
    
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
    
    return traces


def plot_connectivity_examples(data, connectivity, max_lag, plot_dir, n_examples=5):
    """Plot examples of connected and unconnected neuron pairs."""
    print(f"Plotting connectivity examples to '{plot_dir}'...")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)
    
    num_timesteps, num_neurons = data.shape
    
    # Get connected pairs
    connected_pairs_set = set()
    for post_synaptic, pre_synaptics in connectivity.items():
        for pre_synaptic, strength, lag in pre_synaptics:
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
        fig_connected.suptitle('Examples of Significantly Connected Neuron Pairs', fontsize=16)
        for i, (neuron1, neuron2) in enumerate(connected_pairs[:n_examples]):
            plot_pair_analysis(data, neuron1, neuron2, max_lag, axes_connected[i])
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
        fig_unconnected.suptitle('Examples of Unconnected Neuron Pairs', fontsize=16)
        for i, (neuron1, neuron2) in enumerate(unconnected_pairs[:n_examples]):
            plot_pair_analysis(data, neuron1, neuron2, max_lag, axes_unconnected[i])
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
        for pre_synaptic, strength, lag in pre_synaptics:
            G.add_edge(pre_synaptic, post_synaptic, weight=strength, lag=lag)
    
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
    np.random.seed(42)
    
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Determine number of workers
    n_workers = cfg['n_workers'] if cfg['n_workers'] is not None else mp.cpu_count()
    print(f"Using {n_workers} CPU workers")
    
    # Load all data (ignoring conditions)
    neural_data = load_all_data(cfg['traces_path'], cfg['max_neurons'])
    
    # Compute global threshold from random pairs
    threshold, null_dist = compute_global_threshold(
        neural_data,
        cfg['n_threshold_samples'],
        cfg['max_lag'],
        cfg['p_value_threshold']
    )
    
    # Validate sparsity assumption
    print("\nValidating sparsity assumption...")
    test_samples = 1000
    _, test_dist = compute_global_threshold(
        neural_data,
        test_samples,
        cfg['max_lag'],
        p_value=1.0
    )
    sparsity_rate = (test_dist > threshold).sum() / test_samples
    print(f"Estimated connectivity rate: {sparsity_rate:.2%}")
    
    if sparsity_rate > 0.10:
        print("WARNING: Connectivity rate > 10%, sparsity assumption may be violated!")
    
    # Calculate connectivity
    connectivity_graph = calculate_connectivity(
        neural_data,
        threshold,
        cfg['max_lag'],
        n_workers
    )
    
    n_connections = sum(len(v) for v in connectivity_graph.values())
    print(f"\nFound {n_connections} significant connections")
    
    # Save results
    print(f"Saving connectivity graph to {cfg['output_file']}...")
    with open(cfg['output_file'], 'wb') as f:
        pickle.dump(connectivity_graph, f)
    
    # Save diagnostics
    diagnostics = {
        'threshold': threshold,
        'null_distribution': null_dist,
        'estimated_sparsity': sparsity_rate,
        'n_connections': n_connections,
        'config': cfg
    }
    diagnostics_file = cfg['output_file'].replace('.pkl', '_diagnostics.pkl')
    with open(diagnostics_file, 'wb') as f:
        pickle.dump(diagnostics, f)
    
    # Plot examples
    plot_connectivity_examples(neural_data, connectivity_graph, cfg['max_lag'], cfg['plot_dir'])
    
    # Plot network graph
    plot_connectivity_graph(connectivity_graph, cfg['plot_dir'])
    
    print("\nAnalysis complete.")
