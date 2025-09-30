import os
import pickle
import numpy as np
import tensorstore as ts
from scipy.signal import correlate
from scipy.stats import zscore
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import networkx as nx
import multiprocessing as mp
from functools import partial

# Try to import cupy for GPU acceleration
import cupy as cp
from cupyx.scipy.signal import correlate as cp_correlate
CUPY_AVAILABLE = True

from zapbench import constants, data_utils

# --- Configuration ---
cfg = {
    'traces_path': 'file:///home/v/proj/zebra/data/traces',
    'max_neurons': None,  # Limit neurons for faster processing, set to None for all
    'max_lag': 1,  # Maximum time lag in steps to consider for correlation
    'n_shuffles': 0,  # Number of shuffles for significance testing
    'p_value_threshold': 0.001, # Significance level
    'output_file': 'combined_connectivity_graph.pkl',
    'plot_dir': 'connectivity_plots',
    'use_gpu': True, # Set to False to use the CPU implementation
    'gpu_chunk_size': 2048, # Number of pairs to process in a batch on GPU to manage memory

    # --- Windowed Analysis Configuration ---
    'use_windowed_analysis': False,    # Enable to use windowed analysis for more robust connections
    'window_size_fraction': 0.2,      # e.g., 20% of the trace length
    'window_step_fraction': 0.1,      # e.g., 10% of the trace length
    'min_significant_windows': 5      # How many windows a pair must be significant in to count
}




def batch_cross_correlate_fft(signal, batch_signals, mode='full'):
    """
    Compute cross-correlation between a single signal and a batch of signals using FFT.
    
    Args:
        signal: 1D array of shape (n_timesteps,)
        batch_signals: 2D array of shape (n_signals, n_timesteps)
        mode: 'full', 'valid', or 'same' (currently only 'full' is implemented)
    
    Returns:
        2D array of shape (n_signals, correlation_length)
    """
    n_signals, n_timesteps = batch_signals.shape
    assert signal.shape[0] == n_timesteps, "Signal lengths must match"
    
    if mode != 'full':
        raise NotImplementedError("Only 'full' mode is currently implemented")
    
    # Determine output size for 'full' mode
    corr_length = 2 * n_timesteps - 1
    
    # Pad signals for FFT (need to pad to at least corr_length)
    fft_size = 2 ** int(cp.ceil(cp.log2(corr_length)))
    
    # Pad the single signal
    signal_padded = cp.pad(signal, (0, fft_size - n_timesteps), mode='constant')
    
    # Pad the batch of signals
    batch_padded = cp.pad(batch_signals, ((0, 0), (0, fft_size - n_timesteps)), mode='constant')
    
    # Compute FFT of all signals at once
    signal_fft = cp.fft.fft(signal_padded)
    batch_fft = cp.fft.fft(batch_padded, axis=1)
    
    # Cross-correlation in frequency domain: conj(fft(a)) * fft(b)
    # To compute correlate(signal, batch[i]), we use conj(signal_fft) * batch_fft
    corr_fft = cp.conj(signal_fft[cp.newaxis, :]) * batch_fft
    
    # Inverse FFT to get correlation
    corr_full = cp.fft.ifft(corr_fft, axis=1).real
    
    # Extract the valid part (first corr_length samples)
    # For 'full' mode, we need to rearrange the output
    # The standard correlate gives lags from -(n-1) to (n-1)
    # FFT correlation gives it in a different order, so we need to roll it
    corr_result = cp.roll(corr_full[:, :corr_length], n_timesteps - 1, axis=1)
    
    return corr_result


def batch_compute_significance_thresholds(signal, batch_signals, n_shuffles, p_value_threshold, max_lag):
    """
    Compute significance thresholds for a batch of neuron pairs using shuffling.
    Uses vectorized operations for efficiency.
    """
    n_signals, n_timesteps = batch_signals.shape
    
    # Pre-allocate array for max correlations
    max_corr_shuffled = cp.zeros((n_signals, n_shuffles))
    
    for shuffle_idx in range(n_shuffles):
        # Generate random permutation indices for all signals at once
        shuffle_indices = cp.random.rand(n_signals, n_timesteps).argsort(axis=1)
        shuffled_batch = cp.take_along_axis(batch_signals, shuffle_indices, axis=1)
        
        # Compute batch cross-correlation using FFT
        shuffled_corr = batch_cross_correlate_fft(signal, shuffled_batch)
        
        # Extract window around zero lag
        center_idx = shuffled_corr.shape[1] // 2
        start = center_idx - max_lag
        end = center_idx + max_lag + 1
        corr_window = shuffled_corr[:, start:end]
        
        # Store max absolute correlation for each signal
        max_corr_shuffled[:, shuffle_idx] = cp.max(cp.abs(corr_window), axis=1)
    
    # Compute significance thresholds
    if n_shuffles > 0:
        significance_thresholds = cp.percentile(max_corr_shuffled, (1 - p_value_threshold) * 100, axis=1)
    else:
        significance_thresholds = cp.zeros(n_signals) + 94.
    
    return significance_thresholds


def gpu_process_wrapper_optimized(gpu_id, neuron_indices, data, max_lag, n_shuffles, p_value_threshold, chunk_size, use_windowed_analysis, window_size, window_step, min_significant_windows, results_queue):
    """
    Optimized GPU wrapper using batch FFT-based cross-correlation.
    """
    cp.cuda.Device(gpu_id).use()
    data_gpu = cp.asarray(data)
    num_timesteps, num_neurons = data_gpu.shape
    mempool = cp.get_default_memory_pool()
    
    for i in tqdm(neuron_indices, desc=f"GPU {gpu_id}", position=gpu_id):
        if i >= num_neurons - 1:
            continue

        all_connections_for_i = []
        all_target_indices = np.arange(i + 1, num_neurons)

        # Loop over chunks of target neurons
        for k in range(0, len(all_target_indices), chunk_size):
            chunk_indices_np = all_target_indices[k:k+chunk_size]
            if chunk_indices_np.size == 0:
                continue

            chunk_indices = cp.asarray(chunk_indices_np)
            n_targets = chunk_indices.shape[0]
            
            # --- Windowed Analysis Logic ---
            if use_windowed_analysis:
                significant_window_counts = cp.zeros(n_targets, dtype=cp.int32)

                # Loop over time windows
                for t_start in range(0, num_timesteps - window_size, window_step):
                    t_end = t_start + window_size
                    
                    neuron_i_window = data_gpu[t_start:t_end, i]
                    targets_j_window = data_gpu[t_start:t_end, chunk_indices].T

                    significance_thresholds = batch_compute_significance_thresholds(
                        neuron_i_window, targets_j_window, n_shuffles, p_value_threshold, max_lag
                    )
                    actual_corr = batch_cross_correlate_fft(neuron_i_window, targets_j_window)
                    
                    center_idx = actual_corr.shape[1] // 2
                    start, end = center_idx - max_lag, center_idx + max_lag + 1
                    corr_window = actual_corr[:, start:end]
                    
                    peak_corr_vals = cp.max(cp.abs(corr_window), axis=1)
                    significant_mask_window = peak_corr_vals > significance_thresholds
                    significant_window_counts += significant_mask_window.astype(cp.int32)

                final_significant_mask = significant_window_counts >= min_significant_windows
                significant_indices_local = cp.where(final_significant_mask)[0]

            # --- Original (non-windowed) Analysis Logic ---
            else:
                neuron_i_activity = data_gpu[:, i]
                targets_j = data_gpu[:, chunk_indices].T
                significance_thresholds = batch_compute_significance_thresholds(
                    neuron_i_activity, targets_j, n_shuffles, p_value_threshold, max_lag
                )
                actual_corr = batch_cross_correlate_fft(neuron_i_activity, targets_j)
                center_idx = actual_corr.shape[1] // 2
                start, end = center_idx - max_lag, center_idx + max_lag + 1
                corr_window = actual_corr[:, start:end]
                peak_corr_vals = cp.max(cp.abs(corr_window), axis=1)
                final_significant_mask = peak_corr_vals > significance_thresholds
                significant_indices_local = cp.where(final_significant_mask)[0]

            # --- Aggregate results for the chunk ---
            if significant_indices_local.size > 0:
                # Re-calculate correlation on full trace to get the definitive final lag
                neuron_i_activity_full = data_gpu[:, i]
                significant_chunk_indices = chunk_indices[significant_indices_local]
                targets_j_full_significant = data_gpu[:, significant_chunk_indices].T

                final_corr = batch_cross_correlate_fft(neuron_i_activity_full, targets_j_full_significant)
                center_idx = final_corr.shape[1] // 2
                start, end = center_idx - max_lag, center_idx + max_lag + 1
                final_corr_window = final_corr[:, start:end]
                peak_lags = cp.argmax(cp.abs(final_corr_window), axis=1).get() - max_lag

                significant_global_js = chunk_indices[significant_indices_local]

                for idx, lag in enumerate(peak_lags):
                    j = significant_global_js[idx].item()
                    # With correlate(i, j), a positive lag means i -> j
                    if lag > 0:
                        all_connections_for_i.append((i, j))
                    else:
                        all_connections_for_i.append((j, i))
            
            mempool.free_all_blocks()

        if all_connections_for_i:
            results_queue.put(all_connections_for_i)


def calculate_connectivity_gpu_optimized(data, max_lag, n_shuffles, p_value_threshold, chunk_size, use_windowed_analysis, window_size, window_step, min_significant_windows):
    """
    Optimized GPU connectivity calculation using batch FFT cross-correlation.
    """
    if not cp or cp.cuda.runtime.getDeviceCount() == 0:
        print("CuPy not found or no GPUs available. Falling back to CPU implementation.")
        return None  # You can call your CPU implementation here
    
    num_gpus = cp.cuda.runtime.getDeviceCount()
    num_timesteps, num_neurons = data.shape
    connectivity = {i: [] for i in range(num_neurons)}
    
    print(f"Calculating functional connectivity using {num_gpus} GPUs (optimized FFT version)...")
    
    procs = []
    manager = mp.Manager()
    results_queue = manager.Queue()
    
    # Distribute neurons across GPUs
    neuron_chunks = [[] for _ in range(num_gpus)]
    for i in range(num_neurons):
        neuron_chunks[i % num_gpus].append(i)
    
    for gpu_id in range(num_gpus):
        p = mp.Process(target=gpu_process_wrapper_optimized, args=(
            gpu_id, neuron_chunks[gpu_id], data, max_lag, n_shuffles, 
            p_value_threshold, chunk_size, use_windowed_analysis, 
            window_size, window_step, min_significant_windows, results_queue
        ))
        procs.append(p)
        p.start()
    
    # Wait for all processes
    for p in procs:
        p.join()
    
    print("Aggregating results...")
    while not results_queue.empty():
        connection_list = results_queue.get()
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
    # Replace any NaNs that might result from zero variance channels
    traces = np.nan_to_num(traces)

    return traces

def analyze_neuron_worker(i, data, max_lag, n_shuffles, p_value_threshold):
    """Worker function to analyze one neuron against all subsequent neurons."""
    num_timesteps, num_neurons = data.shape
    neuron_i_activity = data[:, i]
    connections = []

    for j in range(i + 1, num_neurons):
        neuron_j_activity = data[:, j]

        # --- Significance Testing ---
        max_corr_shuffled = []
        for _ in range(n_shuffles):
            shuffled_j = np.random.permutation(neuron_j_activity)
            shuffled_corr = correlate(neuron_i_activity, shuffled_j, mode='full')
            max_corr_shuffled.append(np.max(np.abs(shuffled_corr)))
        
        significance_threshold = np.percentile(max_corr_shuffled, (1 - p_value_threshold) * 100)

        # --- Actual Cross-Correlation ---
        cross_corr = correlate(neuron_i_activity, neuron_j_activity, mode='full')
        lags = np.arange(-num_timesteps + 1, num_timesteps)

        center_idx = len(cross_corr) // 2
        start = center_idx - max_lag
        end = center_idx + max_lag + 1
        cross_corr_window = cross_corr[start:end]
        lags_window = lags[start:end]

        peak_corr_val = np.max(np.abs(cross_corr_window))
        
        if peak_corr_val > significance_threshold:
            peak_lag_idx = np.argmax(np.abs(cross_corr_window))
            peak_lag = lags_window[peak_lag_idx]
            if peak_lag > 0: # i -> j
                connections.append((i, j))
            else: # j -> i
                connections.append((j, i))
    return connections

# --- GPU Implementation ---

_worker_data_gpu = None

def init_worker_gpu(data_np, device_id):
    """Initializer for GPU worker processes."""
    global _worker_data_gpu
    cp.cuda.Device(device_id).use()
    _worker_data_gpu = cp.asarray(data_np)

def analyze_neuron_worker_gpu(i, max_lag, n_shuffles, p_value_threshold):
    """GPU worker to analyze one neuron against all subsequent neurons."""
    num_timesteps, num_neurons = _worker_data_gpu.shape
    neuron_i_activity = _worker_data_gpu[:, i]
    connections = []

    for j in range(i + 1, num_neurons):
        neuron_j_activity = _worker_data_gpu[:, j]

        # --- Significance Testing (on GPU) ---
        max_corr_shuffled = []
        for _ in range(n_shuffles):
            shuffled_j = cp.random.permutation(neuron_j_activity)
            shuffled_corr = cp_correlate(neuron_i_activity, shuffled_j, mode='full')
            max_corr_shuffled.append(cp.max(cp.abs(shuffled_corr)))
        
        # Percentile calculation on GPU requires a cupy array
        significance_threshold = cp.percentile(cp.array(max_corr_shuffled), (1 - p_value_threshold) * 100)

        # --- Actual Cross-Correlation (on GPU) ---
        cross_corr = cp_correlate(neuron_i_activity, neuron_j_activity, mode='full')
        
        center_idx = len(cross_corr) // 2
        start = center_idx - max_lag
        end = center_idx + max_lag + 1
        cross_corr_window = cross_corr[start:end]

        peak_corr_val = cp.max(cp.abs(cross_corr_window))
        
        if peak_corr_val > significance_threshold:
            peak_lag_idx = cp.argmax(cp.abs(cross_corr_window))
            # Get lag value back to CPU for the if condition
            peak_lag = (peak_lag_idx - max_lag).get()
            if peak_lag > 0: # i -> j
                connections.append((i, j))
            else: # j -> i
                connections.append((j, i))
    return connections

def calculate_connectivity_gpu(data, max_lag, n_shuffles, p_value_threshold, chunk_size):
    """Calculates connectivity on multiple GPUs."""
    if not CUPY_AVAILABLE or cp.cuda.runtime.getDeviceCount() == 0:
        print("CuPy not found or no GPUs available. Falling back to CPU implementation.")
        return calculate_connectivity_cpu(data, max_lag, n_shuffles, p_value_threshold)

    num_gpus = cp.cuda.runtime.getDeviceCount()
    num_timesteps, num_neurons = data.shape
    connectivity = {i: [] for i in range(num_neurons)}

    print(f"Calculating functional connectivity using {num_gpus} GPUs...")

    # Create a list of arguments for workers, assigning a device ID to each
    worker_args = [(i, i % num_gpus) for i in range(num_neurons)]

    # Create a pool of workers, one for each neuron, but the initializer ensures they use the correct GPU
    with mp.Pool(processes=mp.cpu_count()) as pool:
        # We need a way to initialize each worker with its specific GPU and data
        # A simple approach is to pass device_id and data to each task
        # A more complex but efficient way uses initializer, but that's harder with multiple devices.
        # Let's stick to a clear, if slightly less optimal, approach for now.
        # Re-evaluating: The best way is to launch separate pools or processes per GPU.
        # For simplicity with imap, let's pass the device ID and let the worker handle data loading.
        
        # Let's use a manager and separate process lists for each GPU for clarity
        
        # Simplified approach: The worker receives the device_id and loads data itself.
        # This is inefficient due to repeated data transfer. Let's try a better way.

        # Final approach: A single pool where each worker is told which GPU to use.
        # The initializer is tricky. Let's pass the device_id with the task.
        worker_func = partial(analyze_neuron_worker_gpu, max_lag=max_lag, n_shuffles=n_shuffles, p_value_threshold=p_value_threshold)
        
        # This is a bit of a trick. We can't use an initializer with different args.
        # So, the worker will have to be responsible for setting its device and loading data.
        # This means data is copied for each task, which is slow.
        # The best performant way is to have one process per GPU managing its own tasks.
        
        # Let's go with a simple but correct multiprocessing implementation.
        # We'll map chunks of neurons to each GPU process.
        
        procs = []
        manager = mp.Manager()
        results_queue = manager.Queue()
        # Distribute work in a round-robin fashion for better load balancing
        neuron_chunks = [[] for _ in range(num_gpus)]
        for i in range(num_neurons):
            neuron_chunks[i % num_gpus].append(i)

        for gpu_id in range(num_gpus):
            p = mp.Process(target=gpu_process_wrapper, args=(
                gpu_id, neuron_chunks[gpu_id], data, max_lag, n_shuffles, p_value_threshold, chunk_size, results_queue
            ))
            procs.append(p)
            p.start()

        # Wait for all processes to finish
        for p in procs:
            p.join()

        print("Aggregating results...")
        while not results_queue.empty():
            connection_list = results_queue.get()
            for pre_synaptic, post_synaptic in connection_list:
                connectivity[post_synaptic].append(pre_synaptic)

    return connectivity



# --- CPU Implementation ---

def calculate_connectivity_cpu(data, max_lag, n_shuffles, p_value_threshold):
    """Calculates functional connectivity between neurons using multiprocessing."""
    num_timesteps, num_neurons = data.shape
    connectivity = {i: [] for i in range(num_neurons)}
    
    print("Calculating functional connectivity using multiprocessing...")
    
    # Use partial to create a function with fixed arguments for the worker pool
    worker_func = partial(analyze_neuron_worker, data=data, max_lag=max_lag, n_shuffles=n_shuffles, p_value_threshold=p_value_threshold)
    num_workers = 16

    with mp.Pool(num_workers) as pool:
        # Use imap_unordered for efficiency and tqdm for progress bar
        results = list(tqdm(pool.imap_unordered(worker_func, range(num_neurons)), total=num_neurons, desc="Analyzing neurons"))

    print("Aggregating results...")
    for connection_list in results:
        for pre_synaptic, post_synaptic in connection_list:
            connectivity[post_synaptic].append(pre_synaptic)
            
    return connectivity

def plot_connectivity_examples(data, connectivity, max_lag, plot_dir, n_examples=5):
    """Plots examples of connected and unconnected neuron pairs."""
    print(f"Plotting connectivity examples to '{plot_dir}'...")
    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir)

    num_timesteps, num_neurons = data.shape

    # Efficiently get a set of connected pairs
    connected_pairs_set = set()
    for post_synaptic, pre_synaptics in connectivity.items():
        for pre_synaptic in pre_synaptics:
            connected_pairs_set.add(tuple(sorted((pre_synaptic, post_synaptic))))
    
    connected_pairs = list(connected_pairs_set)
    random.shuffle(connected_pairs)

    # Find unconnected pairs by random sampling to avoid memory overload
    unconnected_pairs = []
    attempts = 0
    max_attempts = n_examples * 1000 # Safeguard against infinite loops
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
        print(f"Warning: Could only find {len(unconnected_pairs)} unconnected pairs after {max_attempts} attempts.")

    # Plot connected pairs
    fig_connected, axes_connected = plt.subplots(n_examples, 2, figsize=(12, 4 * n_examples), constrained_layout=True)
    fig_connected.suptitle('Examples of Significantly Connected Neuron Pairs', fontsize=16)
    for i, (neuron1, neuron2) in enumerate(connected_pairs[:n_examples]):
        plot_pair_analysis(data, neuron1, neuron2, max_lag, axes_connected[i])
    plt.savefig(os.path.join(plot_dir, "connected_pairs.png"))
    plt.close(fig_connected)

    # Plot unconnected pairs
    fig_unconnected, axes_unconnected = plt.subplots(n_examples, 2, figsize=(12, 4 * n_examples), constrained_layout=True)
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
    peak_val = cross_corr_window[peak_lag_idx]
    axes[1].axvline(peak_lag, color='r', linestyle='--', label=f'Peak Lag: {peak_lag}')
    axes[1].set_title('Cross-Correlation')
    axes[1].set_xlabel('Time Lag (steps)')
    axes[1].set_ylabel('Correlation')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)


if __name__ == '__main__':
    os.environ['PYTHONWARNINGS'] = 'ignore::UserWarning,ignore::FutureWarning'

    # Set start method for multiprocessing
    try:
        mp.set_start_method('spawn')
    except RuntimeError:
        pass # Start method can only be set once

    all_conditions_connectivity = {}
    num_neurons = None

    for condition_name in constants.CONDITION_NAMES:
        print(f"\n--- Processing condition: {condition_name} ---")
        condition_plot_dir = os.path.join(cfg['plot_dir'], condition_name)

        # 1. Load Data
        neural_data = load_preprocessed_data(
            cfg['traces_path'],
            condition_name,
            cfg['max_neurons']
        )

        if num_neurons is None:
            num_neurons = neural_data.shape[1]

        # 2. Calculate Connectivity
        # Dynamic window size calculation
        num_timesteps_condition = neural_data.shape[0]
        window_size = int(num_timesteps_condition * cfg['window_size_fraction'])
        window_step = int(num_timesteps_condition * cfg['window_step_fraction'])

        if cfg['use_gpu']:
            connectivity_graph = calculate_connectivity_gpu_optimized(
                neural_data,
                cfg['max_lag'],
                cfg['n_shuffles'],
                cfg['p_value_threshold'],
                cfg['gpu_chunk_size'],
                cfg['use_windowed_analysis'],
                window_size, # Pass dynamic value
                window_step, # Pass dynamic value
                cfg['min_significant_windows']
            )
        else:
            connectivity_graph = calculate_connectivity_cpu(
                neural_data,
                cfg['max_lag'],
                cfg['n_shuffles'],
                cfg['p_value_threshold']
            )
        all_conditions_connectivity[condition_name] = connectivity_graph

        # 3. Save individual results
        condition_output_file = os.path.join(condition_plot_dir, 'connectivity_graph.pkl')
        print(f"Saving connectivity graph for '{condition_name}' to {condition_output_file}...")
        if not os.path.exists(condition_plot_dir):
            os.makedirs(condition_plot_dir)
        with open(condition_output_file, 'wb') as f:
            pickle.dump(connectivity_graph, f)

        print(f"Found {sum(len(v) for v in connectivity_graph.values())} significant connections in '{condition_name}'.")

        # 4. Plot examples for the condition
        plot_connectivity_examples(neural_data, connectivity_graph, cfg['max_lag'], condition_plot_dir)

        # 5. Plot network graph for the condition
        plot_connectivity_graph(connectivity_graph, condition_plot_dir)

    # --- Merge and Save Combined Results ---
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
