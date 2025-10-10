#!/usr/bin/env python

"""
Analyze and visualize connectivity graph statistics from pickle files.
Compatible with outputs from cross_corr3.py, pos_based_con.py, and random_con.py.

All connectivity graphs have the format:
    dict mapping post_synaptic_id -> list of (pre_synaptic_id, strength, lag_or_pos) tuples
"""

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from collections import Counter


def load_connectivity(filepath):
    """Load connectivity graph from pickle file."""
    print(f"Loading connectivity graph from {filepath}...")
    with open(filepath, 'rb') as f:
        connectivity = pickle.load(f)
    
    # Validate format
    if not isinstance(connectivity, dict):
        raise ValueError("Expected connectivity to be a dictionary")
    
    print(f"Loaded connectivity with {len(connectivity)} neurons")
    return connectivity


def extract_statistics(connectivity):
    """
    Extract comprehensive statistics from connectivity graph.
    
    Returns:
        dict with various statistics
    """
    print("Extracting statistics...")
    
    stats = {}
    
    # Basic counts
    num_neurons = len(connectivity)
    stats['num_neurons'] = num_neurons
    
    # In-degree (incoming connections)
    in_degrees = {nid: len(connections) for nid, connections in connectivity.items()}
    stats['in_degrees'] = in_degrees
    
    # Out-degree (outgoing connections)
    out_degrees = {nid: 0 for nid in connectivity.keys()}
    for post_synaptic_id, connections in connectivity.items():
        for pre_synaptic_id, _, _ in connections:
            if pre_synaptic_id in out_degrees:
                out_degrees[pre_synaptic_id] += 1
            else:
                # Handle case where pre_synaptic neuron not in connectivity dict
                out_degrees[pre_synaptic_id] = 1
    stats['out_degrees'] = out_degrees
    
    # Connection strengths
    strengths = []
    for connections in connectivity.values():
        for _, strength, _ in connections:
            strengths.append(strength)
    stats['strengths'] = np.array(strengths)
    
    # Third parameter (lag or position difference)
    third_params = []
    for connections in connectivity.values():
        for _, _, param in connections:
            third_params.append(param)
    stats['third_params'] = third_params
    
    # Total connections
    total_connections = sum(len(v) for v in connectivity.values())
    stats['total_connections'] = total_connections
    
    # Density (actual connections / possible connections)
    max_possible = num_neurons * (num_neurons - 1)  # directed graph, no self-loops
    stats['density'] = total_connections / max_possible if max_possible > 0 else 0
    
    # Reciprocal connections (A->B and B->A)
    reciprocal_count = 0
    edge_set = set()
    for post_id, connections in connectivity.items():
        for pre_id, _, _ in connections:
            edge_set.add((pre_id, post_id))
    
    for pre_id, post_id in edge_set:
        if (post_id, pre_id) in edge_set:
            reciprocal_count += 1
    stats['reciprocal_connections'] = reciprocal_count // 2  # Each pair counted twice
    
    print(f"  Total connections: {total_connections}")
    print(f"  Density: {stats['density']:.6f}")
    print(f"  Reciprocal connections: {stats['reciprocal_connections']}")
    
    return stats


def detect_third_param_type(third_params):
    """
    Detect whether third parameter is lag (scalar) or position difference (tuple).
    
    Returns:
        'lag' or 'position'
    """
    if len(third_params) == 0:
        return 'unknown'
    
    sample = third_params[0]
    if isinstance(sample, (tuple, list, np.ndarray)):
        return 'position'
    else:
        return 'lag'


def plot_degree_distributions(stats, ax1, ax2):
    """Plot in-degree and out-degree distributions."""
    in_degree_values = list(stats['in_degrees'].values())
    out_degree_values = list(stats['out_degrees'].values())
    
    # In-degree histogram
    ax1.hist(in_degree_values, bins=50, alpha=0.7, color='blue', edgecolor='black')
    ax1.set_xlabel('In-Degree (Incoming Connections)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.set_title(f'In-Degree Distribution\n(mean={np.mean(in_degree_values):.2f}, '
                  f'std={np.std(in_degree_values):.2f})', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axvline(np.mean(in_degree_values), color='red', linestyle='--', 
                linewidth=2, label=f'Mean={np.mean(in_degree_values):.2f}')
    ax1.legend()
    
    # Out-degree histogram
    ax2.hist(out_degree_values, bins=50, alpha=0.7, color='green', edgecolor='black')
    ax2.set_xlabel('Out-Degree (Outgoing Connections)', fontsize=10)
    ax2.set_ylabel('Frequency', fontsize=10)
    ax2.set_title(f'Out-Degree Distribution\n(mean={np.mean(out_degree_values):.2f}, '
                  f'std={np.std(out_degree_values):.2f})', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.axvline(np.mean(out_degree_values), color='red', linestyle='--', 
                linewidth=2, label=f'Mean={np.mean(out_degree_values):.2f}')
    ax2.legend()


def plot_strength_distribution(stats, ax):
    """Plot connection strength distribution."""
    strengths = stats['strengths']
    
    ax.hist(strengths, bins=50, alpha=0.7, color='red', edgecolor='black')
    ax.set_xlabel('Connection Strength', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'Strength Distribution\n(mean={np.mean(strengths):.4f}, '
                 f'std={np.std(strengths):.4f}, median={np.median(strengths):.4f})', 
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(strengths), color='blue', linestyle='--', 
               linewidth=2, label=f'Mean={np.mean(strengths):.4f}')
    ax.axvline(np.median(strengths), color='orange', linestyle='--', 
               linewidth=2, label=f'Median={np.median(strengths):.4f}')
    ax.legend()


def plot_lag_distribution(third_params, ax):
    """Plot lag distribution (for cross-correlation graphs)."""
    lags = np.array(third_params)
    
    if len(lags) == 0:
        ax.text(0.5, 0.5, 'No lag data', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Create bins for integer lags
    unique_lags = np.unique(lags)
    if len(unique_lags) < 50:
        bins = np.arange(lags.min() - 0.5, lags.max() + 1.5, 1)
    else:
        bins = 50
    
    ax.hist(lags, bins=bins, alpha=0.7, color='purple', edgecolor='black')
    ax.set_xlabel('Time Lag (timesteps)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'Lag Distribution\n(mean={np.mean(lags):.2f}, '
                 f'std={np.std(lags):.2f}, range=[{lags.min()}, {lags.max()}])', 
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(lags), color='red', linestyle='--', 
               linewidth=2, label=f'Mean={np.mean(lags):.2f}')
    ax.legend()


def plot_position_distribution(third_params, ax):
    """Plot position difference distribution (for position-based graphs)."""
    # Extract position differences
    positions = np.array(third_params)
    
    if len(positions) == 0:
        ax.text(0.5, 0.5, 'No position data', ha='center', va='center', transform=ax.transAxes)
        return
    
    # Calculate Euclidean distances
    distances = np.linalg.norm(positions, axis=1)
    
    ax.hist(distances, bins=50, alpha=0.7, color='orange', edgecolor='black')
    ax.set_xlabel('Euclidean Distance', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.set_title(f'Position Distance Distribution\n(mean={np.mean(distances):.2f}, '
                 f'std={np.std(distances):.2f})', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.axvline(np.mean(distances), color='red', linestyle='--', 
               linewidth=2, label=f'Mean={np.mean(distances):.2f}')
    ax.legend()


def plot_degree_correlation(stats, ax):
    """Plot correlation between in-degree and out-degree."""
    # Get neurons that appear in both dictionaries
    common_neurons = set(stats['in_degrees'].keys()) & set(stats['out_degrees'].keys())
    
    in_degrees = [stats['in_degrees'][nid] for nid in common_neurons]
    out_degrees = [stats['out_degrees'][nid] for nid in common_neurons]
    
    ax.scatter(in_degrees, out_degrees, alpha=0.3, s=10)
    ax.set_xlabel('In-Degree', fontsize=10)
    ax.set_ylabel('Out-Degree', fontsize=10)
    
    # Calculate correlation
    if len(in_degrees) > 1:
        corr = np.corrcoef(in_degrees, out_degrees)[0, 1]
        ax.set_title(f'In-Degree vs Out-Degree\n(correlation={corr:.3f})', fontsize=11)
    else:
        ax.set_title('In-Degree vs Out-Degree', fontsize=11)
    
    ax.grid(True, alpha=0.3)


def plot_strength_vs_degree(stats, ax):
    """Plot relationship between connection strength and node degree."""
    # Calculate average strength per neuron (for incoming connections)
    neuron_avg_strengths = []
    neuron_in_degrees = []
    
    for nid, connections in stats['in_degrees'].items():
        if connections > 0:
            # Get strengths for this neuron
            strengths_for_neuron = []
            # Need to find this neuron's connections in the original connectivity
            # We'll use the stats we already have
            neuron_in_degrees.append(connections)
    
    # Alternative: plot strength vs out-degree
    # For each connection, get the out-degree of the source neuron
    connectivity_temp = {}
    for post_id in stats['in_degrees'].keys():
        connectivity_temp[post_id] = []
    
    # We need the original connectivity for this - skip for now or use simpler plot
    ax.text(0.5, 0.5, 'Strength vs Degree analysis\nrequires original connectivity', 
            ha='center', va='center', transform=ax.transAxes, fontsize=10)


def plot_cumulative_degree_distribution(stats, ax):
    """Plot cumulative degree distribution (log-log for scale-free detection)."""
    in_degree_values = np.array(list(stats['in_degrees'].values()))
    
    # Get degree counts
    degree_counts = Counter(in_degree_values)
    degrees = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degrees]
    
    # Calculate cumulative distribution
    total = sum(counts)
    cumulative = np.cumsum(counts[::-1])[::-1] / total
    
    ax.loglog(degrees, cumulative, 'bo-', alpha=0.6, markersize=4)
    ax.set_xlabel('In-Degree (log scale)', fontsize=10)
    ax.set_ylabel('P(degree ≥ k) (log scale)', fontsize=10)
    ax.set_title('Cumulative In-Degree Distribution\n(log-log scale)', fontsize=11)
    ax.grid(True, alpha=0.3, which='both')


def plot_hub_analysis(stats, ax, top_n=20):
    """Identify and visualize hub neurons (high degree nodes)."""
    # Combine in-degree and out-degree
    all_neurons = set(stats['in_degrees'].keys()) | set(stats['out_degrees'].keys())
    total_degrees = {}
    
    for nid in all_neurons:
        in_deg = stats['in_degrees'].get(nid, 0)
        out_deg = stats['out_degrees'].get(nid, 0)
        total_degrees[nid] = in_deg + out_deg
    
    # Get top hubs
    sorted_neurons = sorted(total_degrees.items(), key=lambda x: x[1], reverse=True)
    top_hubs = sorted_neurons[:top_n]
    
    if len(top_hubs) == 0:
        ax.text(0.5, 0.5, 'No hub data', ha='center', va='center', transform=ax.transAxes)
        return
    
    neuron_ids = [str(nid) for nid, _ in top_hubs]
    degrees = [deg for _, deg in top_hubs]
    
    ax.barh(range(len(neuron_ids)), degrees, alpha=0.7, color='teal')
    ax.set_yticks(range(len(neuron_ids)))
    ax.set_yticklabels(neuron_ids, fontsize=8)
    ax.set_xlabel('Total Degree (In + Out)', fontsize=10)
    ax.set_ylabel('Neuron ID', fontsize=10)
    ax.set_title(f'Top {top_n} Hub Neurons', fontsize=11)
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()


def plot_connectivity_matrix_sample(connectivity, ax, sample_size=100):
    """Plot a sample of the connectivity matrix."""
    # Get sample of neurons
    all_neurons = sorted(connectivity.keys())
    if len(all_neurons) > sample_size:
        sample_neurons = np.random.choice(all_neurons, sample_size, replace=False)
        sample_neurons = sorted(sample_neurons)
    else:
        sample_neurons = all_neurons
    
    # Create adjacency matrix
    n = len(sample_neurons)
    neuron_to_idx = {nid: i for i, nid in enumerate(sample_neurons)}
    adj_matrix = np.zeros((n, n))
    
    for post_id in sample_neurons:
        if post_id in connectivity:
            for pre_id, strength, _ in connectivity[post_id]:
                if pre_id in neuron_to_idx:
                    i = neuron_to_idx[pre_id]
                    j = neuron_to_idx[post_id]
                    adj_matrix[i, j] = strength
    
    im = ax.imshow(adj_matrix, cmap='hot', aspect='auto', interpolation='nearest')
    ax.set_xlabel('Post-synaptic Neuron Index', fontsize=10)
    ax.set_ylabel('Pre-synaptic Neuron Index', fontsize=10)
    ax.set_title(f'Connectivity Matrix Sample\n({n} neurons)', fontsize=11)
    plt.colorbar(im, ax=ax, label='Connection Strength')


def plot_strength_percentiles(stats, ax):
    """Plot strength distribution with percentiles."""
    strengths = stats['strengths']
    
    if len(strengths) == 0:
        ax.text(0.5, 0.5, 'No strength data', ha='center', va='center', transform=ax.transAxes)
        return
    
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    percentile_values = np.percentile(strengths, percentiles)
    
    ax.bar(range(len(percentiles)), percentile_values, alpha=0.7, color='coral', edgecolor='black')
    ax.set_xticks(range(len(percentiles)))
    ax.set_xticklabels([f'{p}th' for p in percentiles])
    ax.set_xlabel('Percentile', fontsize=10)
    ax.set_ylabel('Connection Strength', fontsize=10)
    ax.set_title('Strength Percentiles', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, val in enumerate(percentile_values):
        ax.text(i, val, f'{val:.3f}', ha='center', va='bottom', fontsize=8)


def plot_network_motifs(connectivity, ax):
    """Analyze and plot simple network motifs (triangles, reciprocal connections)."""
    # Count different motif types
    motif_counts = {
        'Reciprocal': 0,
        'Convergent': 0,
        'Divergent': 0,
        'Chain': 0
    }
    
    # Build edge set for quick lookup
    edge_set = set()
    for post_id, connections in connectivity.items():
        for pre_id, _, _ in connections:
            edge_set.add((pre_id, post_id))
    
    # Count reciprocal connections
    for pre_id, post_id in edge_set:
        if (post_id, pre_id) in edge_set:
            motif_counts['Reciprocal'] += 1
    motif_counts['Reciprocal'] //= 2  # Each pair counted twice
    
    # Sample-based counting for other motifs (to avoid O(n^3) complexity)
    sample_size = min(1000, len(connectivity))
    sampled_neurons = np.random.choice(list(connectivity.keys()), sample_size, replace=False)
    
    for neuron in sampled_neurons:
        if neuron not in connectivity:
            continue
        
        # Get incoming connections
        incoming = [pre_id for pre_id, _, _ in connectivity[neuron]]
        
        # Convergent: multiple neurons connect to this one
        if len(incoming) >= 2:
            motif_counts['Convergent'] += 1
        
        # Divergent: this neuron connects to multiple others
        outgoing = [post_id for post_id in connectivity.keys() 
                   if any(pre_id == neuron for pre_id, _, _ in connectivity[post_id])]
        if len(outgoing) >= 2:
            motif_counts['Divergent'] += 1
    
    # Normalize by sample size
    scale_factor = len(connectivity) / sample_size
    motif_counts['Convergent'] = int(motif_counts['Convergent'] * scale_factor)
    motif_counts['Divergent'] = int(motif_counts['Divergent'] * scale_factor)
    
    # Plot
    motifs = list(motif_counts.keys())
    counts = list(motif_counts.values())
    
    ax.bar(motifs, counts, alpha=0.7, color=['red', 'blue', 'green', 'orange'], edgecolor='black')
    ax.set_xlabel('Motif Type', fontsize=10)
    ax.set_ylabel('Count (estimated)', fontsize=10)
    ax.set_title('Network Motif Analysis', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for i, val in enumerate(counts):
        ax.text(i, val, str(val), ha='center', va='bottom', fontsize=9)


def create_comprehensive_plots(connectivity, stats):
    """Create comprehensive visualization with multiple subplots."""
    print("Creating visualizations...")
    
    # Detect third parameter type
    param_type = detect_third_param_type(stats['third_params'])
    print(f"  Detected third parameter type: {param_type}")
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
    
    # Row 1: Degree distributions
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    plot_degree_distributions(stats, ax1, ax2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    plot_degree_correlation(stats, ax3)
    
    # Row 2: Strength and lag/position
    ax4 = fig.add_subplot(gs[1, 0])
    plot_strength_distribution(stats, ax4)
    
    ax5 = fig.add_subplot(gs[1, 1])
    if param_type == 'lag':
        plot_lag_distribution(stats['third_params'], ax5)
    elif param_type == 'position':
        plot_position_distribution(stats['third_params'], ax5)
    else:
        ax5.text(0.5, 0.5, 'Unknown parameter type', ha='center', va='center', transform=ax5.transAxes)
    
    ax6 = fig.add_subplot(gs[1, 2])
    plot_strength_percentiles(stats, ax6)
    
    # Row 3: Advanced analysis
    ax7 = fig.add_subplot(gs[2, 0])
    plot_cumulative_degree_distribution(stats, ax7)
    
    ax8 = fig.add_subplot(gs[2, 1])
    plot_hub_analysis(stats, ax8, top_n=20)
    
    ax9 = fig.add_subplot(gs[2, 2])
    plot_network_motifs(connectivity, ax9)
    
    # Row 4: Connectivity matrix and summary
    ax10 = fig.add_subplot(gs[3, 0])
    plot_connectivity_matrix_sample(connectivity, ax10, sample_size=100)
    
    # Summary statistics text
    ax11 = fig.add_subplot(gs[3, 1:])
    ax11.axis('off')
    
    summary_text = f"""
    CONNECTIVITY GRAPH SUMMARY STATISTICS
    ═══════════════════════════════════════════════════════════════
    
    Network Size:
      • Total neurons: {stats['num_neurons']:,}
      • Total connections: {stats['total_connections']:,}
      • Network density: {stats['density']:.6f}
      • Reciprocal connections: {stats['reciprocal_connections']:,}
    
    Degree Statistics:
      • In-degree:  mean={np.mean(list(stats['in_degrees'].values())):.2f}, 
                    std={np.std(list(stats['in_degrees'].values())):.2f}, 
                    max={max(stats['in_degrees'].values())}
      • Out-degree: mean={np.mean(list(stats['out_degrees'].values())):.2f}, 
                    std={np.std(list(stats['out_degrees'].values())):.2f}, 
                    max={max(stats['out_degrees'].values())}
    
    Connection Strength:
      • Mean: {np.mean(stats['strengths']):.4f}
      • Median: {np.median(stats['strengths']):.4f}
      • Std: {np.std(stats['strengths']):.4f}
      • Range: [{np.min(stats['strengths']):.4f}, {np.max(stats['strengths']):.4f}]
    
    Third Parameter ({param_type}):
      • Type: {param_type}
    """
    
    if param_type == 'lag':
        lags = np.array(stats['third_params'])
        summary_text += f"""  • Mean lag: {np.mean(lags):.2f} timesteps
      • Lag range: [{np.min(lags)}, {np.max(lags)}]
    """
    elif param_type == 'position':
        positions = np.array(stats['third_params'])
        distances = np.linalg.norm(positions, axis=1)
        summary_text += f"""  • Mean distance: {np.mean(distances):.2f}
      • Distance range: [{np.min(distances):.2f}, {np.max(distances):.2f}]
    """
    
    ax11.text(0.05, 0.95, summary_text, transform=ax11.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    fig.suptitle('Connectivity Graph Analysis', fontsize=18, fontweight='bold')
    
    print("  All plots created")
    return fig


def main():
    parser = argparse.ArgumentParser(
        description='Analyze and visualize connectivity graph statistics from pickle files.'
    )
    parser.add_argument('input_file', type=str,
                       help='Path to input pickle file containing connectivity graph')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output_dir', type=str, default='/tmp',
                       help='Directory to save plots (default: /tmp)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved plots (default: 300)')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Load connectivity
    connectivity = load_connectivity(args.input_file)
    
    # Extract statistics
    stats = extract_statistics(connectivity)
    
    # Create visualizations
    fig = create_comprehensive_plots(connectivity, stats)
    
    # Save plots
    import os
    base_name = os.path.basename(args.input_file).replace('.pkl', '')
    output_path = os.path.join(args.output_dir, f'connectivity_analysis_{base_name}.png')
    print(f"\nSaving plot to {output_path} at {args.dpi} DPI...")
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    print(f"Saved: {output_path}")
    
    plt.close(fig)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
