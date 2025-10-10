#!/usr/bin/env python

"""
Generates a random connectivity graph for neurons.
Each neuron receives a random number of incoming connections (uniformly sampled 
between min_connections and max_connections), with pre-synaptic neurons randomly 
selected from all available neurons.
"""

import argparse
import pickle
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


def generate_random_connectivity(num_neurons, min_connections, max_connections, seed=42):
    """
    Generate a random connectivity graph.
    
    Args:
        num_neurons: Total number of neurons in the network
        min_connections: Minimum number of incoming connections per neuron
        max_connections: Maximum number of incoming connections per neuron
        seed: Random seed for reproducibility
    
    Returns:
        connectivity: dict mapping post_synaptic_id -> list of (pre_synaptic_id, strength, lag) tuples
    """
    np.random.seed(seed)
    
    print(f"Generating random connectivity for {num_neurons} neurons...")
    print(f"Connections per neuron: [{min_connections}, {max_connections}]")
    
    # Initialize connectivity dictionary (1-indexed to match existing scripts)
    connectivity = {i: [] for i in range(1, num_neurons + 1)}
    
    # For each neuron, randomly sample number of connections and pre-synaptic neurons
    for post_synaptic_id in tqdm(range(1, num_neurons + 1), desc="Generating connections"):
        # Uniformly sample number of connections
        n_connections = np.random.randint(min_connections, max_connections + 1)
        
        # Ensure we don't try to sample more neurons than available (excluding self)
        n_connections = min(n_connections, num_neurons - 1)
        
        if n_connections > 0:
            # Get all possible pre-synaptic neurons (excluding self)
            possible_pre_synaptic = [i for i in range(1, num_neurons + 1) if i != post_synaptic_id]
            
            # Randomly sample pre-synaptic neurons without replacement
            pre_synaptic_ids = np.random.choice(
                possible_pre_synaptic, 
                size=n_connections, 
                replace=False
            )
            
            # Add connections with random strengths and lags
            for pre_synaptic_id in pre_synaptic_ids:
                # Random strength between 0.1 and 1.0
                strength = np.random.uniform(0.1, 1.0)
                
                # Random lag between 1 and 3 timesteps (positive lag means pre leads post)
                lag = np.random.randint(1, 4)
                
                connectivity[post_synaptic_id].append((int(pre_synaptic_id), strength, lag))
    
    # Calculate statistics
    total_connections = sum(len(v) for v in connectivity.values())
    avg_connections = total_connections / num_neurons
    
    print(f"\nGenerated {total_connections} total connections")
    print(f"Average connections per neuron: {avg_connections:.2f}")
    
    return connectivity


def visualize_connectivity_stats(connectivity, output_prefix):
    """
    Visualize statistics of the connectivity graph.
    
    Args:
        connectivity: dict mapping post_synaptic_id -> list of connections
        output_prefix: prefix for output files
    """
    print("Generating visualization...")
    
    # Calculate in-degree and out-degree for each neuron
    in_degrees = {nid: len(connections) for nid, connections in connectivity.items()}
    
    # Calculate out-degrees
    out_degrees = {nid: 0 for nid in connectivity.keys()}
    for post_synaptic_id, connections in connectivity.items():
        for pre_synaptic_id, _, _ in connections:
            out_degrees[pre_synaptic_id] += 1
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Random Connectivity Graph Statistics', fontsize=16)
    
    # Plot in-degree distribution
    in_degree_values = list(in_degrees.values())
    axes[0, 0].hist(in_degree_values, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].set_xlabel('In-Degree (Incoming Connections)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title(f'In-Degree Distribution (mean={np.mean(in_degree_values):.2f})')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot out-degree distribution
    out_degree_values = list(out_degrees.values())
    axes[0, 1].hist(out_degree_values, bins=30, alpha=0.7, color='green', edgecolor='black')
    axes[0, 1].set_xlabel('Out-Degree (Outgoing Connections)')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title(f'Out-Degree Distribution (mean={np.mean(out_degree_values):.2f})')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot strength distribution
    strengths = []
    for connections in connectivity.values():
        for _, strength, _ in connections:
            strengths.append(strength)
    
    axes[1, 0].hist(strengths, bins=30, alpha=0.7, color='red', edgecolor='black')
    axes[1, 0].set_xlabel('Connection Strength')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title(f'Strength Distribution (mean={np.mean(strengths):.2f})')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot lag distribution
    lags = []
    for connections in connectivity.values():
        for _, _, lag in connections:
            lags.append(lag)
    
    axes[1, 1].hist(lags, bins=range(min(lags), max(lags) + 2), alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_xlabel('Time Lag (timesteps)')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_title(f'Lag Distribution (mean={np.mean(lags):.2f})')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_stats.png', dpi=150)
    plt.close()
    
    print(f"Saved statistics plot to {output_prefix}_stats.png")


def visualize_network_graph(connectivity, output_prefix, max_nodes=100):
    """
    Visualize the network graph structure (limited to first max_nodes for clarity).
    
    Args:
        connectivity: dict mapping post_synaptic_id -> list of connections
        output_prefix: prefix for output files
        max_nodes: maximum number of nodes to visualize
    """
    print(f"Generating network graph visualization (limited to {max_nodes} nodes)...")
    
    # Create directed graph
    G = nx.DiGraph()
    
    # Add edges (limit to first max_nodes)
    node_ids = sorted(connectivity.keys())[:max_nodes]
    for post_synaptic_id in node_ids:
        connections = connectivity[post_synaptic_id]
        for pre_synaptic_id, strength, _ in connections:
            if pre_synaptic_id in node_ids:  # Only include if both nodes are in subset
                G.add_edge(pre_synaptic_id, post_synaptic_id, weight=strength)
    
    if not G.nodes():
        print("No connections to visualize in the subset.")
        return
    
    # Create layout
    plt.figure(figsize=(14, 14))
    pos = nx.spring_layout(G, seed=42, k=0.5, iterations=50)
    
    # Draw network
    nx.draw_networkx_nodes(G, pos, node_size=100, node_color='lightblue', alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=0.5, alpha=0.3, arrows=True, 
                           arrowsize=10, arrowstyle='->', edge_color='gray')
    
    # Add labels for small networks
    if len(G.nodes()) <= 50:
        nx.draw_networkx_labels(G, pos, font_size=8)
    
    plt.title(f'Random Connectivity Network Graph (first {len(G.nodes())} neurons)')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_network.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved network graph to {output_prefix}_network.png")


def main():
    parser = argparse.ArgumentParser(
        description='Generate a random connectivity graph for neural network training.'
    )
    parser.add_argument('--num_neurons', type=int, default=71721,
                        help='Total number of neurons in the network (default: 71721)')
    parser.add_argument('--min_connections', type=int, default=5,
                        help='Minimum number of incoming connections per neuron (default: 5)')
    parser.add_argument('--max_connections', type=int, default=20,
                        help='Maximum number of incoming connections per neuron (default: 20)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--output', type=str, default='connectivity_graph_random.pkl',
                        help='Output pickle file path (default: connectivity_graph_random.pkl)')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.min_connections < 0:
        raise ValueError("min_connections must be non-negative")
    if args.max_connections < args.min_connections:
        raise ValueError("max_connections must be >= min_connections")
    if args.num_neurons < 2:
        raise ValueError("num_neurons must be at least 2")
    
    # Generate connectivity
    connectivity = generate_random_connectivity(
        num_neurons=args.num_neurons,
        min_connections=args.min_connections,
        max_connections=args.max_connections,
        seed=args.seed
    )
    
    # Save to pickle file
    print(f"\nSaving connectivity graph to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(connectivity, f)
    print(f"Successfully saved connectivity graph with {len(connectivity)} neurons")
    
    # Generate visualizations if requested
    if args.visualize:
        output_prefix = args.output.replace('.pkl', '')
        visualize_connectivity_stats(connectivity, output_prefix)
        visualize_network_graph(connectivity, output_prefix)
    
    print("\nDone!")


if __name__ == '__main__':
    main()
