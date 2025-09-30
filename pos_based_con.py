#!/usr/bin/env python

"""
Extracts neuron positions, computes a distance-based connectivity graph, 
and visualizes the result.
"""

import argparse
import os
import pickle
import numpy as np
import tensorstore as ts
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from scipy.spatial.distance import cdist

def get_centroids(segmentation_data):
    """Calculates the centroid for each neuron using a localized search."""
    print("Calculating centroids with localized search...")
    positions = {}
    shape = segmentation_data.shape

    # Get unique labels and the flat index of their first occurrence
    labels, first_indices = np.unique(segmentation_data, return_index=True)

    # Convert flat indices to 3D coordinates
    first_coords = np.unravel_index(first_indices, shape)
    first_coords_map = {label: (z, y, x) for label, z, y, x in zip(labels, first_coords[0], first_coords[1], first_coords[2])}

    # Define the search window size (Z, Y, X)
    slice_size = np.array([5, 15, 15])
    slice_half = slice_size // 2

    # Exclude background label 0
    for neuron_id in tqdm(labels[labels != 0], desc="Processing neurons"):
        center_coord = np.array(first_coords_map[neuron_id])

        # Define the local search slice, clamping to volume boundaries
        z_start, y_start, x_start = np.maximum(0, center_coord - slice_half)
        z_end, y_end, x_end = np.minimum(shape, center_coord + slice_half + 1)
        
        local_slice = np.s_[z_start:z_end, y_start:y_end, x_start:x_end]
        local_volume = segmentation_data[local_slice]

        # Find coordinates of the neuron's voxels within the local volume
        local_indices = np.argwhere(local_volume == neuron_id)

        if local_indices.size > 0:
            # Offset local indices to get absolute coordinates
            absolute_indices = local_indices + [z_start, y_start, x_start]
            # Calculate the mean to get the centroid
            centroid = np.mean(absolute_indices, axis=0)
            positions[neuron_id] = tuple(centroid)

    return positions

from scipy.spatial.distance import cdist

def build_connectivity_graph(positions, distance_threshold, chunk_size):
    """Builds a connectivity graph using chunked, vectorized distance calculation to save memory."""
    print(f"Building connectivity graph with threshold: {distance_threshold} using chunked method...")
    
    neuron_ids = list(positions.keys())
    pos_array = np.array([positions[nid] for nid in neuron_ids])
    num_neurons = len(neuron_ids)
    connectivity = {nid: [] for nid in neuron_ids}

    for i in tqdm(range(0, num_neurons, chunk_size), desc="Processing chunks"):
        chunk_start = i
        chunk_end = min(i + chunk_size, num_neurons)
        chunk_pos = pos_array[chunk_start:chunk_end]

        # Compute distances from the current chunk to all other neurons
        dist_chunk = cdist(chunk_pos, pos_array, 'euclidean')

        # Find connections within the threshold
        source_indices, target_indices = np.where(
            (dist_chunk > 0) & (dist_chunk <= distance_threshold)
        )

        # Add connections to the dictionary
        for src_local_idx, tgt_idx in zip(source_indices, target_indices):
            # Convert local chunk index to global index
            src_global_idx = chunk_start + src_local_idx
            
            id1 = neuron_ids[src_global_idx]
            id2 = neuron_ids[tgt_idx]
            dist = dist_chunk[src_local_idx, tgt_idx]
            
            connectivity[id1].append((id2, dist))
            
    return connectivity

def visualize_graph(positions, connectivity):
    """Visualizes the neuron connectivity graph using NetworkX (2D projection)."""
    print("Visualizing graph with NetworkX...")
    
    G = nx.Graph()
    
    # Create a layout dictionary for node positions (X, Y)
    # The stored positions are (Z, Y, X), so we take indices 2 and 1.
    pos = {neuron_id: (coords[2], coords[1]) for neuron_id, coords in positions.items()}
    
    # Add nodes and edges from the connectivity data
    for neuron_id, neighbors in connectivity.items():
        G.add_node(neuron_id)
        for neighbor_id, _ in neighbors:
            G.add_edge(neuron_id, neighbor_id)
            
    plt.figure(figsize=(15, 15))
    nx.draw(
        G, 
        pos, 
        with_labels=False, 
        node_size=10, 
        width=0.5, 
        node_color='blue', 
        edge_color='red', 
        alpha=0.6
    )
    plt.title('Neuron Connectivity Graph (2D Top-Down View)')
    plt.savefig('neuron_connectivity_graph.png')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Generate and visualize a position-based connectivity graph.')
    parser.add_argument('--segmentation_path', type=str, default='/home/v/proj/zebra/data/segmentation',
                        help='Path to the segmentation mask zarr file.')
    parser.add_argument('--distance_threshold', type=float, default=50.0,
                        help='Maximum distance for two neurons to be considered connected.')
    args = parser.parse_args()

    # --- Load Data ---
    print(f"Loading segmentation data from: {args.segmentation_path}")
    ds_seg = ts.open({
        'driver': 'zarr3',
        'kvstore': f'file://{args.segmentation_path}'
    }).result()
    segmentation_data = ds_seg.read().result()

    # --- Calculate Centroids and Save ---
    positions = get_centroids(segmentation_data)
    with open('neuron_positions.pkl', 'wb') as f:
        pickle.dump(positions, f)
    print(f"Saved {len(positions)} neuron positions to neuron_positions.pkl")

    # --- Build Connectivity Graph and Save ---
    connectivity = build_connectivity_graph(positions, args.distance_threshold, chunk_size=2000)
    with open('connectivity_graph.pkl', 'wb') as f:
        pickle.dump(connectivity, f)
    print(f"Saved connectivity graph to connectivity_graph.pkl")

    # --- Visualize ---
    visualize_graph(positions, connectivity)

if __name__ == '__main__':
    main()
