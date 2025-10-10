#!/usr/bin/env python
"""
Test script for massive RNN training.
This script tests the implementation with a small subset of data.
"""

import os
import pickle
import numpy as np
import torch
from sparse_gru import SparseGRUBrain

def test_connectivity_loading():
    """Test that we can load and process connectivity graphs."""
    print("Testing connectivity graph loading...")
    
    # Check which connectivity graph exists
    if os.path.exists('connectivity_graph_pos.pkl'):
        with open('connectivity_graph_pos.pkl', 'rb') as f:
            conn_graph = pickle.load(f)
        print(f"✓ Loaded position-based connectivity graph")
    elif os.path.exists('connectivity_graph_global_threshold.pkl'):
        with open('connectivity_graph_global_threshold.pkl', 'rb') as f:
            conn_graph = pickle.load(f)
        print(f"✓ Loaded cross-correlation connectivity graph")
    else:
        print("✗ No connectivity graph found. Please run pos_based_con.py or cross_corr3.py first.")
        return None
    
    # Print graph statistics
    n_neurons = len(conn_graph)
    total_connections = sum(len(v) for v in conn_graph.values())
    print(f"  Graph has {n_neurons} neurons with {total_connections} total connections")
    
    return conn_graph

def test_sparse_gru_creation():
    """Test creating a sparse GRU from connectivity graph."""
    print("\nTesting SparseGRUBrain creation...")
    
    # Create a small test connectivity graph
    test_graph = {
        1: [(2, 0.8), (3, 0.6)],
        2: [(1, 0.7), (3, 0.5)],
        3: [(1, 0.9), (2, 0.4)]
    }
    
    # Create model
    model = SparseGRUBrain.from_connectivity_graph(
        test_graph,
        hidden_dim=16,
        stimulus_dim=10,
        include_self_connections=True,
        min_strength=0.0,
        bias=True,
        num_neurons=3,
        assume_zero_indexed=False  # Test graph uses 1-indexed
    )
    
    print(f"✓ Created model with {model.num_neurons} neurons")
    
    # Test forward pass
    batch_size = 2
    calcium = torch.randn(batch_size, 3)
    stimulus = torch.randn(batch_size, 10)
    hidden = model.init_hidden(batch_size)
    
    calcium_pred, hidden_new = model(calcium, hidden, stimulus)
    
    assert calcium_pred.shape == (batch_size, 3)
    assert hidden_new.shape == (batch_size, 3, 16)
    print(f"✓ Forward pass successful")
    
    return model

def test_data_loading():
    """Test that data paths exist."""
    print("\nTesting data paths...")
    
    traces_path = 'file:///home/v/proj/zebra/data/traces'
    stimulus_path = '/home/v/proj/zebra/data/stimuli_raw/stimuli_and_ephys.10chFlt'
    
    # Check traces (remove file:// prefix)
    traces_local = traces_path.replace('file://', '')
    if os.path.exists(traces_local):
        print(f"✓ Traces path exists: {traces_local}")
    else:
        print(f"✗ Traces path not found: {traces_local}")
    
    # Check stimulus
    if os.path.exists(stimulus_path):
        print(f"✓ Stimulus path exists: {stimulus_path}")
    else:
        print(f"✗ Stimulus path not found: {stimulus_path}")

def test_training_script():
    """Test that the training script can be imported."""
    print("\nTesting massive_rnn_train.py import...")
    
    try:
        import massive_rnn_train
        print("✓ massive_rnn_train.py imported successfully")
        
        # Check main components
        assert hasattr(massive_rnn_train, 'CalciumTracesDataset')
        assert hasattr(massive_rnn_train, 'CalciumDataModule')
        assert hasattr(massive_rnn_train, 'MassiveRNNModule')
        assert hasattr(massive_rnn_train, 'main')
        print("✓ All main components found")
        
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False
    
    return True

def main():
    print("="*60)
    print("Testing Massive RNN Training Implementation")
    print("="*60)
    
    # Run tests
    conn_graph = test_connectivity_loading()
    model = test_sparse_gru_creation()
    test_data_loading()
    success = test_training_script()
    
    print("\n" + "="*60)
    if conn_graph and model and success:
        print("✓ All tests passed! Ready to train.")
        print("\nTo start training, run:")
        print("  python massive_rnn_train.py")
        print("\nRecommended first steps:")
        print("1. Start with max_neurons=100 to test")
        print("2. Monitor memory usage and training speed")
        print("3. Gradually increase max_neurons")
        print("4. Set max_neurons=None for all neurons")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    print("="*60)

if __name__ == '__main__':
    main()
