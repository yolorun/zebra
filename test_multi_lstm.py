#!/usr/bin/env python3
"""
Test script for the multi-LSTM architecture.
This script tests the updated model with individual LSTMs per neuron.
"""

import torch
import numpy as np
from all_neurons import AllNeuronsDataset, NeuronDataModule, LSTMForecaster

def test_dataset():
    """Test that the dataset returns neuron indices correctly."""
    print("Testing dataset...")
    
    # Create dummy data
    traces = np.random.randn(1000, 10)  # 1000 timesteps, 10 neurons
    embeddings = np.random.randn(10, 192)  # 10 neurons, 192 embedding dims
    stimulus = np.random.randn(1000, 10)  # 1000 timesteps, 10 stimulus channels
    
    dataset = AllNeuronsDataset(traces, embeddings, stimulus, sequence_length=50, prediction_horizon=1)
    
    # Test a few samples
    for i in range(5):
        (activity, embedding, stim, neuron_idx), target = dataset[i]
        expected_neuron = i // dataset.samples_per_neuron
        assert neuron_idx == expected_neuron, f"Expected neuron {expected_neuron}, got {neuron_idx}"
        print(f"  Sample {i}: Neuron index {neuron_idx} ✓")
    
    print("Dataset test passed! ✓\n")

def test_model_forward():
    """Test that the model forward pass works correctly."""
    print("Testing model forward pass...")
    
    # Create a small model configuration
    cfg = {
        'model_params': {
            'input_size': 203,
            'hidden_size': 32,  # Small hidden size for testing
            'num_layers': 1,
            'output_size': 1,
        },
        'optimizer_params': {'lr': 1e-3},
        'scheduler_params': {'factor': 0.5, 'patience': 3, 'min_lr': 1e-6},
    }
    
    num_neurons = 5
    batch_size = 8
    seq_length = 50
    
    # Create model
    model = LSTMForecaster(cfg, num_neurons=num_neurons)
    
    # Create dummy batch
    activity_seq = torch.randn(batch_size, seq_length, 1)
    embedding = torch.randn(batch_size, 192)
    stimulus_seq = torch.randn(batch_size, seq_length, 10)
    
    # Create neuron indices for the batch
    # Simulate that we have samples from different neurons
    neuron_indices = torch.tensor([0, 0, 1, 1, 2, 2, 3, 4])
    
    # Forward pass
    predictions = model(activity_seq, embedding, stimulus_seq, neuron_indices)
    
    # Check output shape
    assert predictions.shape == (batch_size, 1), f"Expected shape {(batch_size, 1)}, got {predictions.shape}"
    print(f"  Output shape: {predictions.shape} ✓")
    
    # Check that predictions are not all the same
    assert not torch.allclose(predictions[0], predictions[2]), "Different neurons should give different predictions"
    print(f"  Different neurons give different outputs ✓")
    
    print("Model forward pass test passed! ✓\n")

def test_training_step():
    """Test that the training step works correctly."""
    print("Testing training step...")
    
    cfg = {
        'model_params': {
            'input_size': 203,
            'hidden_size': 32,
            'num_layers': 1,
            'output_size': 1,
        },
        'optimizer_params': {'lr': 1e-3},
        'scheduler_params': {'factor': 0.5, 'patience': 3, 'min_lr': 1e-6},
    }
    
    num_neurons = 3
    model = LSTMForecaster(cfg, num_neurons=num_neurons)
    
    # Create a dummy batch
    batch_size = 6
    seq_length = 50
    
    activity_seq = torch.randn(batch_size, seq_length, 1)
    embedding = torch.randn(batch_size, 192)
    stimulus_seq = torch.randn(batch_size, seq_length, 10)
    neuron_indices = torch.tensor([0, 0, 1, 1, 2, 2])
    targets = torch.randn(batch_size, 1)
    
    batch = ((activity_seq, embedding, stimulus_seq, neuron_indices), targets)
    
    # Run training step
    loss = model.training_step(batch, 0)
    
    # Check that loss is a scalar tensor
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss should not be NaN"
    assert loss > 0, "Loss should be positive"
    
    print(f"  Loss: {loss.item():.4f} ✓")
    print("Training step test passed! ✓\n")

def main():
    print("=" * 50)
    print("Testing Multi-LSTM Architecture")
    print("=" * 50 + "\n")
    
    test_dataset()
    test_model_forward()
    test_training_step()
    
    print("=" * 50)
    print("All tests passed successfully! ✅")
    print("=" * 50)

if __name__ == "__main__":
    main()
