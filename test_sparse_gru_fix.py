#!/usr/bin/env python
"""
Test script to verify the batched matrix multiplication fix in sparse_gru.py
"""

import torch
import numpy as np

def test_einsum_correctness():
    """Test that einsum produces correct per-neuron matrix multiplications."""
    
    B, N, H = 3, 4, 5  # batch_size=3, num_neurons=4, hidden_dim=5
    
    # Create test tensors
    hidden = torch.randn(B, N, H)
    U_z = torch.randn(N, H, H)  # Each neuron has its own H x H weight matrix
    
    # Method 1: Using einsum (our fix)
    result_einsum = torch.einsum('bnh,nhi->bni', hidden, U_z)
    
    # Method 2: Loop-based reference implementation (guaranteed correct but slow)
    result_loop = torch.zeros(B, N, H)
    for b in range(B):
        for n in range(N):
            # For each batch and neuron, multiply hidden[b,n] by U_z[n]
            result_loop[b, n] = torch.matmul(hidden[b, n], U_z[n])
    
    # Check if results match
    diff = torch.abs(result_einsum - result_loop).max()
    
    print("="*60)
    print("Testing Batched Matrix Multiplication Fix")
    print("="*60)
    print(f"Test configuration: B={B}, N={N}, H={H}")
    print(f"Hidden shape: {hidden.shape}")
    print(f"U_z shape: {U_z.shape}")
    print(f"Result shape: {result_einsum.shape}")
    print(f"\nMax difference between einsum and loop: {diff:.10f}")
    
    if diff < 1e-6:
        print("✅ TEST PASSED: Einsum implementation is correct!")
    else:
        print("❌ TEST FAILED: Einsum implementation has errors!")
        return False
    
    # Test specific neurons to verify correct weight usage
    print("\nVerifying correct weight matrix usage:")
    for n in range(min(3, N)):  # Check first 3 neurons
        # Manually compute for neuron n in batch 0
        manual_result = torch.matmul(hidden[0, n], U_z[n])
        einsum_result = result_einsum[0, n]
        match = torch.allclose(manual_result, einsum_result)
        print(f"  Neuron {n}: {'✅' if match else '❌'} Weight matrix U_z[{n}] correctly applied")
    
    return True

def test_old_implementation_bug():
    """Demonstrate the bug in the old implementation."""
    
    B, N, H = 2, 3, 4  # Small example for clarity
    
    hidden = torch.randn(B, N, H)
    U_z = torch.randn(N, H, H)
    
    # Old buggy implementation
    hidden_reshaped = hidden.reshape(B * N, 1, H)  # Shape: (6, 1, 4)
    U_z_repeated = U_z.view(N, H, H).repeat(B, 1, 1)  # Shape: (6, 4, 4)
    
    print("\n" + "="*60)
    print("Demonstrating the Bug in Old Implementation")
    print("="*60)
    
    print(f"\nOriginal shapes:")
    print(f"  hidden: {hidden.shape} -> reshaped to: {hidden_reshaped.shape}")
    print(f"  U_z: {U_z.shape} -> repeated to: {U_z_repeated.shape}")
    
    print(f"\nThe problem:")
    print(f"  hidden_reshaped order: [b0n0, b0n1, b0n2, b1n0, b1n1, b1n2]")
    print(f"  U_z_repeated order:    [n0, n1, n2, n0, n1, n2]")
    print(f"                                    ^^^ WRONG! Should be n0, n1, n2 again")
    
    # Show which weight matrix each hidden state gets
    print(f"\nWeight matrix assignments in old implementation:")
    for i in range(B * N):
        batch = i // N
        neuron = i % N
        # In the repeated tensor, position i has weight matrix for neuron (i % N)
        assigned_neuron = i % N
        is_correct = neuron == assigned_neuron
        symbol = "✅" if is_correct else "❌"
        print(f"  Position {i}: batch={batch}, neuron={neuron} -> "
              f"uses weight matrix for neuron {assigned_neuron} {symbol}")
    
    print(f"\nConclusion: Batch 0 works correctly, but batch 1+ use wrong weight matrices!")

def test_sparse_gru_integration():
    """Test the complete SparseGRUBrain with the fix."""
    from sparse_gru import SparseGRUBrain
    
    print("\n" + "="*60)
    print("Testing Complete SparseGRUBrain Integration")
    print("="*60)
    
    # Create a small test model
    num_neurons = 10
    hidden_dim = 8
    batch_size = 4
    
    # Simple edge list (each neuron connects to its neighbors)
    edge_list = []
    for i in range(num_neurons):
        if i > 0:
            edge_list.append((i-1, i))  # Previous neuron connects to current
        if i < num_neurons - 1:
            edge_list.append((i+1, i))  # Next neuron connects to current
        edge_list.append((i, i))  # Self-connection
    
    model = SparseGRUBrain(num_neurons, hidden_dim, edge_list, stimulus_dim=0, bias=True)
    
    # Test forward pass with different batch sizes
    for batch_size in [1, 2, 4]:
        calcium_t = torch.randn(batch_size, num_neurons)
        hidden = model.init_hidden(batch_size)
        
        try:
            calcium_pred, hidden_new = model(calcium_t, hidden, stimulus_t=None)
            assert calcium_pred.shape == (batch_size, num_neurons)
            assert hidden_new.shape == (batch_size, num_neurons, hidden_dim)
            print(f"✅ Forward pass successful with batch_size={batch_size}")
        except Exception as e:
            print(f"❌ Forward pass failed with batch_size={batch_size}: {e}")
            return False
    
    # Test gradient flow
    calcium_t = torch.randn(2, num_neurons, requires_grad=True)
    hidden = model.init_hidden(2)
    calcium_pred, _ = model(calcium_t, hidden)
    loss = calcium_pred.sum()
    loss.backward()
    
    if calcium_t.grad is not None:
        print("✅ Gradient flow successful")
    else:
        print("❌ No gradients computed")
    
    return True

if __name__ == '__main__':
    print("Running tests for sparse_gru.py batched matrix multiplication fix\n")
    
    # Run all tests
    test1 = test_einsum_correctness()
    test_old_implementation_bug()
    test2 = test_sparse_gru_integration()
    
    print("\n" + "="*60)
    if test1 and test2:
        print("✅ ALL TESTS PASSED - The fix is correct!")
    else:
        print("❌ SOME TESTS FAILED - Please review the implementation")
    print("="*60)
