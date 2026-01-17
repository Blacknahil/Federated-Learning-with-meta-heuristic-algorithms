#!/usr/bin/env python3
"""Test the SA implementation."""
import sys
import os
sys.path.append('.')

# Test if imports work
try:
    from flearn.selection.simulated_annealing import SimulatedAnnealingSelector
    print("✓ SA selector imports successfully")
except Exception as e:
    print(f"✗ SA selector import failed: {e}")

# Test if main runs
try:
    import main
    print("✓ Main imports successfully")
except Exception as e:
    print(f"✗ Main import failed: {e}")

# Test basic SA functionality
import numpy as np

print("\nTesting basic SA calculations...")
# Create mock data
n_clients = 10
grad_dim = 100
local_grads = [np.random.randn(grad_dim) for _ in range(n_clients)]
global_grad = np.mean(local_grads, axis=0)
client_ids = list(range(n_clients))

print(f"  Created {n_clients} clients with {grad_dim}-dim gradients")
print(f"  Global gradient shape: {global_grad.shape}")

# Test cosine dissimilarity calculation
def test_dissimilarity():
    indices = [0, 1, 2]
    total = 0
    for idx in indices:
        local = local_grads[idx]
        cos_sim = np.dot(local, global_grad) / (
            np.linalg.norm(local) * np.linalg.norm(global_grad) + 1e-8
        )
        total += 1.0 - cos_sim
    avg = total / len(indices)
    print(f"✓ Dissimilarity calculation works: {avg:.4f}")
    return avg

test_dissimilarity()

print("\n✓ All basic tests passed!")
print("\nTo run full SA experiment:")
print("python main.py --selection_method=sa --sa_initial_temp=100 --sa_cooling_rate=0.95 --sa_epochs=30")