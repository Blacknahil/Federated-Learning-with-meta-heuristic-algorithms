import numpy as np
import time

"""Particle Swarm Optimization based selector for federated client subsets.

Provides a single function `pso_select` which performs a PSO search over
subsets of clients of fixed cardinality k, maximizing a provided fitness.

Fitness used here (default): negative L2 distance between the mean gradient
of the selected subset and the global gradient (higher is better because we
return the negative distance).

The function is intentionally self-contained and deterministic when provided
an `np.random.RandomState` via the `rng` argument.

Binary PSO Implementation:
- Particles represent binary selection vectors (1 = selected, 0 = not selected)
- Velocity represents probability of flipping bits
- Sigmoid function converts velocity to probability
- Repair function ensures exactly k clients are selected
"""


def sigmoid(x):
    """Sigmoid activation function for binary PSO."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def pso_select(local_grads,
               global_grad,
               k,
               samples=None,
               selection_counts=None,
               pso_params=None,
               rng=None):
    """Run PSO to select k clients whose mean gradient is closest to global_grad.

    Args:
        local_grads: np.ndarray, shape (n_clients, grad_dim)
        global_grad: np.ndarray, shape (grad_dim,)
        k: int, number of clients to select
        samples: optional np.ndarray shape (n_clients,) (unused by default)
        selection_counts: optional np.ndarray shape (n_clients,) used as penalty
        pso_params: dict with keys 'swarm_size', 'num_iter', 'w', 'c1', 'c2',
                   'selection_penalty'
        rng: optional np.random.RandomState for deterministic behavior

    Returns:
        indices: np.ndarray of shape (k,) with selected client indices
        info: dict with keys 'dissimilarity', 'best_fitness', 'selection_time'
    """
    start_time = time.time()
    
    if rng is None:
        rng = np.random.RandomState()

    n = local_grads.shape[0]
    k = int(k)

    # Trivial cases
    if k <= 0:
        return np.array([], dtype=int), {
            'dissimilarity': None, 
            'best_fitness': None,
            'selection_time': time.time() - start_time
        }
    if k >= n:
        indices = np.arange(n)
        subset_grad = np.mean(local_grads, axis=0)
        dissimilarity = np.linalg.norm(subset_grad - global_grad)
        return indices, {
            'dissimilarity': dissimilarity, 
            'best_fitness': None,
            'selection_time': time.time() - start_time
        }

    # Default PSO parameters
    if pso_params is None:
        pso_params = {}
    swarm_size = int(pso_params.get('swarm_size', min(50, max(10, 5 * k))))
    num_iter = int(pso_params.get('num_iter', 20))
    w = float(pso_params.get('w', 0.7))  # Inertia weight
    c1 = float(pso_params.get('c1', 1.5))  # Cognitive coefficient (personal best)
    c2 = float(pso_params.get('c2', 1.5))  # Social coefficient (global best)
    selection_penalty = float(pso_params.get('selection_penalty', 0.0))
    v_max = float(pso_params.get('v_max', 4.0))  # Velocity clamp

    def fitness(particle):
        """Calculate fitness of a particle (binary selection vector)."""
        idx = np.where(particle == 1)[0]
        if idx.size == 0:
            return -1e9
        subset_grad = np.mean(local_grads[idx], axis=0)
        dissimilarity = np.linalg.norm(subset_grad - global_grad)
        penalty = 0.0
        if selection_counts is not None:
            penalty = selection_penalty * np.sum(selection_counts[idx])
        return -dissimilarity - penalty

    def repair(particle):
        """Repair particle to ensure exactly k ones."""
        particle = particle.copy()
        s = particle.sum()
        if s > k:
            ones_idx = np.where(particle == 1)[0]
            drop = rng.choice(ones_idx, int(s - k), replace=False)
            particle[drop] = 0
        elif s < k:
            zeros_idx = np.where(particle == 0)[0]
            add = rng.choice(zeros_idx, int(k - s), replace=False)
            particle[add] = 1
        return particle

    # Initialize swarm: positions (binary) and velocities (continuous)
    positions = np.zeros((swarm_size, n), dtype=int)
    velocities = rng.uniform(-1, 1, (swarm_size, n))
    
    for i in range(swarm_size):
        ones = rng.choice(n, k, replace=False)
        positions[i, ones] = 1

    # Personal best positions and fitness
    personal_best_pos = positions.copy()
    personal_best_fit = np.array([fitness(p) for p in positions])

    # Global best
    global_best_idx = np.argmax(personal_best_fit)
    global_best_pos = personal_best_pos[global_best_idx].copy()
    global_best_fit = personal_best_fit[global_best_idx]

    # PSO main loop
    for iteration in range(num_iter):
        for i in range(swarm_size):
            r1 = rng.rand(n)
            r2 = rng.rand(n)

            # Update velocity (standard PSO equation)
            velocities[i] = (w * velocities[i] +
                           c1 * r1 * (personal_best_pos[i] - positions[i]) +
                           c2 * r2 * (global_best_pos - positions[i]))
            
            # Clamp velocity
            velocities[i] = np.clip(velocities[i], -v_max, v_max)

            # Binary PSO: use sigmoid to get selection probability
            prob = sigmoid(velocities[i])
            new_position = (rng.rand(n) < prob).astype(int)
            
            # Repair to ensure exactly k clients selected
            new_position = repair(new_position)
            positions[i] = new_position

            # Evaluate fitness
            fit = fitness(positions[i])

            # Update personal best
            if fit > personal_best_fit[i]:
                personal_best_fit[i] = fit
                personal_best_pos[i] = positions[i].copy()

                # Update global best
                if fit > global_best_fit:
                    global_best_fit = fit
                    global_best_pos = positions[i].copy()

        # Adaptive inertia weight decay (optional enhancement)
        w = max(0.4, w * 0.99)

    # Extract final result
    indices = np.where(global_best_pos == 1)[0]
    subset_grad = np.mean(local_grads[indices], axis=0)
    dissimilarity = np.linalg.norm(subset_grad - global_grad)
    selection_time = time.time() - start_time

    return indices, {
        'dissimilarity': dissimilarity,
        'best_fitness': global_best_fit,
        'selection_time': selection_time
    }
