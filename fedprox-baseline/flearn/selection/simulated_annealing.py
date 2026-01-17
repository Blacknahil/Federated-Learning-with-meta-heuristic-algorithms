import numpy as np
import math


def _compute_dissimilarity(selected_idx, local_grads, global_grad):
    """Average per-client squared L2 dissimilarity between local and global gradients."""
    if selected_idx.size == 0:
        return 0.0
    sel = local_grads[selected_idx]
    # per-client squared norm of difference
    diffs = np.sum((sel - global_grad) ** 2, axis=1)
    return float(np.mean(diffs))


def simulated_annealing_select(local_grads, global_grad, k, samples=None,
                               selection_counts=None, sa_params=None, rng=None):
    """
    Simulated Annealing selector for client selection.

    Args:
        local_grads: np.array shape (n_clients, model_len)
        global_grad: np.array shape (model_len,)
        k: number of clients to select
        samples: not used currently
        selection_counts: optional historical selection counts
        sa_params: dict with optional keys: 'init_temp', 'cooling_rate', 'iterations', 'selection_penalty'
        rng: np.random.RandomState or similar

    Returns:
        indices: np.array of selected client indices
        info: dict { 'dissimilarity': float, 'best_cost': float }
    """
    if rng is None:
        rng = np.random.RandomState()

    n = int(local_grads.shape[0])
    k = int(min(k, n))

    if k <= 0:
        return np.array([], dtype=int), {'dissimilarity': 0.0, 'best_cost': 0.0}
    if k >= n:
        indices = np.arange(n)
        dis = _compute_dissimilarity(indices, local_grads, global_grad)
        return indices, {'dissimilarity': dis, 'best_cost': dis}

    # SA params and defaults
    params = {} if sa_params is None else dict(sa_params)
    T = float(params.get('init_temp', 1.0))
    cooling = float(params.get('cooling_rate', 0.995))
    iterations = int(params.get('iterations', 1000))
    selection_penalty = float(params.get('selection_penalty', 0.0))

    # helper to compute cost: lower is better
    def cost_of(indices):
        dis = _compute_dissimilarity(indices, local_grads, global_grad)
        # penalty term encourages fairness (higher if selected clients historically over-selected)
        penalty = 0.0
        if selection_counts is not None and selection_counts.size == n and selection_penalty > 0.0:
            penalty = selection_penalty * np.mean(selection_counts[indices])
        return dis + penalty

    # initialize with random subset
    current = np.array(rng.choice(np.arange(n), size=k, replace=False), dtype=int)
    current_cost = cost_of(current)
    best = current.copy()
    best_cost = current_cost

    for it in range(iterations):
        # propose neighbor by swapping one selected with one unselected
        sel_pos = rng.randint(0, k)
        unselected = np.setdiff1d(np.arange(n), current)
        if unselected.size == 0:
            break
        new_elem = rng.choice(unselected)
        candidate = current.copy()
        candidate[sel_pos] = new_elem

        candidate_cost = cost_of(candidate)

        # accept if better or with Metropolis probability
        if candidate_cost < current_cost:
            current = candidate
            current_cost = candidate_cost
        else:
            # probability of accepting worse solution
            if T > 0 and rng.rand() < math.exp((current_cost - candidate_cost) / T):
                current = candidate
                current_cost = candidate_cost

        # update best
        if current_cost < best_cost:
            best = current.copy()
            best_cost = current_cost

        # cool down
        T *= cooling

    # finalize
    best_indices = np.unique(best)
    # in rare cases of duplicates, ensure size k by sampling missing
    if best_indices.size < k:
        missing = k - best_indices.size
        pool = np.setdiff1d(np.arange(n), best_indices)
        if pool.size > 0:
            extra = rng.choice(pool, size=min(missing, pool.size), replace=False)
            best_indices = np.concatenate([best_indices, extra])

    dissimilarity = _compute_dissimilarity(best_indices, local_grads, global_grad)
    info = {'dissimilarity': dissimilarity, 'best_cost': float(best_cost), 'iterations': iterations}
    return np.sort(best_indices), info
