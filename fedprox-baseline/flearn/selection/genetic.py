import numpy as np

"""Genetic Algorithm based selector for federated client subsets.

Provides a single function `genetic_select` which performs a GA search over
subsets of clients of fixed cardinality k, maximizing a provided fitness.

Fitness used here (default): negative L2 distance between the mean gradient
of the selected subset and the global gradient (higher is better because we
return the negative distance).

The function is intentionally self-contained and deterministic when provided
an `np.random.RandomState` via the `rng` argument.
"""


def genetic_select(local_grads,
                   global_grad,
                   k,
                   samples=None,
                   selection_counts=None,
                   ga_params=None,
                   rng=None):
    """Run GA to select k clients whose mean gradient is closest to global_grad.

    Args:
        local_grads: np.ndarray, shape (n_clients, grad_dim)
        global_grad: np.ndarray, shape (grad_dim,)
        k: int, number of clients to select
        samples: optional np.ndarray shape (n_clients,) (unused by default)
        selection_counts: optional np.ndarray shape (n_clients,) used as penalty
        ga_params: dict with keys 'pop_size','num_gen','mutation_rate',
                   'crossover_rate','tournament_size','selection_penalty'
        rng: optional np.random.RandomState for deterministic behavior

    Returns:
        indices: np.ndarray of shape (k,) with selected client indices
        info: dict with keys 'dissimilarity' and 'best_fitness'
    """
    if rng is None:
        rng = np.random.RandomState()

    n = local_grads.shape[0]
    k = int(k)

    # trivial
    if k <= 0:
        return np.array([], dtype=int), {'dissimilarity': None, 'best_fitness': None}
    if k >= n:
        indices = np.arange(n)
        subset_grad = np.mean(local_grads, axis=0)
        dissimilarity = np.linalg.norm(subset_grad - global_grad)
        return indices, {'dissimilarity': dissimilarity, 'best_fitness': None}

    # GA hyperparameters with sensible defaults
    if ga_params is None:
        ga_params = {}
    pop_size = int(ga_params.get('pop_size', 50))
    generations = int(ga_params.get('num_gen', 20))
    mutation_rate = float(ga_params.get('mutation_rate', 0.1))
    crossover_rate = float(ga_params.get('crossover_rate', 0.9))
    lambda_info = float(ga_params.get('lambda_info', 0.5))
    selection_penalty = float(ga_params.get('selection_penalty', 0.0))

    # initialize population (include a single random baseline individual)
    population = []
    idx_random = rng.choice(n, k, replace=False)
    chrom_random = np.zeros(n, dtype=int)
    chrom_random[idx_random] = 1
    population.append(chrom_random)

    for _ in range(pop_size - 1):
        chrom = np.zeros(n, dtype=int)
        chrom[rng.choice(n, k, replace=False)] = 1
        population.append(chrom)

    def evaluate_fitness(chrom):
        indices = np.where(chrom == 1)[0]
        if len(indices) == 0:
            return -np.inf

        selected_grads = local_grads[indices]
        selected_samples = samples[indices] if samples is not None else None

        subset_avg = np.mean(selected_grads, axis=0)

        drift = np.linalg.norm(subset_avg - global_grad)
        magnitude = np.linalg.norm(subset_avg)

        penalty = 0.0
        if selection_counts is not None:
            penalty = selection_penalty * np.sum(selection_counts[indices])

        return -drift + (lambda_info * magnitude) - penalty

    # Evolution loop with simple elitism + tournament-like selection
    for _ in range(generations):
        scores = np.array([evaluate_fitness(c) for c in population])
        best_idx = np.argmax(scores)
        next_pop = [population[best_idx].copy()]

        while len(next_pop) < pop_size:
            # pick two parents via random tournament of size 2
            p1_idx = rng.randint(pop_size)
            p2_idx = rng.randint(pop_size)
            parent1 = population[p1_idx] if scores[p1_idx] > scores[p2_idx] else population[p2_idx]

            child = parent1.copy()
            if rng.rand() < crossover_rate:
                parent2 = population[rng.randint(pop_size)]
                mask = rng.rand(n) > 0.5
                child[mask] = parent2[mask]

            if rng.rand() < mutation_rate:
                ones = np.where(child == 1)[0]
                zeros = np.where(child == 0)[0]
                if len(ones) > 0 and len(zeros) > 0:
                    swap_out = rng.choice(ones)
                    swap_in = rng.choice(zeros)
                    child[swap_out] = 0
                    child[swap_in] = 1

            # repair to ensure exactly k ones
            current_k = child.sum()
            if current_k > k:
                remove_indices = rng.choice(np.where(child == 1)[0], int(current_k - k), replace=False)
                child[remove_indices] = 0
            elif current_k < k:
                add_indices = rng.choice(np.where(child == 0)[0], int(k - current_k), replace=False)
                child[add_indices] = 1

            next_pop.append(child)

        population = next_pop

    final_scores = np.array([evaluate_fitness(c) for c in population])
    best_chrom = population[np.argmax(final_scores)]
    indices = np.where(best_chrom == 1)[0]

    subset_grad = np.mean(local_grads[indices], axis=0)
    dissimilarity = np.linalg.norm(subset_grad - global_grad)
    best_f = float(final_scores.max())

    return indices, {'dissimilarity': dissimilarity, 'best_fitness': best_f}
