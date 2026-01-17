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

    # defaults
    if ga_params is None:
        ga_params = {}
    pop_size = int(ga_params.get('pop_size', min(50, max(10, 5 * k))))
    generations = int(ga_params.get('num_gen', 15))
    mutation_rate = float(ga_params.get('mutation_rate', 0.1))
    crossover_rate = float(ga_params.get('crossover_rate', 0.9))
    tournament_size = int(ga_params.get('tournament_size', 3))
    selection_penalty = float(ga_params.get('selection_penalty', 0.0))

    # population: binary vectors with exactly k ones
    pop = np.zeros((pop_size, n), dtype=int)
    for i in range(pop_size):
        ones = rng.choice(n, k, replace=False)
        pop[i, ones] = 1

    def fitness(ind):
        idx = np.where(ind == 1)[0]
        if idx.size == 0:
            return -1e9
        subset_grad = np.mean(local_grads[idx], axis=0)
        dissimilarity = np.linalg.norm(subset_grad - global_grad)
        penalty = 0.0
        if selection_counts is not None:
            penalty = selection_penalty * np.sum(selection_counts[idx])
        return -dissimilarity - penalty

    fitness_vals = np.array([fitness(ind) for ind in pop])
    best = pop[np.argmax(fitness_vals)].copy()

    for _ in range(generations):
        new_pop = np.zeros_like(pop)
        new_pop[0] = best  # elitism

        for i in range(1, pop_size, 2):
            # tournament selection
            ids = rng.choice(pop_size, tournament_size, replace=False)
            parent1 = pop[ids[np.argmax(fitness_vals[ids])]]
            ids = rng.choice(pop_size, tournament_size, replace=False)
            parent2 = pop[ids[np.argmax(fitness_vals[ids])]]

            # crossover
            if rng.rand() < crossover_rate:
                cp = rng.randint(1, n)
                child1 = np.concatenate([parent1[:cp], parent2[cp:]])
                child2 = np.concatenate([parent2[:cp], parent1[cp:]])
            else:
                child1 = parent1.copy()
                child2 = parent2.copy()

            # repair
            def repair(child):
                s = child.sum()
                if s > k:
                    ones_idx = np.where(child == 1)[0]
                    drop = rng.choice(ones_idx, int(s - k), replace=False)
                    child[drop] = 0
                elif s < k:
                    zeros_idx = np.where(child == 0)[0]
                    add = rng.choice(zeros_idx, int(k - s), replace=False)
                    child[add] = 1
                return child

            child1 = repair(child1)
            child2 = repair(child2)

            # mutation: swap a selected and an unselected gene
            def mutate(child):
                if rng.rand() < mutation_rate:
                    ones_idx = np.where(child == 1)[0]
                    zeros_idx = np.where(child == 0)[0]
                    if ones_idx.size > 0 and zeros_idx.size > 0:
                        i1 = rng.choice(ones_idx)
                        i0 = rng.choice(zeros_idx)
                        child[i1] = 0
                        child[i0] = 1
                return child

            child1 = mutate(child1)
            child2 = mutate(child2)

            new_pop[i] = child1
            if i + 1 < pop_size:
                new_pop[i + 1] = child2

        pop = new_pop
        fitness_vals = np.array([fitness(ind) for ind in pop])
        gen_best = pop[np.argmax(fitness_vals)].copy()
        if fitness_vals.max() > fitness(best):
            best = gen_best

    indices = np.where(best == 1)[0]
    subset_grad = np.mean(local_grads[indices], axis=0)
    dissimilarity = np.linalg.norm(subset_grad - global_grad)
    best_f = fitness(best)

    return indices, {'dissimilarity': dissimilarity, 'best_fitness': best_f}
