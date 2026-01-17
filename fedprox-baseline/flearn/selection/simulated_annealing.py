"""
Simulated Annealing (SA) client selector - Physics-based optimization algorithm.
Based on metallurgical annealing process.
"""
import numpy as np
import time
import math
from flearn.selection.selector import Selector

class SimulatedAnnealingSelector(Selector):
    """
    Physics-inspired client selector using simulated annealing.
    Mimics the annealing process in metallurgy where temperature controls
    the exploration vs exploitation trade-off.
    """
    
    def __init__(self, args, logger):
        super().__init__(args, logger)
        
        # SA parameters with physics-inspired defaults
        self.initial_temp = getattr(args, 'sa_initial_temp', 100.0)  # Starting "temperature"
        self.cooling_rate = getattr(args, 'sa_cooling_rate', 0.95)   # Geometric cooling
        self.min_temp = getattr(args, 'sa_min_temp', 0.1)            # Stopping temperature
        self.epochs = getattr(args, 'sa_epochs', 30)                 # Iterations per round
        self.current_temp = self.initial_temp
        
        # Selection history for fairness
        self.selection_counts = {}
        self.round_num = 0
        
        logger.info(f"SA Selector initialized: T0={self.initial_temp}, "
                   f"cooling={self.cooling_rate}, epochs={self.epochs}")
    
    def select(self, client_ids, num_clients, 
               local_grads, global_grad,
               client_samples=None,
               client_losses=None,
               selection_counts=None,
               rng=np.random.RandomState()):
        """
        Select clients using Simulated Annealing optimization.
        
        Args:
            client_ids: List of available client IDs
            num_clients: Number of clients to select
            local_grads: Local gradients from clients
            global_grad: Global gradient
            client_samples: Number of samples per client
            client_losses: Loss values per client
            selection_counts: Previous selection counts
            rng: Random number generator
            
        Returns:
            selected_ids: List of selected client IDs
            info: Dictionary with selection metrics
        """
        start_time = time.time()
        self.round_num += 1
        
        # Update selection counts
        if selection_counts:
            self.selection_counts = selection_counts
        
        # Reset temperature with logarithmic cooling
        self.current_temp = self.initial_temp / math.log(self.round_num + 2)
        
        # Generate initial random solution
        available_clients = list(client_ids)
        if len(available_clients) <= num_clients:
            # Not enough clients to select from
            selected = available_clients
            info = {
                'dissimilarity': 0.0,
                'selection_time': 0.0,
                'method': 'sa',
                'temperature': self.current_temp
            }
            return selected, info
        
        # Initial random selection
        current_solution = rng.choice(
            available_clients, 
            size=num_clients, 
            replace=False
        )
        
        # Calculate initial energy (fitness)
        current_energy = self._energy_function(
            current_solution, local_grads, global_grad, client_losses
        )
        
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        # Simulated Annealing iterations
        for epoch in range(self.epochs):
            # Generate neighbor solution
            neighbor_solution = self._get_neighbor(
                current_solution, available_clients, num_clients, rng
            )
            
            # Calculate neighbor energy
            neighbor_energy = self._energy_function(
                neighbor_solution, local_grads, global_grad, client_losses
            )
            
            # Energy difference (we minimize energy)
            delta_energy = neighbor_energy - current_energy
            
            # Acceptance probability using Boltzmann distribution
            if delta_energy < 0:
                # Better solution, always accept
                current_solution = neighbor_solution
                current_energy = neighbor_energy
                
                if current_energy < best_energy:
                    best_solution = current_solution.copy()
                    best_energy = current_energy
            else:
                # Worse solution, accept with probability exp(-ΔE/T)
                accept_prob = math.exp(-delta_energy / self.current_temp)
                if rng.random() < accept_prob:
                    current_solution = neighbor_solution
                    current_energy = neighbor_energy
            
            # Cool down (geometric cooling schedule)
            self.current_temp *= self.cooling_rate
            if self.current_temp < self.min_temp:
                self.current_temp = self.min_temp
        
        # Calculate metrics for selected solution
        dissimilarity = self._calculate_dissimilarity(
            best_solution, local_grads, global_grad
        )
        
        # Update selection counts
        for client_id in best_solution:
            self.selection_counts[client_id] = self.selection_counts.get(client_id, 0) + 1
        
        selection_time = time.time() - start_time
        
        info = {
            'dissimilarity': dissimilarity,
            'selection_time': selection_time,
            'method': 'sa',
            'temperature': self.current_temp,
            'energy': best_energy,
            'round': self.round_num
        }
        
        return list(best_solution), info
    
    def _energy_function(self, solution, local_grads, global_grad, client_losses=None):
        """
        Energy function to minimize. Lower energy = better solution.
        
        Combines:
        1. Gradient dissimilarity (main term)
        2. Fairness penalty (encourage uniform selection)
        3. Loss consideration (if available)
        """
        energy = 0.0
        
        # 1. Gradient dissimilarity term
        dissim = self._calculate_dissimilarity(solution, local_grads, global_grad)
        energy += dissim
        
        # 2. Fairness penalty term (encourage uniform selection)
        if solution and self.selection_counts:
            selected_counts = [self.selection_counts.get(cid, 0) for cid in solution]
            avg_count = np.mean(selected_counts) if selected_counts else 0
            if avg_count > 0:
                fairness_penalty = np.std(selected_counts) / avg_count
                energy += 0.1 * fairness_penalty  # Weighted fairness term
        
        # 3. Loss consideration (if losses are available)
        if client_losses and solution:
            avg_loss = np.mean([client_losses.get(cid, 0) for cid in solution])
            energy += 0.05 * avg_loss  # Small weight on loss
        
        return energy
    
    def _calculate_dissimilarity(self, client_ids, local_grads, global_grad):
        """Calculate average gradient dissimilarity for selected clients."""
        if not client_ids or global_grad is None:
            return 0.0
        
        total_dissimilarity = 0.0
        valid_clients = 0
        
        for client_id in client_ids:
            if client_id in local_grads and local_grads[client_id] is not None:
                local_grad = local_grads[client_id]
                
                # Handle different gradient formats
                if isinstance(local_grad, list):
                    # List of gradients
                    local_flat = np.concatenate([g.flatten() for g in local_grad])
                    global_flat = np.concatenate([g.flatten() for g in global_grad])
                else:
                    # Single gradient array
                    local_flat = local_grad.flatten()
                    global_flat = global_grad.flatten()
                
                # Calculate cosine dissimilarity
                norm_local = np.linalg.norm(local_flat)
                norm_global = np.linalg.norm(global_flat)
                
                if norm_local > 0 and norm_global > 0:
                    cos_similarity = np.dot(local_flat, global_flat) / (norm_local * norm_global)
                    dissimilarity = 1.0 - cos_similarity
                    total_dissimilarity += dissimilarity
                    valid_clients += 1
        
        return total_dissimilarity / max(valid_clients, 1)
    
    def _get_neighbor(self, current_solution, available_clients, num_clients, rng):
        """
        Generate a neighboring solution by swapping one client.
        This is the "thermal perturbation" in annealing.
        """
        neighbor = current_solution.copy()
        
        if len(available_clients) <= num_clients:
            return neighbor
        
        # Randomly select a client to swap out
        swap_out_idx = rng.randint(0, len(neighbor))
        client_to_swap = neighbor[swap_out_idx]
        
        # Find clients not in current solution
        not_selected = [c for c in available_clients if c not in neighbor]
        
        if not_selected:
            # Swap with a random non-selected client
            new_client = rng.choice(not_selected)
            neighbor[swap_out_idx] = new_client
        
        return neighbor
    
    def reset(self):
        """Reset selector state for new experiment."""
        self.current_temp = self.initial_temp
        self.selection_counts = {}
        self.round_num = 0