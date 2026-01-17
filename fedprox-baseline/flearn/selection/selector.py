"""
Base selector class for client selection algorithms in Federated Learning.
All metaheuristic selectors (GA, PSO, SA) inherit from this base class.
"""

import numpy as np
import time
from abc import ABC, abstractmethod


class Selector(ABC):
    """
    Abstract base class for all client selectors.
    Defines the interface that all selection algorithms must implement.
    """
    
    def __init__(self, args, logger):
        """
        Initialize the selector with arguments and logger.
        
        Args:
            args: Configuration arguments for the selector
            logger: Logger instance for tracking selection information
        """
        self.args = args
        self.logger = logger
        self.selection_counts = {}  # Track how often each client is selected
        self.selection_history = {}  # Store selection info per round
        self.round_num = 0
        
        # Common parameters for all selectors
        self.method = getattr(args, 'selection_method', 'random')
        
        logger.info(f"Initialized {self.__class__.__name__} selector")
    
    @abstractmethod
    def select(self, client_ids, num_clients, 
               local_grads, global_grad,
               client_samples=None,
               client_losses=None,
               selection_counts=None,
               rng=None):
        """
        Select clients based on the algorithm's strategy.
        
        Args:
            client_ids: List of available client IDs
            num_clients: Number of clients to select
            local_grads: Dictionary mapping client_id -> local gradient
            global_grad: Global gradient for comparison
            client_samples: Dictionary mapping client_id -> number of samples
            client_losses: Dictionary mapping client_id -> loss value
            selection_counts: Dictionary mapping client_id -> historical selection count
            rng: Random number generator (for reproducibility)
            
        Returns:
            selected_ids: List of selected client IDs
            info: Dictionary containing selection metrics
        """
        pass
    
    def reset(self):
        """Reset selector state for a new experiment."""
        self.selection_counts = {}
        self.selection_history = {}
        self.round_num = 0
        self.logger.info(f"Reset {self.__class__.__name__} selector")
    
    def update_selection_counts(self, selected_ids):
        """Update selection counts for selected clients."""
        for client_id in selected_ids:
            self.selection_counts[client_id] = self.selection_counts.get(client_id, 0) + 1
    
    def calculate_gradient_dissimilarity(self, selected_ids, local_grads, global_grad):
        """
        Calculate average gradient dissimilarity for selected clients.
        
        Args:
            selected_ids: List of selected client IDs
            local_grads: Dictionary of local gradients
            global_grad: Global gradient
            
        Returns:
            Average cosine dissimilarity (1 - cosine similarity)
        """
        if not selected_ids or global_grad is None:
            return 0.0
        
        total_dissimilarity = 0.0
        valid_clients = 0
        
        for client_id in selected_ids:
            if client_id in local_grads and local_grads[client_id] is not None:
                local_grad = local_grads[client_id]
                
                # Flatten gradients for comparison
                if isinstance(local_grad, list):
                    # List of gradients (e.g., for neural networks)
                    local_flat = np.concatenate([g.flatten() for g in local_grad])
                    global_flat = np.concatenate([g.flatten() for g in global_grad])
                else:
                    # Single gradient array
                    local_flat = local_grad.flatten()
                    global_flat = global_grad.flatten()
                
                # Calculate cosine similarity
                norm_local = np.linalg.norm(local_flat)
                norm_global = np.linalg.norm(global_flat)
                
                if norm_local > 0 and norm_global > 0:
                    cos_similarity = np.dot(local_flat, global_flat) / (norm_local * norm_global)
                    dissimilarity = 1.0 - cos_similarity
                    total_dissimilarity += dissimilarity
                    valid_clients += 1
        
        return total_dissimilarity / max(valid_clients, 1)
    
    def calculate_fairness_penalty(self, selected_ids):
        """
        Calculate fairness penalty based on selection history.
        
        Args:
            selected_ids: List of selected client IDs
            
        Returns:
            Fairness penalty (higher = less fair)
        """
        if not selected_ids or not self.selection_counts:
            return 0.0
        
        selected_counts = [self.selection_counts.get(cid, 0) for cid in selected_ids]
        if not selected_counts:
            return 0.0
        
        avg_count = np.mean(selected_counts)
        if avg_count > 0:
            return np.std(selected_counts) / avg_count
        return 0.0
    
    def get_selection_stats(self):
        """Get statistics about client selection."""
        if not self.selection_counts:
            return {
                'total_selections': 0,
                'unique_clients': 0,
                'avg_selections': 0,
                'std_selections': 0,
                'min_selections': 0,
                'max_selections': 0
            }
        
        counts = list(self.selection_counts.values())
        return {
            'total_selections': sum(counts),
            'unique_clients': len(counts),
            'avg_selections': np.mean(counts),
            'std_selections': np.std(counts),
            'min_selections': np.min(counts),
            'max_selections': np.max(counts)
        }


class RandomSelector(Selector):
    """
    Random client selector (baseline).
    Selects clients uniformly at random.
    """
    
    def __init__(self, args, logger):
        super().__init__(args, logger)
        self.logger.info("Initialized Random selector")
    
    def select(self, client_ids, num_clients, 
               local_grads=None, global_grad=None,
               client_samples=None, client_losses=None,
               selection_counts=None, rng=None):
        """
        Select clients randomly.
        
        Args:
            client_ids: List of available client IDs
            num_clients: Number of clients to select
            local_grads: Not used in random selection
            global_grad: Not used in random selection
            client_samples: Not used in random selection
            client_losses: Not used in random selection
            selection_counts: Optional, for updating counts
            rng: Random number generator
            
        Returns:
            selected_ids: Randomly selected client IDs
            info: Dictionary with selection metrics
        """
        start_time = time.time()
        
        if rng is None:
            rng = np.random.RandomState()
        
        # Ensure we don't select more clients than available
        num_select = min(num_clients, len(client_ids))
        
        # Random selection
        selected_ids = list(rng.choice(client_ids, size=num_select, replace=False))
        
        # Update selection counts if provided
        if selection_counts:
            self.selection_counts = selection_counts
        self.update_selection_counts(selected_ids)
        
        # Calculate gradient dissimilarity if gradients are available
        dissimilarity = 0.0
        if local_grads is not None and global_grad is not None:
            dissimilarity = self.calculate_gradient_dissimilarity(selected_ids, local_grads, global_grad)
        
        selection_time = time.time() - start_time
        
        # Store selection info
        self.selection_history[self.round_num] = {
            'selected': selected_ids,
            'dissimilarity': dissimilarity,
            'selection_time': selection_time,
            'method': 'random'
        }
        
        info = {
            'dissimilarity': dissimilarity,
            'selection_time': selection_time,
            'method': 'random',
            'round': self.round_num,
            'num_selected': len(selected_ids)
        }
        
        self.round_num += 1
        return selected_ids, info


# Factory function to create selectors
def create_selector(selector_type, args, logger):
    """
    Factory function to create selector instances.
    
    Args:
        selector_type: Type of selector ('random', 'ga', 'pso', 'sa')
        args: Configuration arguments
        logger: Logger instance
        
    Returns:
        Selector instance
    """
    if selector_type == 'random':
        return RandomSelector(args, logger)
    
    elif selector_type == 'ga':
        try:
            from flearn.selection.genetic import GeneticSelector
            return GeneticSelector(args, logger)
        except ImportError:
            logger.warning("Genetic selector not found, falling back to random")
            return RandomSelector(args, logger)
    
    elif selector_type == 'pso':
        try:
            from flearn.selection.particle_swarm import ParticleSwarmSelector
            return ParticleSwarmSelector(args, logger)
        except ImportError:
            logger.warning("PSO selector not found, falling back to random")
            return RandomSelector(args, logger)
    
    elif selector_type == 'sa':
        try:
            from flearn.selection.simulated_annealing import SimulatedAnnealingSelector
            return SimulatedAnnealingSelector(args, logger)
        except ImportError:
            logger.warning("SA selector not found, falling back to random")
            return RandomSelector(args, logger)
    
    else:
        logger.warning(f"Unknown selector type: {selector_type}, using random")
        return RandomSelector(args, logger)


# Utility functions for selection
def calculate_cosine_dissimilarity(grad1, grad2):
    """
    Calculate cosine dissimilarity between two gradients.
    
    Args:
        grad1: First gradient (numpy array)
        grad2: Second gradient (numpy array)
        
    Returns:
        Cosine dissimilarity (1 - cosine similarity)
    """
    # Flatten gradients
    if isinstance(grad1, list):
        grad1_flat = np.concatenate([g.flatten() for g in grad1])
        grad2_flat = np.concatenate([g.flatten() for g in grad2])
    else:
        grad1_flat = grad1.flatten()
        grad2_flat = grad2.flatten()
    
    # Calculate cosine similarity
    norm1 = np.linalg.norm(grad1_flat)
    norm2 = np.linalg.norm(grad2_flat)
    
    if norm1 > 0 and norm2 > 0:
        cos_sim = np.dot(grad1_flat, grad2_flat) / (norm1 * norm2)
        return 1.0 - cos_sim
    else:
        return 1.0  # Maximum dissimilarity if one gradient is zero


def normalize_gradient(gradient):
    """
    Normalize gradient to unit length.
    
    Args:
        gradient: Gradient to normalize
        
    Returns:
        Normalized gradient
    """
    if isinstance(gradient, list):
        # For list of gradients, flatten first
        flat_grad = np.concatenate([g.flatten() for g in gradient])
        norm = np.linalg.norm(flat_grad)
        if norm > 0:
            return [g / norm for g in gradient]
        return gradient
    else:
        # For single gradient array
        norm = np.linalg.norm(gradient)
        if norm > 0:
            return gradient / norm
        return gradient


def calculate_gradient_norms(local_grads):
    """
    Calculate L2 norms of local gradients.
    
    Args:
        local_grads: Dictionary of client_id -> gradient
        
    Returns:
        Dictionary of client_id -> gradient norm
    """
    norms = {}
    for client_id, grad in local_grads.items():
        if grad is not None:
            if isinstance(grad, list):
                flat_grad = np.concatenate([g.flatten() for g in grad])
                norms[client_id] = np.linalg.norm(flat_grad)
            else:
                norms[client_id] = np.linalg.norm(grad.flatten())
        else:
            norms[client_id] = 0.0
    return norms


# Example usage
if __name__ == "__main__":
    # Example of using the selector base class
    import logging
    
    # Setup logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Mock arguments
    class MockArgs:
        selection_method = 'random'
    
    args = MockArgs()
    
    # Create selector
    selector = create_selector('random', args, logger)
    
    # Test data
    client_ids = list(range(100))
    num_select = 10
    
    # Mock gradients (for demonstration)
    local_grads = {i: np.random.randn(100) for i in range(100)}
    global_grad = np.mean([local_grads[i] for i in range(100)], axis=0)
    
    # Select clients
    selected, info = selector.select(
        client_ids=client_ids,
        num_clients=num_select,
        local_grads=local_grads,
        global_grad=global_grad
    )
    
    print(f"Selected clients: {selected}")
    print(f"Selection info: {info}")
    
    # Get statistics
    stats = selector.get_selection_stats()
    print(f"Selection stats: {stats}")