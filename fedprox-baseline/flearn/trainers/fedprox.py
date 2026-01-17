import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from flearn.trainers.fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad
from flearn.selection.genetic import genetic_select


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train')
        self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])
        super(Server, self).__init__(params, learner, dataset)
        self.current_round = 0
        self.selection_history = {}

        # GA control parameters (can be provided via params)
        self.ga_params = {
            'pop_size': int(params.get('ga_pop_size', 30)),
            'num_gen': int(params.get('ga_num_gen', 15)),
            'mutation_rate': float(params.get('ga_mutation_rate', 0.1)),
            'crossover_rate': float(params.get('ga_crossover_rate', 0.9)),
            'selection_method': params.get('ga_selection_method', 'tournament'),
            'tournament_size': int(params.get('ga_tournament_size', 3)),
            'selection_penalty': float(params.get('ga_selection_penalty', 0.0))
        }

        # Historical data for advanced strategies
        n_clients = len(self.clients)
        self.selection_counts = np.zeros(n_clients, dtype=int)  # how often each client was selected
        self.historical_local_loss = np.zeros(n_clients, dtype=float)  # placeholder moving average of local loss
        self.last_stragglers = np.zeros(n_clients, dtype=int)  # 1 if straggler/dropped last round

        # GA immediate/saveable data container
        self.ga_history = {
            'local_grads': None,
            'global_grad': None,
            'last_selected_indices': None,
            'last_gradient_dissimilarity': None,
        }

        # Add to Server.__init__ method after GA params
        self.sa_params = {
            'initial_temp': getattr(options, 'sa_initial_temp', 100.0),
            'cooling_rate': getattr(options, 'sa_cooling_rate', 0.95),
            'min_temp': getattr(options, 'sa_min_temp', 0.1),
            'epochs': getattr(options, 'sa_epochs', 30),
        }

    def train(self):
        '''Train using Federated Proximal'''
        print('Training with {} workers ---'.format(self.clients_per_round))

        for i in range(self.num_rounds):
            # test model
            if i % self.eval_every == 0:
                stats = self.test() # have set the latest model for all clients
                stats_train = self.train_error_and_loss()

                tqdm.write('At round {} accuracy: {}'.format(i, np.sum(stats[3])*1.0/np.sum(stats[2])))  # testing accuracy
                tqdm.write('At round {} training accuracy: {}'.format(i, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
                tqdm.write('At round {} training loss: {}'.format(i, np.dot(stats_train[4], stats_train[2])*1.0/np.sum(stats_train[2])))

            model_len = process_grad(self.latest_model).size
            global_grads = np.zeros(model_len)
            client_grads = np.zeros(model_len)
            num_samples = []
            local_grads = []

            for c in self.clients:
                num, client_grad = c.get_grads(model_len)
                local_grads.append(client_grad)
                num_samples.append(num)
                global_grads = np.add(global_grads, client_grad * num)
            global_grads = global_grads * 1.0 / np.sum(np.asarray(num_samples))

            difference = 0
            for idx in range(len(self.clients)):
                difference += np.sum(np.square(global_grads - local_grads[idx]))
            difference = difference * 1.0 / len(self.clients)
            tqdm.write('gradient difference: {}'.format(difference))

            indices, selected_clients = self.select_clients(
                    i,
                    num_clients=self.clients_per_round,
                    local_grads=np.asarray(local_grads),
                    global_grad=global_grads,
                    samples=np.asarray(num_samples))  # uniform sampling
            np.random.seed(i)  # make sure that the stragglers are the same for FedProx and FedAvg
            active_clients = np.random.choice(selected_clients, round(self.clients_per_round * (1 - self.drop_percent)), replace=False)

            csolns = [] # buffer for receiving client solutions

            self.inner_opt.set_params(self.latest_model, self.client_model)

            for idx, c in enumerate(selected_clients.tolist()):
                # communicate the latest model
                c.set_params(self.latest_model)

                total_iters = int(self.num_epochs * c.num_samples / self.batch_size)+2 # randint(low,high)=[low,high)

                # solve minimization locally
                if c in active_clients:
                    soln, stats = c.solve_inner(num_epochs=self.num_epochs, batch_size=self.batch_size)
                else:
                    #soln, stats = c.solve_iters(num_iters=np.random.randint(low=1, high=total_iters), batch_size=self.batch_size)
                    soln, stats = c.solve_inner(num_epochs=np.random.randint(low=1, high=self.num_epochs), batch_size=self.batch_size)

                # gather solutions from client
                csolns.append(soln)
        
                # track communication cost
                self.metrics.update(rnd=i, cid=c.id, stats=stats)

            # update models
            self.latest_model = self.aggregate(csolns)
            self.client_model.set_params(self.latest_model)

        # final test model
        stats = self.test()
        stats_train = self.train_error_and_loss()
        self.metrics.accuracies.append(stats)
        self.metrics.train_accuracies.append(stats_train)
        tqdm.write('At round {} accuracy: {}'.format(self.num_rounds, np.sum(stats[3])*1.0/np.sum(stats[2])))
        tqdm.write('At round {} training accuracy: {}'.format(self.num_rounds, np.sum(stats_train[3])*1.0/np.sum(stats_train[2])))
    
        def select_clients(self, round, num_clients, local_grads=None, global_grad=None, samples=None):
            """Select clients based on the specified method."""
            
            # Store current round for tracking
            self.current_round = round
            
            # Convert client objects to indices
            client_indices = list(range(len(self.clients)))
            num_clients = min(num_clients, len(client_indices))
            
            if self.selection_method == 'random':
                # Use parent class random selection
                indices, selected_clients = super().select_clients(round, num_clients)
                
                # Calculate gradient dissimilarity for random selection
                if local_grads is not None and global_grad is not None:
                    dissimilarity = self._calculate_gradient_dissimilarity(indices, local_grads, global_grad)
                else:
                    dissimilarity = 0.0
                
                # Update selection info
                self.selection_history[round] = {
                    'selected': indices,
                    'dissimilarity': dissimilarity,
                    'selection_time': 0.0,
                    'method': 'random'
                }
                
                # Update selection counts
                self.selection_counts[indices] += 1
                
                return indices, selected_clients
            
            elif self.selection_method == 'ga':
                # Delegate to GA and pass precomputed gradients when available
                indices, selected_clients = self.genetic(num_clients, local_grads=local_grads, 
                                                        global_grad=global_grad, samples=samples)
                return indices, selected_clients
                
            elif self.selection_method == 'pso':
                # Particle Swarm Optimization selection
                indices = self.particle_swarm(num_clients, local_grads, global_grad, samples)
                selected_clients = np.asarray(self.clients)[indices]
                return indices, selected_clients
            
            elif self.selection_method == 'sa':
                # Simulated Annealing selection
                indices = self.simulated_annealing(num_clients, local_grads, global_grad, samples)
                selected_clients = np.asarray(self.clients)[indices]
                return indices, selected_clients
            
            else:
                # Default to random selection
                np.random.seed(round)
                indices = np.random.choice(client_indices, num_clients, replace=False)
                
                # Calculate gradient dissimilarity
                if local_grads is not None and global_grad is not None:
                    dissimilarity = self._calculate_gradient_dissimilarity(indices, local_grads, global_grad)
                else:
                    dissimilarity = 0.0
                
                # Update selection info
                self.selection_history[round] = {
                    'selected': indices,
                    'dissimilarity': dissimilarity,
                    'selection_time': 0.0,
                    'method': 'random'
                }
                
                # Update selection counts
                self.selection_counts[indices] += 1
                
                return indices, np.asarray(self.clients)[indices]
    
    def genetic(self, num_clients, local_grads=None, global_grad=None, samples=None):
        # Delegate to the modular GA selector (keeps this class clean).
        num_clients = min(num_clients, len(self.clients))
        n = len(self.clients)
        k = int(num_clients)

        # trivial cases
        if k <= 0:
            return np.array([], dtype=int), np.asarray(self.clients)[[]]
        if k >= n:
            indices = np.arange(n)
            if hasattr(self, 'selection_counts'):
                self.selection_counts += 1
            self.ga_history['last_selected_indices'] = indices
            self.ga_history['last_gradient_dissimilarity'] = 0.0
            return indices, np.asarray(self.clients)[indices]

        # If gradients weren't passed in, compute them here (fallback)
        if local_grads is None or global_grad is None or samples is None:
            model_len = process_grad(self.latest_model).size
            local_grads = []
            samples = []
            self.client_model.set_params(self.latest_model)
            for c in self.clients:
                num, grad = c.get_grads(model_len)
                samples.append(num)
                local_grads.append(grad)
            local_grads = np.asarray(local_grads)
            samples = np.asarray(samples, dtype=float)

            if samples.sum() > 0:
                global_grad = np.sum(local_grads * samples[:, None], axis=0) / np.sum(samples)
            else:
                global_grad = np.mean(local_grads, axis=0)

        # save immediate GA data
        self.ga_history['local_grads'] = local_grads
        self.ga_history['global_grad'] = global_grad

        # call generic GA selector in selection/genetic.py
        indices, info = genetic_select(
            local_grads=local_grads,
            global_grad=global_grad,
            k=k,
            samples=samples,
            selection_counts=getattr(self, 'selection_counts', None),
            ga_params=getattr(self, 'ga_params', None),
            rng=np.random.RandomState()
        )

        # update selection counts and history if available
        if hasattr(self, 'selection_counts') and indices.size > 0:
            self.selection_counts[indices] += 1
        self.ga_history['last_selected_indices'] = indices
        self.ga_history['last_gradient_dissimilarity'] = info.get('dissimilarity', None)

        return indices, np.asarray(self.clients)[indices]
    
    def particle_swarm(self, num_clients):
        # Placeholder for Particle Swarm Optimization client selection
        num_clients = min(num_clients, len(self.clients))
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return indices, np.asarray(self.clients)[indices]
    
    def simulated_annealing(self, num_clients, local_grads=None, global_grad=None, samples=None):
        """Select clients using Simulated Annealing algorithm."""
        import time
        start_time = time.time()
        
        # Get client indices
        n_clients = len(self.clients)
        client_indices = list(range(n_clients))
        num_clients = min(num_clients, n_clients)
        
        # Trivial cases
        if num_clients <= 0:
            info = {
                'dissimilarity': 0.0,
                'selection_time': 0.0,
                'method': 'sa',
                'temperature': 0.0,
                'energy': 0.0
            }
            self.selection_history[self.current_round] = info
            return np.array([], dtype=int)
            
        if num_clients >= n_clients:
            indices = np.arange(n_clients)
            self.selection_counts[indices] += 1
            
            # Calculate dissimilarity
            if local_grads is not None and global_grad is not None:
                dissimilarity = self._calculate_gradient_dissimilarity(indices, local_grads, global_grad)
            else:
                dissimilarity = 0.0
            
            info = {
                'dissimilarity': dissimilarity,
                'selection_time': 0.0,
                'method': 'sa',
                'temperature': 0.0,
                'energy': 0.0
            }
            self.selection_history[self.current_round] = info
            return indices
        
        # If gradients weren't passed, compute them
        if local_grads is None or global_grad is None:
            model_len = process_grad(self.latest_model).size
            local_grads = []
            samples = []
            self.client_model.set_params(self.latest_model)
            for c in self.clients:
                num, grad = c.get_grads(model_len)
                samples.append(num)
                local_grads.append(grad)
            local_grads = np.asarray(local_grads)
            samples = np.asarray(samples, dtype=float)
            
            if samples.sum() > 0:
                global_grad = np.sum(local_grads * samples[:, None], axis=0) / np.sum(samples)
            else:
                global_grad = np.mean(local_grads, axis=0)
        
        # Initialize SA parameters
        initial_temp = self.sa_params['initial_temp']
        cooling_rate = self.sa_params['cooling_rate']
        min_temp = self.sa_params['min_temp']
        epochs = self.sa_params['epochs']
        
        current_temp = initial_temp
        
        # Generate initial random solution
        rng = np.random.RandomState(self.current_round)
        current_solution = rng.choice(client_indices, size=num_clients, replace=False)
        
        # Calculate initial energy (fitness)
        current_energy = self._sa_energy_function(
            current_solution, local_grads, global_grad, samples
        )
        
        best_solution = current_solution.copy()
        best_energy = current_energy
        
        # SA iterations
        for epoch in range(epochs):
            # Generate neighbor solution by swapping one client
            neighbor_solution = self._sa_generate_neighbor(
                current_solution, client_indices, num_clients, rng
            )
            
            # Calculate neighbor energy
            neighbor_energy = self._sa_energy_function(
                neighbor_solution, local_grads, global_grad, samples
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
                accept_prob = np.exp(-delta_energy / current_temp)
                if rng.random() < accept_prob:
                    current_solution = neighbor_solution
                    current_energy = neighbor_energy
            
            # Cool down temperature
            current_temp *= cooling_rate
            if current_temp < min_temp:
                current_temp = min_temp
        
        # Calculate gradient dissimilarity for selected solution
        dissimilarity = self._calculate_gradient_dissimilarity(
            best_solution, local_grads, global_grad
        )
        
        # Update selection counts
        self.selection_counts[best_solution] += 1
        
        selection_time = time.time() - start_time
        
        # Store selection info
        info = {
            'dissimilarity': dissimilarity,
            'selection_time': selection_time,
            'method': 'sa',
            'temperature': current_temp,
            'energy': best_energy,
            'selected': best_solution.tolist()
        }
        
        self.selection_history[self.current_round] = info
        
        # Log selection info
        from tqdm import tqdm
        tqdm.write(f'Round {self.current_round}: SA selected {len(best_solution)} clients')
        tqdm.write(f'  Gradient dissimilarity: {dissimilarity:.4f}')
        tqdm.write(f'  Final temperature: {current_temp:.2f}')
        tqdm.write(f'  Energy: {best_energy:.4f}')
        tqdm.write(f'  Selection time: {selection_time:.3f}s')
        
        return best_solution

    def _sa_energy_function(self, solution, local_grads, global_grad, samples):
        """Energy function to minimize (lower = better)."""
        energy = 0.0
        
        # 1. Gradient dissimilarity term
        if len(solution) > 0:
            dissim = self._calculate_gradient_dissimilarity(solution, local_grads, global_grad)
            energy += dissim
        
        # 2. Fairness penalty term
        if len(solution) > 0:
            selected_counts = self.selection_counts[solution]
            if np.mean(selected_counts) > 0:
                fairness_penalty = np.std(selected_counts) / np.mean(selected_counts)
                energy += 0.1 * fairness_penalty  # Weighted fairness
        
        # 3. Sample size consideration
        if samples is not None and len(solution) > 0:
            # Encourage selecting clients with more samples
            sample_weights = samples[solution]
            avg_samples = np.mean(sample_weights)
            if avg_samples > 0:
                sample_penalty = 1.0 / avg_samples  # Inverse so more samples = lower penalty
                energy += 0.05 * sample_penalty
        
        return energy


    def _sa_generate_neighbor(self, current_solution, all_clients, num_clients, rng):
        """Generate a neighboring solution by swapping one client."""
        neighbor = current_solution.copy()
        
        if len(all_clients) <= num_clients:
            return neighbor
        
        # Randomly select a client to swap out
        swap_out_idx = rng.randint(0, len(neighbor))
        client_to_swap = neighbor[swap_out_idx]
        
        # Find clients not in current solution
        not_selected = [c for c in all_clients if c not in neighbor]
        
        if not_selected:
            # Swap with a random non-selected client
            new_client = rng.choice(not_selected)
            neighbor[swap_out_idx] = new_client
        
        return neighbor
    
    def _calculate_gradient_dissimilarity(self, indices, local_grads, global_grad):
        """Calculate average gradient dissimilarity for selected clients."""
        if len(indices) == 0 or global_grad is None:
            return 0.0
        
        total_dissimilarity = 0.0
        valid_clients = 0
        
        for idx in indices:
            if idx < len(local_grads) and local_grads[idx] is not None:
                local_grad = local_grads[idx]
                
                # Calculate cosine dissimilarity
                cos_similarity = np.dot(local_grad, global_grad) / (
                    np.linalg.norm(local_grad) * np.linalg.norm(global_grad) + 1e-8
                )
                dissimilarity = 1.0 - cos_similarity
                total_dissimilarity += dissimilarity
                valid_clients += 1
        
        return total_dissimilarity / max(valid_clients, 1)
