import numpy as np
from tqdm import trange, tqdm
import tensorflow as tf

from flearn.trainers.fedbase import BaseFedarated
from flearn.optimizer.pgd import PerturbedGradientDescent
from flearn.utils.tf_utils import process_grad, process_sparse_grad
from flearn.selection.genetic import genetic_select
from flearn.selection.particle_swarm import pso_select


class Server(BaseFedarated):
    def __init__(self, params, learner, dataset):
        print('Using Federated prox to Train')
        self.inner_opt = PerturbedGradientDescent(params['learning_rate'], params['mu'])
        super(Server, self).__init__(params, learner, dataset)

        # GA control parameters (can be provided via params)
        self.ga_params = {
            'pop_size': int(params.get('ga_pop_size', 50)),
            'num_gen': int(params.get('ga_num_gen', 20)),
            'mutation_rate': float(params.get('ga_mutation_rate', 0.1)),
            'crossover_rate': float(params.get('ga_crossover_rate', 0.9)),
            'selection_method': params.get('ga_selection_method', 'tournament'),
            'tournament_size': int(params.get('ga_tournament_size', 3)),
            'selection_penalty': float(params.get('ga_selection_penalty', 0.0)),
            'elitism': int(params.get('ga_elitism', 1))
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

        # PSO control parameters
        self.pso_params = {
            'swarm_size': int(params.get('pso_swarm_size', 30)),
            'num_iter': int(params.get('pso_num_iter', 20)),
            'w': float(params.get('pso_w', 0.7)),
            'c1': float(params.get('pso_c1', 1.5)),
            'c2': float(params.get('pso_c2', 1.5)),
            'v_max': float(params.get('pso_v_max', 4.0)),
            'selection_penalty': float(params.get('pso_selection_penalty', 0.0))
        }

        # PSO immediate/saveable data container
        self.pso_history = {
            'local_grads': None,
            'global_grad': None,
            'last_selected_indices': None,
            'last_gradient_dissimilarity': None,
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
        if self.selection_method == 'random':
            return super().select_clients(round, num_clients)
        
        elif self.selection_method == 'ga':
            # Delegate to GA and pass precomputed gradients when available
            return self.genetic(round, num_clients, local_grads=local_grads, global_grad=global_grad, samples=samples)
            
        elif self.selection_method == 'pso':
            # Delegate to PSO with precomputed gradients
            return self.particle_swarm(round, num_clients, local_grads=local_grads, global_grad=global_grad, samples=samples)
        
        elif self.selection_method == 'sa':
            # Your Simulated Annealing logic here
            return self.simulated_annealing(num_clients)
    
    def genetic(self, round_idx, num_clients, local_grads=None, global_grad=None, samples=None):
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
            rng=np.random.RandomState(round_idx)
        )

        # update selection counts and history if available
        if hasattr(self, 'selection_counts') and indices.size > 0:
            self.selection_counts[indices] += 1
        self.ga_history['last_selected_indices'] = indices
        self.ga_history['last_gradient_dissimilarity'] = info.get('dissimilarity', None)

        return indices, np.asarray(self.clients)[indices]
    
    def particle_swarm(self, round_idx, num_clients, local_grads=None, global_grad=None, samples=None):
        """PSO-based client selection optimizing gradient similarity."""
        num_clients = min(num_clients, len(self.clients))
        n = len(self.clients)
        k = int(num_clients)

        # Trivial cases
        if k <= 0:
            return np.array([], dtype=int), np.asarray(self.clients)[[]]
        if k >= n:
            indices = np.arange(n)
            if hasattr(self, 'selection_counts'):
                self.selection_counts += 1
            self.pso_history['last_selected_indices'] = indices
            self.pso_history['last_gradient_dissimilarity'] = 0.0
            return indices, np.asarray(self.clients)[indices]

        # If gradients weren't passed in, compute them here
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

        # Save immediate PSO data
        self.pso_history['local_grads'] = local_grads
        self.pso_history['global_grad'] = global_grad

        # Call generic PSO selector
        indices, info = pso_select(
            local_grads=local_grads,
            global_grad=global_grad,
            k=k,
            samples=samples,
            selection_counts=getattr(self, 'selection_counts', None),
            pso_params=getattr(self, 'pso_params', None),
            rng=np.random.RandomState(round_idx)
        )

        # Update selection counts and history
        if hasattr(self, 'selection_counts') and indices.size > 0:
            self.selection_counts[indices] += 1
        self.pso_history['last_selected_indices'] = indices
        self.pso_history['last_gradient_dissimilarity'] = info.get('dissimilarity', None)

        return indices, np.asarray(self.clients)[indices]
    
    def simulated_annealing(self, num_clients):
        # Placeholder for Simulated Annealing client selection
        num_clients = min(num_clients, len(self.clients))
        indices = np.random.choice(range(len(self.clients)), num_clients, replace=False)
        return indices, np.asarray(self.clients)[indices]
