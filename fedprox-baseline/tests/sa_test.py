import numpy as np
from flearn.selection.simulated_annealing import simulated_annealing_select

rng = np.random.RandomState(123)

n_clients = 100
model_len = 50
local_grads = rng.randn(n_clients, model_len)
global_grad = np.mean(local_grads, axis=0)

indices, info = simulated_annealing_select(local_grads=local_grads, global_grad=global_grad, k=10,
                                          sa_params={'init_temp':1.0,'cooling_rate':0.99,'iterations':200}, rng=rng)

print('indices:', indices)
print('info:', info)
print('unique count:', len(np.unique(indices)))
