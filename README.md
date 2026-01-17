# Federated Learning with Meta-Heuristic Algorithms

This workspace hosts a FedProx baseline extended with meta-heuristic client selection experiments.

## Repository layout
- `fedprox-baseline/` – FedProx baseline plus genetic algorithm (GA), particle swarm optimization (PSO), simulated annealing (SA), and random client selection experiments.
- `fedprox-baseline/comparisons/` – experiment outputs and comparison artifacts.
- `fedprox-baseline/logs/` – training logs for different selection strategies.
- `fedprox-baseline/data/` – MNIST dataset folder expected by the baseline scripts.
- `fedprox-baseline/tests/` – small sanity checks for optimizers.
- `requirements*.txt` – dependency lists for different environments.

## Environment
- Python 3.8+ recommended.
- TensorFlow 1.15 or TensorFlow 2.x (compat.v1). The code uses `tf.compat.v1` with small compatibility shims when running on TF 2.x.
- Numpy, tqdm.

Suggested setup (virtualenv):

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy tqdm tensorflow==1.15
# or for TF 2.x compatibility: pip install tensorflow
```

## Running the FedProx baseline with meta-heuristics
The `fedprox-baseline/fedprox.sh` wrapper calls `fedprox-baseline/main.py`.
It expects: dataset, drop_percent, mu, selection_method.

Arguments to `fedprox.sh`:
- `$1` dataset (e.g., `mnist`)
- `$2` drop_percent (fraction of slower devices to drop; 0–1)
- `$3` mu (proximal term; float)
- `$4` selection_method (`random`, `ga`, `pso`, `sa`)

Example (MNIST, 10% drop, mu=1.0, random selection):

```bash
cd fedprox-baseline
./fedprox.sh mnist 0.1 1.0 random
```

Equivalent `main.py` invocation:

```bash
python3 -u main.py --dataset mnist --optimizer fedprox \
    --learning_rate 0.03 --num_rounds 200 --clients_per_round 10 \
    --eval_every 1 --batch_size 10 --num_epochs 20 \
    --model mclr --drop_percent 0.1 --mu 1.0 --selection_method random --seed 0
```

Key implementation points (baseline):
- Trainer: `fedprox-baseline/flearn/trainers/fedprox.py` (Server class)
- Model (MNIST): `fedprox-baseline/flearn/models/mnist/mclr.py`
- Entry point: `fedprox-baseline/main.py`
- Shell runner: `fedprox-baseline/fedprox.sh`

Notes and troubleshooting:
- Accuracy scaling: The trainer expects per-client evaluation to return a count of correct predictions (not a mean). If you see ~10–20% accuracy, verify model `test()` returns counts. MNIST code is patched accordingly.
- Default optimizer in `main.py` may still read `'fedavg'` while only `'fedprox'` is supported in `OPTIMIZERS`; pass `--optimizer fedprox` or use `fedprox.sh`.
- If TF 2.x reports missing `tf.compat.v1.layers.dense`, `main.py` maps it to `tf.keras.layers.Dense`.

## Paper
The paper describing these meta-heuristic client selection methods is included in the Nature-Inspired book uploaded at the repository root.

Direct link: [Nature-Inspired Metaheuristics_for_Client_Selection_FL.pdf](Nature-Inspired%20Metaheuristics_for_Client_Selection_FL.pdf)

## Upstream FedProx reference
This repository does not include the original FedProx implementation. For the upstream reference, see:

https://github.com/litian96/FedProx

If you need the vanilla baselines or original scripts (e.g., `run_fedprox.sh`, `run_fedavg.sh`), clone the upstream project separately.
