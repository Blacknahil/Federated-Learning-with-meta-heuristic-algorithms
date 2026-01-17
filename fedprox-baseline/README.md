# Federated Learning — FedProx baseline

This repository contains a FedProx implementation and meta-heuristic client selection experiments.

Quick summary
- Trainer: `flearn/trainers/fedprox.py` (Server class)
- Model (MNIST): `flearn/models/mnist/mclr.py`
- Entry point: `main.py`
- Shell runner: `fedprox.sh`

Requirements
- Python 3.8+ recommended
- TensorFlow 1.15 or TensorFlow 2.x (compat.v1). If using TF 2.x, the code uses `tf.compat.v1` and includes small compatibility shims.
- Numpy, tqdm

Install (recommended in a virtualenv)

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install numpy tqdm tensorflow==1.15
# or for TF 2.x compatibility: pip install tensorflow
```

Run federated training (example)

The provided `fedprox.sh` script wraps `main.py`. It expects arguments: dataset, drop_percent, mu, selection_method.

Arguments to `fedprox.sh`:
- `$1` dataset (e.g. `mnist`)
- `$2` drop_percent (float between 0 and 1; fraction of slower devices to simulate)
- `$3` mu (proximal term; float)
- `$4` selection_method (one of `random`, `ga`, `pso`, `sa`)

Example usage (MNIST, 10% drop, mu=1.0, random selection):

```bash
./fedprox.sh mnist 0.1 1.0 random
```

This expands to the equivalent `main.py` invocation shown in the script:

```bash
python3 -u main.py --dataset mnist --optimizer fedprox \
    --learning_rate 0.03 --num_rounds 200 --clients_per_round 10 \
    --eval_every 1 --batch_size 10 --num_epochs 20 \
    --model mclr --drop_percent 0.1 --mu 1.0 --selection_method random --seed 0
```



Notes and troubleshooting
- Accuracy scaling: The trainer expects per-client evaluation to return a count of correct predictions (not a mean). If you see unusually low accuracies (e.g. ~10-20%), verify model `test()` returns counts. The repository has been patched to return counts for MNIST.
- `main.py` default optimizer value in the code may be `'fedavg'` while only `'fedprox'` is supported in the `OPTIMIZERS` list; pass `--optimizer fedprox` or use `fedprox.sh`.
- If using TF 2.x and you hit missing `tf.compat.v1.layers.dense`, `main.py` already include a small compatibility shim that maps `tf.compat.v1.layers.dense` to `tf.keras.layers.Dense`.

Reproducibility
- Random seeds are set in `main.py` (seed flags). Use `--seed` to change randomness.

