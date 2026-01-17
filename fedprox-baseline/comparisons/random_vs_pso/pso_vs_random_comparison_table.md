# Comparison: Random vs PSO

## Metrics

| Metric | Random | PSO |
|---:|:---:|:---:|
| best_acc | 0.8976794680417968 | 0.8986293934048039 |
| rounds_to_90% | None | None |
| loss_std_last10 | 0.010780086003025692 | 0.011286012763595216 |
| avg_grad_diff | 15.495740692833978 | 18.29166481569364 |

## Training Parameters

| Parameter | Random | PSO |
|---|---|---|
| batch_size | 10 | 10 |
| clients_per_round | 10 | 10 |
| dataset | mnist | mnist |
| drop_percent | 0.5 | 0.5 |
| eval_every | 1 | 1 |
| learning_rate | 0.03 | 0.03 |
| model | mclr | mclr |
| model_params | (10,) | (10,) |
| mu | 1.0 | 1.0 |
| num_epochs | 20 | 20 |
| num_iters | 1 | 1 |
| num_rounds | 200 | 200 |
| optimizer | fedprox | fedprox |
| seed | 0 | 0 |
| selection_method | random | pso |
