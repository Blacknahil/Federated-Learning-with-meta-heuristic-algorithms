import os
import sys
import re
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)


def parse_log(file_name):
    """Parse a training log file for rounds, training loss, test accuracy, and gradient difference."""
    rounds = []
    train_loss = []
    test_acc = []
    grad_diff = []

    if not os.path.isfile(file_name):
        raise FileNotFoundError(file_name)

    with open(file_name, 'r') as fh:
        for line in fh:
            m = re.search(r'At round (\d+) training loss: ([0-9.eE+-]+)', line)
            if m:
                rounds.append(int(m.group(1)))
                train_loss.append(float(m.group(2)))
                continue

            m = re.search(r'At round (\d+) training accuracy: ([0-9.eE+-]+)', line)
            if m:
                # ensure rounds list contains this round (some logs print accuracy separately)
                r = int(m.group(1))
                if r not in rounds:
                    rounds.append(r)
                continue

            m = re.search(r'At round (\d+) accuracy: ([0-9.eE+-]+)', line)
            if m:
                test_acc.append((int(m.group(1)), float(m.group(2))))
                continue

            m = re.search(r'gradient difference: ([0-9.eE+-]+)', line)
            if m:
                grad_diff.append(float(m.group(1)))

    # Align arrays by rounds: train_loss may miss some rounds, test_acc is list of (round,value)
    if len(test_acc) > 0:
        # sort by round
        test_acc = sorted(test_acc, key=lambda x: x[0])
        test_rounds, test_vals = zip(*test_acc)
        test_rounds = np.array(test_rounds)
        test_vals = np.array(test_vals)
    else:
        test_rounds = np.array([])
        test_vals = np.array([])

    return np.array(rounds), np.array(train_loss), test_rounds, test_vals, np.array(grad_diff)


def rounds_to_target(test_rounds, test_vals, target):
    if test_vals.size == 0:
        return None
    idx = np.where(test_vals >= target)[0]
    if idx.size == 0:
        return None
    return int(test_rounds[idx[0]])


def summarize_and_plot(log1, log2, label1='Random', label2='Genetic', target_acc=0.9, last_n=10, out_prefix='comparison'):
    r1, loss1, tr1, acc1, gd1 = parse_log(log1)
    r2, loss2, tr2, acc2, gd2 = parse_log(log2)

    # Metrics
    best_acc1 = acc1.max() if acc1.size else None
    best_acc2 = acc2.max() if acc2.size else None

    rounds_target1 = rounds_to_target(tr1, acc1, target_acc)
    rounds_target2 = rounds_to_target(tr2, acc2, target_acc)

    stability1 = np.std(loss1[-last_n:]) if loss1.size >= 1 else None
    stability2 = np.std(loss2[-last_n:]) if loss2.size >= 1 else None

    avg_gd1 = np.mean(gd1) if gd1.size else None
    avg_gd2 = np.mean(gd2) if gd2.size else None

    # Print summary
    print('Summary:')
    print(f'  {label1}: best_acc={best_acc1}, rounds_to_{int(target_acc*100)}%={rounds_target1}, loss_std_last{last_n}={stability1}, avg_grad_diff={avg_gd1}')
    print(f'  {label2}: best_acc={best_acc2}, rounds_to_{int(target_acc*100)}%={rounds_target2}, loss_std_last{last_n}={stability2}, avg_grad_diff={avg_gd2}')

    if rounds_target1 is None or rounds_target2 is None:
        if rounds_target1 is None and rounds_target2 is None:
            print('\nNeither run reached target accuracy; comparisons use best achieved accuracy instead.')
        else:
            print('\nOne run reached the target while the other did not.')

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Training loss curves
    ax = axes[0, 0]
    if r1.size and loss1.size:
        ax.plot(r1[:len(loss1)], loss1, label=label1)
    if r2.size and loss2.size:
        ax.plot(r2[:len(loss2)], loss2, label=label2)
    ax.set_title('Training Loss')
    ax.set_xlabel('Round')
    ax.set_ylabel('Loss')
    ax.legend()

    # Test accuracy curves
    ax = axes[0, 1]
    if tr1.size and acc1.size:
        ax.plot(tr1, acc1, label=label1)
    if tr2.size and acc2.size:
        ax.plot(tr2, acc2, label=label2)
    ax.axhline(target_acc, color='k', linestyle='--', label=f'Target {target_acc:.2f}')
    ax.set_title('Test Accuracy')
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy')
    ax.legend()

    # Gradient dissimilarity
    ax = axes[1, 0]
    if gd1.size:
        ax.plot(np.arange(len(gd1)), gd1, label=label1)
    if gd2.size:
        ax.plot(np.arange(len(gd2)), gd2, label=label2)
    ax.set_title('Gradient Dissimilarity (per round)')
    ax.set_xlabel('Round')
    ax.set_ylabel('Dissimilarity')
    ax.legend()

    # Bar: final accuracy and rounds to target
    ax = axes[1, 1]
    bars = [best_acc1 if best_acc1 is not None else 0, best_acc2 if best_acc2 is not None else 0]
    ax.bar([0, 1], bars, color=['#1f77b4', '#ff7f0e'])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([label1, label2])
    ax.set_ylim(0, 1)
    ax.set_title('Best Test Accuracy')

    plt.tight_layout()
    out_png = out_prefix + '.png'
    fig.savefig(out_png)

    # Prepare summary data
    summary = {
        'label': [label1, label2],
        'best_acc': [best_acc1, best_acc2],
        'rounds_to_target': [rounds_target1, rounds_target2],
        'loss_std_lastN': [stability1, stability2],
        'avg_grad_diff': [avg_gd1, avg_gd2]
    }

    # ensure comparisons folder structure
    def detect_method(log_path):
        method = None
        try:
            with open(log_path, 'r') as fh:
                for line in fh:
                    m = re.search(r"selection_method\s*:\s*(\w+)", line)
                    if m:
                        method = m.group(1).strip()
                        break
        except Exception:
            method = None
        if method is None:
            return None
        mapping = {'ga': 'genetic', 'pso': 'pso', 'sa': 'simulated_annealing', 'random': 'random'}
        return mapping.get(method, method)

    m1 = detect_method(log1) or label1
    m2 = detect_method(log2) or label2
    comp_dir = os.path.join('comparisons', f"{m1}_vs_{m2}")
    os.makedirs(comp_dir, exist_ok=True)
    # also ensure per-method folder for the compared method
    method2_dir = os.path.join('comparisons', m2)
    os.makedirs(method2_dir, exist_ok=True)

    # save outputs into comparison folder and into the method folder
    out_png_comp = os.path.join(comp_dir, out_prefix + '.png')
    fig.savefig(out_png_comp)
    out_csv = os.path.join(comp_dir, out_prefix + '_summary.csv')
    import csv
    with open(out_csv, 'w', newline='') as cf:
        writer = csv.writer(cf)
        writer.writerow(list(summary.keys()))
        writer.writerows(zip(*list(summary.values())))

    # copy into per-method folder too
    out_png_method = os.path.join(method2_dir, out_prefix + '.png')
    try:
        import shutil
        shutil.copyfile(out_png_comp, out_png_method)
        shutil.copyfile(out_csv, os.path.join(method2_dir, out_prefix + '_summary.csv'))
    except Exception:
        pass

    # Save JSON summary as well
    try:
        import json
        json_file = os.path.join(comp_dir, out_prefix + '_summary.json')
        with open(json_file, 'w') as jf:
            json.dump(summary, jf, indent=2)
    except Exception:
        pass

    print(f'Plots and summary saved to {comp_dir} (and {method2_dir})')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_log', required=True, help='Log file for random selection run')
    parser.add_argument('--ga_log', required=True, help='Log file for genetic selection run')
    parser.add_argument('--target_acc', type=float, default=0.9, help='Target accuracy for convergence metric')
    parser.add_argument('--last_n', type=int, default=10, help='Number of last rounds to compute stability')
    parser.add_argument('--out_prefix', type=str, default='comparison', help='Output prefix for plots and summaries')
    args = parser.parse_args()

    summarize_and_plot(args.random_log, args.ga_log, label1='Random', label2='Genetic', target_acc=args.target_acc, last_n=args.last_n, out_prefix=args.out_prefix)


if __name__ == '__main__':
    main()
