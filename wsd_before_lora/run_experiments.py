"""Script to run experiments with different hyperparameters."""

import itertools
import json
import os
from datetime import datetime

import numpy as np

from train import train

# Define hyperparameter search space
# HYPERPARAMETERS = {
#     'epochs': [5, 10, 15],
#     'batch_size': [16, 32],
#     'lr': [5e-6, 1e-5, 2e-5, 3e-5, 5e-5],
#     'weight_decay': [1e-5, 1e-4]
# }

# HYPERPARAMETERS = {
#     'epochs': [15],
#     'batch_size': [16, 32],
#     'lr': [2e-4, 5e-5, 1e-4, 1e-3, 1e-2, 2e-5],
#     'weight_decay': [1e-5, 0, 2e-2],
#     'do': [0.1],
#     'optim': ['Adafactor'],
#     'momentum': [0.9, 0.5],
#     'temperature': [1, 1.1, 1.3, 1.5],
#     'encoder': ['mmbert'],
# }
# # 5e-5, is pretty low

# HYPERPARAMETERS = {
#     'epochs': [15],
#     'batch_size': [16],
#     'lr': [2e-4, 1e-3, 2e-5],
#     'weight_decay': [1e-5, 2e-2],
#     'do': [0.1],
#     'optim': ['Adafactor'],
#     'momentum': [0.9, 0.5],
#     'temperature': [2, 1, 1.5],
#     'encoder': ['mmbert'],
# }


HYPERPARAMETERS = {
    'epochs': [15],
    'batch_size': [32],
    'lr': [5e-4, 3e-5],
    'weight_decay': [1e-2],
    'do': [0.1],
    'optim': ['AdamW'],
    'temperature': [0.1],
    'encoder': ['xlmr-large'],
}


def run_experiments(param_grid=None):
    """Run training with different hyperparameter combinations."""
    
    # if param_grid is None:
    param_grid = HYPERPARAMETERS
    
    # Generate all combinations
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))

    print(combinations)
    
    print(f"Running {len(combinations)} experiments...\n")
    
    results = []
    for i, combo in enumerate(combinations, 1):
        config_dict = dict(zip(keys, combo))
        
        print(f"\n{'='*60}")
        print(f"Experiment {i}/{len(combinations)}")
        print(f"Config: {config_dict}")
        print(f"{'='*60}")
        
        try:
            model, eval_preds, test_preds = train(config_dict)
            results.append({
                'config': config_dict,
                'status': 'success',
                'eval_samples': len(eval_preds),
                'test_samples': len(test_preds)
            })
            print(f"✓ Completed successfully")
        except Exception as e:
            results.append({
                'config': config_dict,
                'status': 'failed',
                'error': str(e)
            })
            print(f"✗ Failed: {e}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    successful = [r for r in results if r['status'] == 'success']
    failed = [r for r in results if r['status'] == 'failed']
    
    print(f"Successful: {len(successful)}/{len(results)}")
    print(f"Failed: {len(failed)}/{len(results)}")
    
    if failed:
        print("\nFailed experiments:")
        for r in failed:
            print(f"  {r['config']}: {r['error']}")
    
    return results


def run_limited_grid():
    """Run a smaller grid for quick testing."""
    param_grid = {
        'epochs': [1],
        'batch_size': [8],
        'lr': [1e-3],
        'weight_decay': [1e-5]
    }
    return run_experiments(HYPERPARAMETERS)


def _safe_mean(values):
    return float(np.mean(values)) if values else 0.0


def _safe_std(values):
    if len(values) <= 1:
        return 0.0
    return float(np.std(values, ddof=1))


def _safe_sem(values):
    if len(values) <= 1:
        return 0.0
    return _safe_std(values) / float(np.sqrt(len(values)))


def _metric_stats(values):
    mean = _safe_mean(values)
    std = _safe_std(values)
    sem = _safe_sem(values)
    ci95 = 1.96 * sem
    return {
        'n': len(values),
        'mean': mean,
        'std': std,
        'sem': sem,
        'ci95': ci95,
        'min': float(min(values)) if values else 0.0,
        'max': float(max(values)) if values else 0.0,
    }


def run_statistical_experiments(base_config, seeds, output_root='outputs'):
    """Run the same training config with multiple seeds and report uncertainty.

    Args:
        base_config: dict with training config (without seed)
        seeds: iterable of integer seeds
        output_root: base output directory

    Returns:
        dict with per-seed results and aggregate statistics
    """
    seeds = list(seeds)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stats_dir = os.path.join(output_root, f"stats_{timestamp}")
    os.makedirs(stats_dir, exist_ok=True)

    print(f"Running statistical evaluation with {len(seeds)} seeds")
    print(f"Seeds: {seeds}")
    print(f"Base config: {base_config}")
    print(f"Statistics output directory: {stats_dir}")

    per_seed_results = []
    failures = []

    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*60}")
        print(f"Seed run {i}/{len(seeds)} - seed={seed}")
        print(f"{'='*60}")
        try:
            _, _, _, metrics = train(dict(base_config), seed=seed, return_metrics=True)

            eval_all = metrics.get('eval_scores', {}).get('ALL', {})
            test_all = metrics.get('test_scores', {}).get('ALL', {})

            run_result = {
                'seed': seed,
                'run_name': metrics.get('run_name'),
                'run_output_dir': metrics.get('run_output_dir'),
                'train_loss': metrics.get('train_loss'),
                'eval_loss': metrics.get('eval_loss'),
                'test_loss': metrics.get('test_loss'),
                'eval_f1_all': float(eval_all.get('f1', 0.0)),
                'test_f1_all': float(test_all.get('f1', 0.0)),
                'eval_precision_all': float(eval_all.get('precision', 0.0)),
                'eval_recall_all': float(eval_all.get('recall', 0.0)),
                'test_precision_all': float(test_all.get('precision', 0.0)),
                'test_recall_all': float(test_all.get('recall', 0.0)),
                'eval_scores': metrics.get('eval_scores', {}),
                'test_scores': metrics.get('test_scores', {}),
                'status': 'success',
            }
            per_seed_results.append(run_result)
            print(
                "Completed seed "
                f"{seed}: eval F1(ALL)={run_result['eval_f1_all']:.2f}, "
                f"test F1(ALL)={run_result['test_f1_all']:.2f}"
            )
        except Exception as e:
            err = {'seed': seed, 'status': 'failed', 'error': str(e)}
            failures.append(err)
            print(f"Failed seed {seed}: {e}")

    eval_f1_values = [r['eval_f1_all'] for r in per_seed_results]
    test_f1_values = [r['test_f1_all'] for r in per_seed_results]

    summary = {
        'timestamp': timestamp,
        'base_config': base_config,
        'seeds_requested': seeds,
        'num_success': len(per_seed_results),
        'num_failed': len(failures),
        'eval_f1_all_stats': _metric_stats(eval_f1_values),
        'test_f1_all_stats': _metric_stats(test_f1_values),
        'failures': failures,
    }

    payload = {
        'summary': summary,
        'per_seed_results': per_seed_results,
    }

    json_path = os.path.join(stats_dir, 'statistical_summary.json')
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    txt_path = os.path.join(stats_dir, 'statistical_summary.txt')
    with open(txt_path, 'w') as f:
        f.write("Statistical Evaluation Summary\n")
        f.write("=" * 60 + "\n")
        f.write(f"Requested seeds: {seeds}\n")
        f.write(f"Successful runs: {summary['num_success']}\n")
        f.write(f"Failed runs: {summary['num_failed']}\n\n")

        f.write("Eval F1 (ALL)\n")
        f.write(f"  mean: {summary['eval_f1_all_stats']['mean']:.4f}\n")
        f.write(f"  std:  {summary['eval_f1_all_stats']['std']:.4f}\n")
        f.write(f"  sem:  {summary['eval_f1_all_stats']['sem']:.4f}\n")
        f.write(f"  95% CI (half-width): +/- {summary['eval_f1_all_stats']['ci95']:.4f}\n\n")

        f.write("Test F1 (ALL)\n")
        f.write(f"  mean: {summary['test_f1_all_stats']['mean']:.4f}\n")
        f.write(f"  std:  {summary['test_f1_all_stats']['std']:.4f}\n")
        f.write(f"  sem:  {summary['test_f1_all_stats']['sem']:.4f}\n")
        f.write(f"  95% CI (half-width): +/- {summary['test_f1_all_stats']['ci95']:.4f}\n\n")

        f.write("Per-seed (ALL F1)\n")
        for r in per_seed_results:
            f.write(
                f"  seed={r['seed']}: "
                f"eval_f1={r['eval_f1_all']:.4f}, "
                f"test_f1={r['test_f1_all']:.4f}, "
                f"run_dir={r['run_output_dir']}\n"
            )

        if failures:
            f.write("\nFailures\n")
            for fail in failures:
                f.write(f"  seed={fail['seed']}: {fail['error']}\n")

    print(f"\n{'='*60}")
    print("STATISTICAL SUMMARY")
    print(f"{'='*60}")
    print(
        f"Eval F1 (ALL): {summary['eval_f1_all_stats']['mean']:.2f} "
        f"+/- {summary['eval_f1_all_stats']['ci95']:.2f} (95% CI, n={summary['eval_f1_all_stats']['n']})"
    )
    print(
        f"Test F1 (ALL): {summary['test_f1_all_stats']['mean']:.2f} "
        f"+/- {summary['test_f1_all_stats']['ci95']:.2f} (95% CI, n={summary['test_f1_all_stats']['n']})"
    )
    print(f"Summary JSON: {json_path}")
    print(f"Summary TXT: {txt_path}")

    return payload


if __name__ == '__main__':
    # Run limited grid for testing
    # Change to run_experiments() for full grid
    results = run_limited_grid()
