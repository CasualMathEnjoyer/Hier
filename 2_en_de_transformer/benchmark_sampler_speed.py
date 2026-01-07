#!/usr/bin/env python3
"""
Benchmark script to measure how long it takes for TPE and GP samplers
to suggest parameters in Optuna.
"""

import time
import optuna
from optuna.samplers import TPESampler, GPSampler, RandomSampler
import numpy as np
import statistics

# Parameter space similar to your experiment
PARAM_SPACE = {
    "h": (1, 4),
    "d_k": (16, 48),
    "d_ff": (256, 1536),
    "d_model": (128, 300),
    "n": (1, 3)
}

N_TRIALS = 100  # Number of trials to run for each sampler
SEED = 42


def create_objective(sampler_name):
    """Create an objective function that measures suggestion time."""
    suggestion_times = []
    trial_counter = 0
    
    def objective(trial):
        # Measure time for all parameter suggestions
        start_time = time.perf_counter()
        
        params = {
            "h": trial.suggest_int("h", *PARAM_SPACE["h"]),
            "d_k": trial.suggest_int("d_k", *PARAM_SPACE["d_k"]),
            "d_ff": trial.suggest_int("d_ff", *PARAM_SPACE["d_ff"]),
            "d_model": trial.suggest_int("d_model", *PARAM_SPACE["d_model"]),
            "n": trial.suggest_int("n", *PARAM_SPACE["n"])
        }
        
        suggestion_time = time.perf_counter() - start_time
        suggestion_times.append(suggestion_time)
        
        # Return a different random value for each trial
        # Use trial number to ensure each trial gets a unique random value
        nonlocal trial_counter
        trial_counter += 1
        # Generate a unique random value for this trial using trial number as seed offset
        rng = np.random.RandomState(SEED + trial_counter * 1000)
        value = rng.uniform(0.0, 1.0)
        return value
    
    objective.suggestion_times = suggestion_times
    return objective


def benchmark_sampler(sampler, sampler_name, n_trials=N_TRIALS):
    """Benchmark a single sampler."""
    print(f"\n{'='*60}")
    print(f"Benchmarking {sampler_name} sampler")
    print(f"{'='*60}")
    
    # Create study
    study = optuna.create_study(
        study_name=f"benchmark_{sampler_name}",
        direction="maximize",
        sampler=sampler,
        load_if_exists=False
    )
    
    # Create objective
    objective = create_objective(sampler_name)
    
    # Measure total optimization time
    total_start = time.perf_counter()
    
    # Run trials
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    total_time = time.perf_counter() - total_start
    
    # Get suggestion times
    suggestion_times = objective.suggestion_times
    
    # Calculate statistics
    stats = {
        "sampler": sampler_name,
        "n_trials": len(suggestion_times),
        "total_time": total_time,
        "avg_suggestion_time": statistics.mean(suggestion_times),
        "median_suggestion_time": statistics.median(suggestion_times),
        "min_suggestion_time": min(suggestion_times),
        "max_suggestion_time": max(suggestion_times),
        "std_suggestion_time": statistics.stdev(suggestion_times) if len(suggestion_times) > 1 else 0,
        "avg_time_per_trial": total_time / len(suggestion_times) if suggestion_times else 0
    }
    
    return stats, suggestion_times


def print_stats(stats):
    """Print statistics in a nice format."""
    print(f"\nResults for {stats['sampler']} sampler:")
    print(f"  Number of trials: {stats['n_trials']}")
    print(f"  Total time: {stats['total_time']:.4f} seconds")
    print(f"  Average suggestion time: {stats['avg_suggestion_time']*1000:.4f} ms")
    print(f"  Median suggestion time: {stats['median_suggestion_time']*1000:.4f} ms")
    print(f"  Min suggestion time: {stats['min_suggestion_time']*1000:.4f} ms")
    print(f"  Max suggestion time: {stats['max_suggestion_time']*1000:.4f} ms")
    print(f"  Std deviation: {stats['std_suggestion_time']*1000:.4f} ms")
    print(f"  Average time per trial (including overhead): {stats['avg_time_per_trial']*1000:.4f} ms")


def main():
    print("="*60)
    print("Optuna Sampler Speed Benchmark")
    print("="*60)
    print(f"Parameter space: {PARAM_SPACE}")
    print(f"Number of trials per sampler: {N_TRIALS}")
    print(f"Seed: {SEED}")
    
    samplers = {
        "TPE": TPESampler(seed=SEED),
        "GP": GPSampler(seed=SEED),
        "Random": RandomSampler(seed=SEED)  # Baseline for comparison
    }
    
    all_stats = []
    all_times = {}
    
    for sampler_name, sampler in samplers.items():
        stats, times = benchmark_sampler(sampler, sampler_name, N_TRIALS)
        all_stats.append(stats)
        all_times[sampler_name] = times
        print_stats(stats)
    
    # Comparison summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Sampler':<10} {'Avg (ms)':<12} {'Median (ms)':<14} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("-" * 60)
    for stats in all_stats:
        print(f"{stats['sampler']:<10} "
              f"{stats['avg_suggestion_time']*1000:<12.4f} "
              f"{stats['median_suggestion_time']*1000:<14.4f} "
              f"{stats['min_suggestion_time']*1000:<12.4f} "
              f"{stats['max_suggestion_time']*1000:<12.4f}")
    
    # Speedup comparison
    if len(all_stats) >= 2:
        tpe_stats = next(s for s in all_stats if s['sampler'] == 'TPE')
        gp_stats = next(s for s in all_stats if s['sampler'] == 'GP')
        
        print(f"\n{'='*60}")
        print("SPEEDUP ANALYSIS")
        print(f"{'='*60}")
        if gp_stats['avg_suggestion_time'] > 0:
            speedup = gp_stats['avg_suggestion_time'] / tpe_stats['avg_suggestion_time']
            print(f"TPE is {speedup:.2f}x faster than GP on average")
        
        if tpe_stats['avg_suggestion_time'] > 0:
            speedup = tpe_stats['avg_suggestion_time'] / gp_stats['avg_suggestion_time']
            print(f"GP is {speedup:.2f}x faster than TPE on average")
    
    print(f"\n{'='*60}")
    print("Note: These times measure only the parameter suggestion phase.")
    print("Actual optimization time includes model training, which is much longer.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

