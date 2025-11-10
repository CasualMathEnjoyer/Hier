import json
import keras
import optuna
from optuna.samplers import TPESampler, GPSampler
import os
import numpy as np
import random
import argparse

from s2s_transformer_pipeline import run_model_pipeline
from Utils.data_preparation import get_history_dict


model_compile_settings_path = 'model_compile_settings.json'
with open(model_compile_settings_path, encoding="utf-8") as f:
    model_compile_settings = json.load(f)

parser = argparse.ArgumentParser()
parser.add_argument("--run_settings", type=str, required=True,
                    help="Path to run_settings JSON file")
args = parser.parse_args()

with open(args.run_settings, encoding="utf-8") as f:
    run_settings = json.load(f)


OPTUNA_CROSS_STUDY_CACHE = "main_30_cache.json"
N_SEEDS = 10
N_TRIALS = 20
OPTIMIZERS = ["GP", "TPE"]


def load_trial_cache():
    if os.path.exists(OPTUNA_CROSS_STUDY_CACHE):
        with open(OPTUNA_CROSS_STUDY_CACHE, "r") as f:
            return json.load(f)
    return {}

def save_trial_cache(cache):
    with open(OPTUNA_CROSS_STUDY_CACHE, "w") as f:
        json.dump(cache, f, indent=4)

def model_params_to_cache_key(params):
    return "_".join(f"{k}={v}" for k, v in sorted(params.items()))

def get_cache_stats():
    cache = load_trial_cache()
    total_trials = len(cache)
    successful_trials = sum(1 for v in cache.values() if v["val_accuracy"] != float('-inf'))
    failed_trials = total_trials - successful_trials
    
    if successful_trials > 0:
        best_accuracy = max(v["val_accuracy"] for v in cache.values() if v["val_accuracy"] != float('-inf'))
        best_params = next(v["model_settings"] for v in cache.values() if v["val_accuracy"] == best_accuracy)
    else:
        best_accuracy = None
        best_params = None
    
    return {
        "total_trials": total_trials,
        "successful_trials": successful_trials,
        "failed_trials": failed_trials,
        "best_accuracy": best_accuracy,
        "best_params": best_params
    }

def print_cache_summary():
    stats = get_cache_stats()
    print("\n" + "="*50)
    print("CACHE SUMMARY")
    print("="*50)
    print(f"Total cached trials: {stats['total_trials']}")
    print(f"Successful trials: {stats['successful_trials']}")
    print(f"Failed trials: {stats['failed_trials']}")
    if stats['best_accuracy'] is not None:
        print(f"Best accuracy: {stats['best_accuracy']:.4f}")
        print(f"Best parameters: {stats['best_params']}")
    print("="*50 + "\n")

def objective(trial, optimizer_name, seed):
    trial.set_user_attr("sampler", optimizer_name)
    trial.set_user_attr("seed", seed)

    run_settings['model_name_short'] = f"{trial.study.study_name}_{trial.number}"

    model_settings = {
        "h": trial.suggest_categorical("h", [2, 4, 8]),
        "d_k": trial.suggest_categorical("d_k", [32, 64]),
        "d_ff": trial.suggest_categorical("d_ff", [512, 1024, 2048]),
        "d_model": trial.suggest_categorical("d_model", [256, 512]),
        "n": trial.suggest_categorical("n", [2, 4, 6])
    }

    model_settings['d_v'] = model_settings['d_k']

    cache = load_trial_cache()
    key = model_params_to_cache_key(model_settings)

    if key in cache:
        print(f"[CACHE] - using cached result for {key}")
        trial.set_user_attr("cached_from_previous", True)
        return cache[key]["val_accuracy"]

    trial.set_user_attr("cached_from_previous", False)

    try:
        print(f"[TRIAL] - running new trial with params: {model_settings}")
        run_model_pipeline(model_settings, model_compile_settings, run_settings)

        all_models_path = run_settings["all_models_path"]
        model_name_short = run_settings["model_name_short"]

        model_folder_path = os.path.join(all_models_path, model_name_short)

        if not os.path.exists(model_folder_path): 
            raise FileNotFoundError(f"Model folder not found: {model_folder_path}")

        model_full_path = os.path.join(model_folder_path, model_name_short)
        history_dict = f"{model_full_path}_HistoryDict"

        history_dict_data = get_history_dict(history_dict, False)
        val_accuracy = max(history_dict_data[f"val_accuracy"])

        print(f"[TRIAL] - completed. Val_Accuracy: {val_accuracy}")

        cache[key] = {
            "val_accuracy": val_accuracy,
            "model_settings": model_settings,
            "trial_number": trial.number,
            "study_name": trial.study.study_name,
            "timestamp": trial.datetime_start.isoformat() if trial.datetime_start else None
        }
        save_trial_cache(cache)
        print(f"[CACHE] - saved result to cache for key: {key}")

        return val_accuracy

    except Exception as e:
        print(f"[TRIAL] - failed with error: {str(e)}")
        cache[key] = {
            "val_accuracy": float('-inf'),
            "model_settings": model_settings,
            "trial_number": trial.number,
            "study_name": trial.study.study_name,
            "error": str(e),
            "timestamp": trial.datetime_start.isoformat() if trial.datetime_start else None
        }
        save_trial_cache(cache)
        print(f"[CACHE] - saved failed trial to cache for key: {key}")
        raise e


if __name__ == "__main__":
    print_cache_summary()

    storage_url = "sqlite:///optuna_study.db"

    for seed in range(N_SEEDS):
        print(f"\n{'='*80}")
        print(f"SEED {seed}")
        print(f"{'='*80}\n")

        np.random.seed(seed)
        random.seed(seed)
        run_settings['seed'] = seed

        for optimizer_name in OPTIMIZERS:
            study_name = f"stochastic_comparison_{optimizer_name}_seed_{seed}"

            if optimizer_name == "GP":
                sampler = GPSampler(seed=seed)
            elif optimizer_name == "TPE":
                sampler = TPESampler(seed=seed)
            else:
                raise ValueError(f"Unknown optimizer: {optimizer_name}")

            study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                direction="maximize",
                sampler=sampler,
                load_if_exists=True
            )

            print(f"[OPTIMIZATION] - {optimizer_name} with seed {seed}")
            print(f"[OPTIMIZATION] - study: {study.study_name}")
            print(f"[OPTIMIZATION] - existing trials in study: {len(study.trials)}")

            def objective_wrapper(trial):
                return objective(trial, optimizer_name, seed)

            study.optimize(objective_wrapper, n_trials=N_TRIALS)

            print(f"\n[OPTIMIZATION] - {optimizer_name} seed {seed} completed")
            print(f"  Best value: {study.best_value:.4f}")
            print(f"  Best parameters: {study.best_trial.params}")
            print(f"  Trial number: {study.best_trial.number}")

        print_cache_summary()

    print("\n" + "="*80)
    print("ALL COMPARISONS COMPLETED")
    print("="*80)
    print(f"Created {len(OPTIMIZERS) * N_SEEDS} studies:")
    for seed in range(N_SEEDS):
        for optimizer_name in OPTIMIZERS:
            print(f"  - stochastic_comparison_{optimizer_name}_seed_{seed}")
    print("="*80)

