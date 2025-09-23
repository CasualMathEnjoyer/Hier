import json
import keras
import optuna
import optuna.visualization as vis
from optuna.samplers import TPESampler, RandomSampler, GPSampler, GridSampler
import os
import numpy as np
import random
import argparse

from s2s_transformer_pipeline import run_model_pipeline
from Utils.data_preparation import get_history_dict


model_compile_settings_path = 'model_compile_settings.json'
with open(model_compile_settings_path, encoding="utf-8") as f:
    model_compile_settings = json.load(f)
    
# run_settings = {
#     "all_models_path": "/home/katka/PycharmProjects/Hier/models",
#     "model_name_short": "model_small_1",
#
#     "different_output_model": False,
#     "output_model_name": "model_finetuned",
#
#     # "train_in_file_name" : "../data/src-sep-train.txt",
#     # "train_out_file_name" : "../data/tgt-train.txt",
#     # "val_in_file_name" : "../data/src-sep-val.txt",
#     # "val_out_file_name" : "../data/tgt-val.txt",
#     # "test_in_file_name" : "../data/src-sep-test.txt",
#     # "test_out_file_name" : "../data/tgt-test.txt",
#
#     # "train_in_file_name" : "../data/src-sep-train-short.txt",
#     # "train_out_file_name" : "../data/tgt-train-short.txt",
#     # "val_in_file_name" : "../data/src-sep-train-short.txt",
#     # "val_out_file_name" : "../data/tgt-train-short.txt",
#     # "test_in_file_name" : "../data/src-sep-test.txt",
#     # "test_out_file_name" : "../data/tgt-test.txt",
#
#     "train_in_file_name" : "../data/src-sep-train_30.txt",
#     "train_out_file_name" : "../data/tgt-train_30.txt",
#     "val_in_file_name" : "../data/src-sep-train_30.txt",
#     "val_out_file_name" : "../data/tgt-train_30.txt",
#     "test_in_file_name" : "../data/src-sep-test.txt",
#     "test_out_file_name" : "../data/tgt-test.txt",
#
#
#     "sep": " ",
#     "mezera": "_",
#     "end_line": "\n",
#
#     "new_model": 1,
#     "new_class_dict": 1,
#     "class_data": "processed_data_plk/optuna_30.plk",
#
#     "batch_size": 32,
#
#     "train": True,
#     "epochs": 20,
#     "repeat": 1,
#
#     "use_random_seed": False,
#     "seed": 42,
#
#     "finetune_model": False,
#     "finetune_source": "../data/train_src_separated.txt",
#     "finetune_tgt": "../data/train_trl.txt",
#
#     "test": False,
#     "testing_samples": 4,
#
#     "use_custom_testing": False,
#     "custom_test_src": "../data/test_src_separated.txt",
#     "custom_test_tgt": "../data/test_trl.txt",
#     "clear_testing_cache": 1,
#     "caching_in_testing": 0,
#
#     "version": "version",
#     "keras_version": keras.__version__
# }


# Parse CLI arguments
parser = argparse.ArgumentParser()
parser.add_argument("--run_settings", type=str, required=True,
                    help="Path to run_settings JSON file")
args = parser.parse_args()

# Load run_settings from JSON
with open(args.run_settings, encoding="utf-8") as f:
    run_settings = json.load(f)


OPTUNA_CROSS_STUDY_CACHE = "main_30_cache.json"

SEED = 42
np.random.seed(SEED)
random.seed(SEED)

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

def objective(trial):
    trial.set_user_attr("sampler", sampler_name)

    run_settings['model_name_short'] = f"{trial.study.study_name}_{trial.number}"

    # # first run
    # model_settings = {
    #     "h": trial.suggest_int("h", 1, 6),  # number of heads
    #     "d_k": trial.suggest_categorical("d_k", [16, 32, 64]),  # key dimension
    #     "d_v": trial.suggest_categorical("d_v", [16, 32, 64]),  # value dimension
    #     "d_ff": trial.suggest_int("d_ff", 64, 512, step=64),  # feedforward layer size
    #     "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
    #     "n": trial.suggest_int("n", 1, 4)  # number of layers
    # }

    # like in bakalarka
    model_settings = {
        "h": trial.suggest_categorical("h", [2, 4, 8]),  # number of heads
        "d_k": trial.suggest_categorical("d_k", [32, 64]),  # key dimension
        "d_ff": trial.suggest_categorical("d_ff", [512, 1024, 2048]),  # feedforward layer size
        "d_model": trial.suggest_categorical("d_model", [256, 512]),
        "n": trial.suggest_categorical("n", [2, 4, 6])  # number of layers
    }

    model_settings['d_v'] = model_settings['d_k']

    cache = load_trial_cache()
    key = model_params_to_cache_key(model_settings)

    # If params seen before → return cached result
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

        # Save the result to cache
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
        # Save failed trial to cache to avoid retrying
        cache[key] = {
            "val_accuracy": float('-inf'),  # Mark as failed
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
    # Print initial cache summary
    print_cache_summary()

    search_space = {
        "h": [2, 4, 8],
        "d_k": [32, 64],
        "d_ff": [512, 1024, 2048],
        "d_model": [256, 512],
        "n": [2, 4, 6],
    }
    
    samplers = {
        "TPE": TPESampler(seed=SEED),
        "GP": GPSampler(seed=SEED),
        "Random": RandomSampler(seed=SEED),
        "Grid": GridSampler(search_space=search_space, seed=SEED)
    }

    # sampler_name = "TPE"

    for sampler in samplers.keys():
        sampler_name = sampler

        study = optuna.create_study(study_name=f"bakalarka_{sampler_name}_main_local",
                                    storage="sqlite:///optuna_study.db",
                                    direction="maximize",
                                    sampler=samplers[sampler_name],
                                    load_if_exists=True)

        print(f"[OPTIMIZATION] - starting with {sampler_name} sampler")
        print(f"[OPTIMIZATION] - study: {study.study_name}")
        print(f"[OPTIMIZATION] - existing trials in study: {len(study.trials)}")

        study.optimize(objective, n_trials=50)

        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETED")
        print("="*60)
        print("Best trial:")
        print(f"  Value: {study.best_value:.4f}")
        print(f"  Parameters: {study.best_trial.params}")
        print(f"  Trial number: {study.best_trial.number}")

        # Print final cache summary
        print_cache_summary()

        # Optional: Print all cached results for analysis
        history = load_trial_cache()
        if history:
            print("\n📋 ALL CACHED RESULTS:")
            print("-" * 80)
            for key, result in sorted(history.items(), key=lambda x: x[1]["val_accuracy"], reverse=True):
                status = "✅" if result["val_accuracy"] != float('-inf') else "❌"
                print(f"{status} {key}: {result['val_accuracy']:.4f}")
            print("-" * 80)
