import json
import keras
import optuna
import optuna.visualization as vis
from optuna.samplers import TPESampler, RandomSampler, GPSampler, GridSampler
import os

from s2s_transformer_pipeline import run_model_pipeline
from Utils.data_preparation import get_history_dict


model_compile_settings_path = 'model_compile_settings.json'
with open(model_compile_settings_path, encoding="utf-8") as f:
    model_compile_settings = json.load(f)
    
run_settings = {
    "all_models_path": "/home/katka/PycharmProjects/Hier/models",
    "model_name_short": "model_small_1",

    "different_output_model": False,
    "output_model_name": "model_finetuned",

    # "train_in_file_name" : "../data/src-sep-train.txt",
    # "train_out_file_name" : "../data/tgt-train.txt",
    # "val_in_file_name" : "../data/src-sep-val.txt",
    # "val_out_file_name" : "../data/tgt-val.txt",
    # "test_in_file_name" : "../data/src-sep-test.txt",
    # "test_out_file_name" : "../data/tgt-test.txt",

    "train_in_file_name" : "../data/src-sep-train-short.txt",
    "train_out_file_name" : "../data/tgt-train-short.txt",
    "val_in_file_name" : "../data/src-sep-train-short.txt",
    "val_out_file_name" : "../data/tgt-train-short.txt",
    "test_in_file_name" : "../data/src-sep-test.txt",
    "test_out_file_name" : "../data/tgt-test.txt",

    "sep": " ",
    "mezera": "_",
    "end_line": "\n",

    "new_model": 1,
    "new_class_dict": 0,
    "class_data": "processed_data_plk/optuna_small.plk",

    "batch_size": 32,

    "train": True,
    "epochs": 20,
    "repeat": 1,

    "use_random_seed": False,
    "seed": 12612638,

    "finetune_model": False,
    "finetune_source": "../data/train_src_separated.txt",
    "finetune_tgt": "../data/train_trl.txt",

    "test": False,
    "testing_samples": 4,

    "use_custom_testing": False,
    "custom_test_src": "../data/test_src_separated.txt",
    "custom_test_tgt": "../data/test_trl.txt",
    "clear_testing_cache": 1,
    "caching_in_testing": 0,

    "version": "version",
    "keras_version": keras.__version__
}



def load_history():
    if os.path.exists(OPTUNA_HISTORY_FILE):
        with open(OPTUNA_HISTORY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_history(history):
    with open(OPTUNA_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def params_to_key(params):
    return "_".join(f"{k}={v}" for k, v in sorted(params.items()))




def objective(trial):
    trial.set_user_attr("sampler", sampler_name)

    run_settings['model_name_short'] = f"{trial.study.study_name}_{trial.number}"

    model_settings = {
        "h": trial.suggest_int("h", 1, 6),  # number of heads
        "d_k": trial.suggest_categorical("d_k", [16, 32, 64]),  # key dimension
        "d_v": trial.suggest_categorical("d_v", [16, 32, 64]),  # value dimension
        "d_ff": trial.suggest_int("d_ff", 64, 512, step=64),  # feedforward layer size
        "d_model": trial.suggest_categorical("d_model", [32, 64, 128]),
        "n": trial.suggest_int("n", 1, 4)  # number of layers
    }

    history = load_history()
    key = params_to_key(model_settings)

    # If params seen before → return cached result
    if key in history:
        print(f"⚡ Using cached result for {key}")
        trial.set_user_attr("cached_from_previous", True)
        return history[key]["val_accuracy"]

    trial.set_user_attr("cached_from_previous", False)

    run_model_pipeline(model_settings, model_compile_settings, run_settings)

    all_models_path = run_settings["all_models_path"]
    model_name_short = run_settings["model_name_short"]

    model_folder_path = os.path.join(all_models_path, model_name_short)

    if not os.path.exists(model_folder_path): raise FileNotFoundError(model_folder_path)

    model_full_path = os.path.join(model_folder_path, model_name_short)
    history_dict = f"{model_full_path}_HistoryDict"

    history = get_history_dict(history_dict,False)

    print("Val_Accuracy:", history[f"val_accuracy"])

    return max(history[f"val_accuracy"])





OPTUNA_HISTORY_FILE = "trial_history.json"



if __name__ == "__main__":
    samplers = {
        "TPE": TPESampler(),
        "GP": GPSampler(),
        "Random": RandomSampler(),
        "Grid": GridSampler({
            "h": [2, 4, 6],
            "d_k": [32, 64]
        })
    }

    sampler_name = "Random"

    study = optuna.create_study(study_name="s2s_small_random",
                                storage="sqlite:///optuna_study.db",
                                direction="maximize",
                                sampler=samplers[sampler_name],
                                load_if_exists=True)
    study.optimize(objective, n_trials=50)

    print("Best trial:")
    print(study.best_trial.params)
