import os
import json
import pickle
import random
from tqdm import tqdm
import time

from keras import backend as K

# from model_file_2 import *  # for loading
from model_file_mine import *
from Utils.model_function import save_model, load_model_mine, translate, get_epochs_train_accuracy, test_gpus
from Utils.data_preparation import get_history_dict, join_dicts, get_num_epochs_dict, get_num_epochs_csv, load_cached_dict, cache_dict, create_new_class_dict, load_class_data, create_new_class_dict_testing

import sys
sys.path.append("..")

from visualization.plot_model_history import plot_model_history

print("[PIPELINE] - starting transform2seq")


def set_seed(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    tf.config.experimental.enable_op_determinism()

def save_to_json(data, filename):
    with open(filename, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def run_model_pipeline(model_settings, model_compile_settings, run_settings):
    start_time = time.time()
    # ---------------------------------------------------------------
    all_models_path = run_settings["all_models_path"]
    model_name_short = run_settings["model_name_short"]

    model_folder_path = os.path.join(all_models_path, model_name_short)
    if not os.path.exists(model_folder_path):
        info = {'runs': 0}
    else:
        with open(os.path.join(model_folder_path, 'info.json')) as f:
            info = json.load(f)

    if not os.path.exists(model_folder_path): os.makedirs(model_folder_path)

    model_full_path = os.path.join(model_folder_path, model_name_short)
    model_best_full_path = os.path.join(model_folder_path, model_name_short + "_best")
    history_dict = f"{model_full_path}_HistoryDict"
    history_csv = os.path.join(model_folder_path, "training_history.csv")
    testing_cache_filename = model_full_path + '_TestingCache'

    if run_settings["use_random_seed"]: a = random.randrange(0, 2**32 - 1)
    else: a = run_settings["seed"]
    set_seed(a)

    runs = info['runs']

    run_settings['version'] = f'version{runs}'

    result_json_name = f"testing_{run_settings['version']}_samples_{run_settings['testing_samples']}.json"
    result_json_path = os.path.join(model_folder_path, result_json_name)

    save_to_json(model_settings, os.path.join(model_folder_path, f"model_settings_{runs}.json"))
    save_to_json(model_compile_settings, os.path.join(model_folder_path, f"model_compile_settings_{runs}.json"))
    save_to_json(run_settings, os.path.join(model_folder_path, f"run_settings_{runs}.json"))

    load_best = run_settings.get("load_best", False)

    info["runs"] += 1

    save_to_json(info, os.path.join(model_folder_path, f"info.json"))

    os.environ["KERAS_BACKEND"] = "tensorflow"
    test_gpus()

    # ---------------------------- DATA PROCESSING -------------------------------------------------
    if run_settings["new_class_dict"]: source, target, val_source, val_target = create_new_class_dict(run_settings)
    else: source, target, val_source, val_target = load_class_data(run_settings)

    # --------------------------------- MODEL ---------------------------------------------------------------------------
    old_dict = get_history_dict(history_dict, run_settings["new_model"])

    initial_epoch_dict = get_num_epochs_dict(history_dict)
    initial_epoch_csv = get_num_epochs_csv(history_csv)

    if initial_epoch_dict != 0 and initial_epoch_csv != 0:
        assert initial_epoch_dict == initial_epoch_csv, "initial_epoch_dict != initial_epoch_csv"
        initial_epoch = initial_epoch_dict
    elif initial_epoch_dict != 0:
        initial_epoch = initial_epoch_dict
    elif initial_epoch_csv != 0:
        initial_epoch = initial_epoch_csv
    else:
        initial_epoch = 0


    print("[MODEL] - starting")
    if run_settings["new_model"]: model = model_func(source.vocab_size, target.vocab_size, source.maxlen, target.maxlen, model_settings)
    elif load_best:
        print("[MODEL] - loading best model")
        model = load_model_mine(model_best_full_path)
    else:
        print("[MODEL] - loading last trained model")
        model = load_model_mine(model_full_path)

    print(f"[MODEL] - number of params = {model.count_params()}")
    # model.summary()

    if run_settings["finetune_model"]:
        extend_model_embeddings(model, source.vocab_size, target.vocab_size)
        model = adjust_output_layer(model, new_vocab_size=target.vocab_size)
        print("[MODEL] - fine-tuning preparation finished")
    else:
        print("[MODEL] - SKIPPING fine-tuning")

    # Compile the new model
    model.compile(optimizer=model_compile_settings['optimizer'],
                  loss=model_compile_settings['loss'],
                  metrics=model_compile_settings['metrics'])
    print("[MODEL] - COMPILED")

    # --------------------------------- TRAINING ------------------------------------------------------------------------
    if run_settings["train"]:
        model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath=f"{model_best_full_path}.keras",
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)

        early_stopping_callback = keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=5,
            restore_best_weights=True,
            start_from_epoch=3
        )

        csv_logger_callback = keras.callbacks.CSVLogger(history_csv, append=True)

        print("[TRAINING] - training started")
        for i in range(run_settings["repeat"]):
            history = model.fit(
                (source.padded, target.padded), target.padded_shift_one,
                batch_size=run_settings["batch_size"],
                epochs = run_settings["epochs"],
                initial_epoch=initial_epoch,
                validation_data=((val_source.padded, val_target.padded), val_target.padded_shift_one),
                callbacks=[csv_logger_callback, early_stopping_callback, model_checkpoint_callback])

            save_model(model, model_full_path)

            new_dict = join_dicts(old_dict, history.history)
            old_dict = new_dict
            with open(history_dict, 'wb') as file_pi:
                pickle.dump(new_dict, file_pi)

            plot_model_history(model_folder_path, model_name_short, title=model_name_short, metric="accuracy", show=False, save=True)
            plot_model_history(model_folder_path, model_name_short, title=model_name_short, metric="loss", show=False, save=True)

            K.clear_session()
        print("[TRAINING] - training finished")
    else:
        print("[TRAINING] - SKIPPING")

    # ---------------------------------- TESTING ------------------------------------------------------------------------
    if run_settings["test"]:
        test_source, test_target = create_new_class_dict_testing(run_settings, source, target)

        print("[TESTING] - starting testing")

        if load_best:
            print("[TESTING] - loading BEST model for testing")
            model = load_model_mine(model_best_full_path)
        else:
            print("[TESTING] - loading model for testing")
            model = load_model_mine(model_full_path)

        if run_settings["caching_in_testing"]:
            print("[TESTING] - CACHE ON")
            if run_settings["clear_testing_cache"]: tested_dict = {}
            else: tested_dict = load_cached_dict(testing_cache_filename)
        else:
            print("[TESTING] - CACHE OFF")

        # Testing Loop
        output = []
        for j in tqdm(range(len(test_source.padded))):
            i = 1
            encoder_input = np.array([test_source.padded[j]])
            if run_settings["caching_in_testing"]:
                encoder_cache_code = tuple(encoder_input[0])  # cos I can't use np array or list as a hash, [0] removes [around]
                if encoder_cache_code in tested_dict:
                    output_line = tested_dict[encoder_cache_code]
                else:
                    output_line = translate(model, encoder_input, target.maxlen)
                    tested_dict[encoder_cache_code] = output_line
            else:
                output_line = translate(model, encoder_input, target.maxlen)

            output.append(output_line)
            # try:
            #     visualise_attention(model, encoder_input, np.array([output_line]), n, h, line_num, test_source, test_target, model_full_path)
            # except Exception as e:
            #     print(f"Attention failed due to: {e}")

        if run_settings["caching_in_testing"]:
            cache_dict(tested_dict, testing_cache_filename)

        # PRETTY TESTING PRINTING
        from testing_s2s import test_translation, add_to_json

        rev_dict = test_target.create_reverse_dict(test_target.dict_chars)
        valid = list(test_target.padded.astype(np.int32))

        dict = test_translation(output, valid, rev_dict, run_settings["sep"], run_settings["mezera"], use_custom_rules=False)

        all_epochs, training_data = get_epochs_train_accuracy(history_dict)

        add_to_json(result_json_path, model_name_short, dict, run_settings["testing_samples"],
                    all_epochs, training_data, run_settings["keras_version"])
        print("[TESTING] - FINISHED")
        print(f"[TESTING] - Saved to json for model: {model_name_short}")
        print()
    else:
        print("[TESTING] - skipping testing")

    plot_model_history(model_folder_path, model_name_short, title=model_name_short, metric="accuracy", show=False, save=True)
    plot_model_history(model_folder_path, model_name_short, title=model_name_short, metric="loss", show=False, save=True)

    end_time = time.time()
    time_passed = end_time - start_time

    if "time_passed_run" not in info: info["time_passed_run"] = {}
    info["time_passed_run"][runs] = {}
    info["time_passed_run"][runs][f"sec"] = time_passed
    info["time_passed_run"][runs][f"min"] = time_passed/60
    info["time_passed_run"][runs][f"hou"] = time_passed/60/60

    save_to_json(info, os.path.join(model_folder_path, f"info.json"))

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train transformer model with custom data")
    parser.add_argument("--model-settings", default="model_settings.json",
                       help="Path to model settings JSON file")
    parser.add_argument("--compile-settings", default="model_compile_settings.json",
                       help="Path to model compile settings JSON file")
    parser.add_argument("--train-src", required=True,
                       help="Path to training source file")
    parser.add_argument("--train-tgt", required=True,
                       help="Path to training target file")
    parser.add_argument("--val-src", required=True,
                       help="Path to validation source file")
    parser.add_argument("--val-tgt", required=True,
                       help="Path to validation target file")
    parser.add_argument("--test-src", required=True,
                       help="Path to test source file")
    parser.add_argument("--test-tgt", required=True,
                       help="Path to test target file")
    parser.add_argument("--model-name", default="transformer_TLA",
                       help="Short name for the model")
    parser.add_argument("--models-path", default="/mnt/lustre/helios-home/morovkat/hiero-transformer/models",
                       help="Path to directory where models will be saved")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--class-data", default=None,
                       help="Path to cached class data (optional)")
    parser.add_argument("--new-class-dict", action="store_true",
                       help="Create new class dictionary (don't use cached data)")
    
    args = parser.parse_args()
    
    # Load model settings
    with open(args.model_settings, encoding="utf-8") as f:
        model_settings = json.load(f)
    
    # Load compile settings
    with open(args.compile_settings, encoding="utf-8") as f:
        model_compile_settings = json.load(f)
    
    # Create run_settings from arguments
    if args.class_data is None:
        args.class_data = f"processed_data_plk/{args.model_name}.plk"
    
    run_settings = {
        "all_models_path": args.models_path,
        "model_name_short": args.model_name,
        "different_output_model": False,
        "output_model_name": args.model_name,
        "train_in_file_name": args.train_src,
        "train_out_file_name": args.train_tgt,
        "val_in_file_name": args.val_src,
        "val_out_file_name": args.val_tgt,
        "test_in_file_name": args.test_src,
        "test_out_file_name": args.test_tgt,
        "sep": " ",
        "mezera": "_",
        "end_line": "\n",
        "new_model": 1,
        "new_class_dict": 1 if args.new_class_dict else 0,
        "class_data": args.class_data,
        "batch_size": args.batch_size,
        "train": True,
        "epochs": args.epochs,
        "repeat": 1,
        "load_best": False,
        "use_random_seed": False,
        "seed": 42,
        "finetune_model": False,
        "finetune_source": "",
        "finetune_tgt": "",
        "test": True,
        "testing_samples": -1,
        "use_custom_testing": False,
        "custom_test_src": "",
        "custom_test_tgt": "",
        "clear_testing_cache": 1,
        "caching_in_testing": 0,
        "version": "version",
        "keras_version": keras.__version__
    }
    
    run_model_pipeline(model_settings, model_compile_settings, run_settings)


"""
RESULTS
----------------------------------------
num_lines                      | 4
all_valid_chars                | 74
all_pred_chars                 | 67
max_all_chars                  | 75
word_accuracy                  | 0.6
character_accuracy             | 0.1216216
all_ros_levenstein             | 20.0
avg_RLEV_per_line              | 5.0
avg_RLEV_per_valid_char        | 0.2702703
avg_RLEV_per_pred_char         | 0.2985075
----------------------------------------

"""