import os
import json
import time
import pickle
import random
from tqdm import tqdm

from keras import backend as K
from keras.utils import set_random_seed


# from model_file_2 import *  # for loading
from model_file_mine import *
from model_function import save_model, load_model_mine, translate, get_epochs_train_accuracy, test_gpus
from Data import Data
from data_preparation import get_history_dict, join_dicts, load_cached_dict, cache_dict, create_new_class_dict, load_class_data, create_new_class_dict_testing

print("Starting transform2seq")

# settings json loading ---------------------------------------
model_settings_path = 'model_settings.json'
with open(model_settings_path, encoding="utf-8") as f:
    model_settings = json.load(f)

model_compile_settings_path = 'model_compile_settings.json'
with open(model_compile_settings_path, encoding="utf-8") as f:
    model_compile_settings = json.load(f)

run_settings_path = 'run_settings.json'
with open(run_settings_path, encoding="utf-8") as f:
    run_settings = json.load(f)
# ---------------------------------------------------------------

# Access loaded settings
new = run_settings["new_model"]
new_class_dict = run_settings["new_class_dict"]
caching_in_testing = run_settings["caching_in_testing"]
batch_size = run_settings["batch_size"]
epochs = run_settings["epochs"]
repeat = run_settings["repeat"]
all_models_path = run_settings["all_models_path"]
model_name_short = run_settings["model_name_short"]
finetune_model = run_settings["finetune_model"]
finetune_source = run_settings["finetune_source"]
finetune_tgt = run_settings["finetune_tgt"]
testing_samples = run_settings["testing_samples"]
use_custom_testing = run_settings["use_custom_testing"]
custom_test_src = run_settings["custom_test_src"]
custom_test_tgt = run_settings["custom_test_tgt"]
version = run_settings["version"]
keras_version = run_settings["keras_version"]
class_data = run_settings["class_data"]

train_in_file_name = run_settings["train_in_file_name"]
train_out_file_name = run_settings["train_out_file_name"]
val_in_file_name = run_settings["val_in_file_name"]
val_out_file_name = run_settings["val_out_file_name"]
test_in_file_name = run_settings["test_in_file_name"]
test_out_file_name = run_settings["test_out_file_name"]

# dynamicaly change for namings
result_json_path = f"json_results/transformer_results_{version}_{testing_samples}.json"
model_full_path = os.path.join(all_models_path, model_name_short)
history_dict = f"{model_full_path}_HistoryDict"


if run_settings["use_random_seed"]: a = random.randrange(0, 2**32 - 1)
else: a = run_settings["seed"]
set_random_seed(a)

os.environ["KERAS_BACKEND"] = "tensorflow"
test_gpus()

# ---------------------------- DATA PROCESSING -------------------------------------------------
if run_settings["new_class_dict"]: source, target, val_source, val_target = create_new_class_dict(run_settings)
else: source, target, val_source, val_target = load_class_data(run_settings)

# --------------------------------- MODEL ---------------------------------------------------------------------------
old_dict = get_history_dict(history_dict, new)
print("[MODEL] - starting")
if new: model = model_func(source.vocab_size, target.vocab_size, source.maxlen, target.maxlen, model_settings)
else: model = load_model_mine(model_full_path)

model.compile(optimizer=model_compile_settings['optimizer'],
              loss=model_compile_settings['loss'],
              metrics=model_compile_settings['metrics'])

print("[MODEL] - COMPILED")
# model.summary()

if finetune_model:
    extend_model_embeddings(model, source.vocab_size, target.vocab_size)
    model = adjust_output_layer(model, new_vocab_size=target.vocab_size)
    print("[MODEL] - fine-tuning preparation finished")
else:
    print("[MODEL] - SKIPPING fine-tuning")

# --------------------------------- TRAINING ------------------------------------------------------------------------
if run_settings["train"]:
    print("[TRAINING] - training started")
    for i in range(repeat):
        history = model.fit(
            (source.padded, target.padded), target.padded_shift_one,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=((val_source.padded, val_target.padded), val_target.padded_shift_one))

        save_model(model, model_full_path)

        new_dict = join_dicts(old_dict, history.history)
        old_dict = new_dict
        with open(history_dict, 'wb') as file_pi:
            pickle.dump(new_dict, file_pi)

        K.clear_session()
    print("[TRAINING] - training finished")
else:
    print("[TRAINING] - SKIPPING")


# ---------------------------------- TESTING ------------------------------------------------------------------------
if run_settings["test"]:
    test_source, test_target = create_new_class_dict_testing(run_settings, source, target)

    print("[TESTING] - starting testing")
    model_full_path = os.path.join(all_models_path, model_name_short)
    history_dict = model_full_path + '_HistoryDict'
    testing_cache_filename = model_full_path + '_TestingCache'
    print(f"[TESTING] - {model_full_path}")

    model = load_model_mine(model_full_path)

    if run_settings["caching_in_testing"]:
        print("[TESTING] - CACHE ON")
        tested_dict = load_cached_dict(testing_cache_filename)
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
                output_line = translate(model, encoder_input, target.maxlen, j)
                tested_dict[encoder_cache_code] = output_line
        else:
            output_line = translate(model, encoder_input, target.maxlen, j)
        output.append(output_line)

    if run_settings["caching_in_testing"]:
        cache_dict(tested_dict, testing_cache_filename)

    # PRETTY TESTING PRINTING
    from testing_s2s import test_translation, add_to_json

    rev_dict = test_target.create_reverse_dict(test_target.dict_chars)
    valid = list(test_target.padded.astype(np.int32))

    dict = test_translation(output, valid, rev_dict, run_settings["sep"], run_settings["mezera"], use_custom_rules=False)

    all_epochs, training_data = get_epochs_train_accuracy(history_dict)

    add_to_json(result_json_path, model_name_short, dict, testing_samples,
                all_epochs, training_data, keras_version)
    print("[TESTING] - FINISHED")
    print(f"[TESTING] - Saved to json for model: {model_name_short}")
    print()
else:
    print("[TESTING] - SKIPPING Testing")