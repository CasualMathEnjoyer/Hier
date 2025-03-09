import json
import os
import time

from s2s_transformer_pipeline import run_model_pipeline

model_settings_path = 'model_settings.json'
with open(model_settings_path, encoding="utf-8") as f:
    model_settings = json.load(f)

model_compile_settings_path = 'model_compile_settings.json'
with open(model_compile_settings_path, encoding="utf-8") as f:
    model_compile_settings = json.load(f)

run_settings_path = 'run_settings.json'
with open(run_settings_path, encoding="utf-8") as f:
    run_settings = json.load(f)

custom_training_datasets = '/home/katka/PycharmProjects/Hier/data/ramses_smaller'

from data_processing.plot_model_vs_testing import save_plot

for iteration in range(3, 6):

    start_time = time.time()
    t = {}
    for i in range(1, 68):
        model_time = time.time()

        n = 1000 * i

        print("RUNNING MODEL: ", n)
        # initial model training

        run_settings["train_in_file_name"] = os.path.join(custom_training_datasets, f"src-sep-train_{n}.txt")
        run_settings["train_out_file_name"] = os.path.join(custom_training_datasets, f"tgt-train_{n}.txt")

        run_settings["model_name_short"] = f'model1_samples_{n}'
        run_settings["class_data"] = f"processed_data_plk/processed_data_dict_{n}.plk"
        if os.path.exists(run_settings["class_data"]): run_settings["new_class_dict"] = 0

        run_model_pipeline(model_settings, model_compile_settings, run_settings)

        end_time = time.time()
        tt = end_time - model_time

        t[str(n)] = tt
        whole_time = end_time - start_time

        last_model = f'last_model_5_epochs_run_{iteration}.txt'
        with open(last_model, 'w') as f:
            json.dump({'last_model': n, 'times': t, 'whole_time': whole_time, "whole_time_minutes": whole_time/60, "whole_time_hours": whole_time/60/60,}, f, indent=4)

    try:
        save_plot(iteration, iteration-1, 100)
    except Exception:
        print("EXCEPTION")
        pass