# Script to load model results and plot RLEV metrics boxplots grouped by model number

import os
import json
import re
import matplotlib.pyplot as plt
from collections import defaultdict

def extract_model_info(model_name):
    match = re.match(r"model_7_(\d+)_D_run_(\d+)", model_name)
    return (int(match.group(1)), int(match.group(2))) if match else (None, None)

def load_results(base_path, metrics, results_bool=True):
    results = defaultdict(list)
    epochs = None
    for folder in os.listdir(base_path):
        if '_D_' not in folder:
            continue

        folder_path = os.path.join(base_path, folder)
        json_path = os.path.join(folder_path, 'testing_version0_samples_-1.json')
        if not os.path.isfile(json_path):
            continue

        with open(json_path, 'r') as f:
            data = json.load(f)
            for model_name, model_data in data.items():
                model_num, run_num = extract_model_info(model_name)
                if model_num is None:
                    continue
                if results_bool:
                    results[model_num].append({
                        'run': run_num,
                        metrics: model_data['results'][metrics]
                    })
                else:
                    results[model_num].append({
                        'run': run_num,
                        metrics: model_data[metrics]
                    })
                epochs = model_data['all_epochs']
    value = list(results.keys())[0]
    lengh = len(results[value])
    for key in results.keys():
        assert len(results[key]) == lengh
    return results, lengh, epochs

def plot_boxplots(path, results, metrics, lengh, epochs):
    model_nums = sorted(results.keys())

    data = [
        [entry[metrics] for entry in results[model_num]]
        for model_num in model_nums
    ]

    plt.figure()
    plt.boxplot(data, tick_labels=[f'7_{m}' for m in model_nums])
    plt.title(f'{metrics} Boxplot, runs: {lengh}, epochs: {epochs}')
    plt.xlabel('Model Number')
    plt.ylabel(metrics)
    plt.grid()
    plt.savefig(os.path.join(path, f'{metrics}_model_{[num for num in model_nums]}.png'))
    plt.show()

# Example usage:
metrics = 'avg_RLEV_per_valid_char'
path = '/home/katka/models/models'
results, lengh, epochs = load_results(path, metrics)
plot_boxplots(path, results, metrics, lengh, epochs)