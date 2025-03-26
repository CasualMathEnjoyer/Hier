import os
import json
import matplotlib.pyplot as plt

def get_latest_version_json(model_folder_path, samples):
    latest_version = -1
    latest_file = None

    for file in os.listdir(model_folder_path):
        if file.endswith(f"samples_{samples}.json") or (samples == -1 and "samples_-1.json" in file):
            try:
                version_num = int(file.split('_')[1].replace('version', ''))
                if version_num > latest_version:
                    latest_version = version_num
                    latest_file = file
            except ValueError:
                continue

    return latest_file, latest_version

def plot_latest_versions(models_path, samples, results, metric, show_epochs=False, show_version=False):
    colors = {'small': 'blue', 'large': 'orange', '3': 'green'}

    plt.figure(figsize=(12, 6))

    model_data = {}
    all_x_values = set()

    for selection in colors.keys():
        for model_folder in os.listdir(models_path):
            if "stats_run" in model_folder:
                continue

            print(model_folder)
            if model_folder.split("_")[1] != selection:
                continue



            model_num = int(model_folder.split('_')[-1])

            model_folder_path = os.path.join(models_path, model_folder)
            if not os.path.isdir(model_folder_path):
                continue

            latest_file, version = get_latest_version_json(model_folder_path, samples)
            if latest_file:
                with open(os.path.join(model_folder_path, latest_file), 'r') as f:
                    data = json.load(f)
                    for _, model_info in data.items():
                        ros_value = model_info[results][metric]
                        epochs = model_info["all_epochs"]
                        model_data.setdefault(selection, []).append((model_num, ros_value, epochs, version))
                        all_x_values.add(model_num)

    for selection, color in colors.items():
        sorted_data = sorted(model_data.get(selection, []), key=lambda x: x[0])
        x_values, y_values, epochs_values, versions = zip(*sorted_data) if sorted_data else ([], [], [], [])

        plt.scatter(x_values, y_values, color=color, label=selection)
        plt.plot(x_values, y_values, color=color)

        if show_epochs:
            for x, y, epoch in zip(x_values, y_values, epochs_values):
                plt.text(x, y, str(epoch), fontsize=9, ha='right', va='bottom')

        if show_version:
            for x, y, version in zip(x_values, y_values, versions):
                plt.text(x, y, str(version), fontsize=9, ha='right', va='bottom')

    plt.xlabel('Dataset size (in thousands of sentences)')
    plt.ylabel(metric)
    plt.title(f'{metric} for Latest Versions (Samples {samples})')
    plt.xticks(sorted(all_x_values))
    plt.grid()
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(models_path, f'scatter_latest_{metric}_samples_{samples}_{show_epochs}.png'))
    plt.show()


if __name__ == '__main__':
    models_path = "/home/katka/models/models"
    run = 1
    samples = -1

    results = "results"
    metric = 'avg_RLEV_per_valid_char'

    # results = "training_data"
    # metric = 'val_accuracy'

    plot_latest_versions(models_path, samples, results, metric, show_epochs=False, show_version=False)
    plot_latest_versions(models_path, samples, results, metric, show_epochs=True, show_version=False)
