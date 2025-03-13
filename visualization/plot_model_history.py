import matplotlib.pyplot as plt
import pickle
import os
import numpy as np

def plot_model_history(model_folder, model_name, title='Model', metric="accuracy", show=True, save=False):
    history_dict_path = os.path.join(model_folder, f'{model_name}_HistoryDict')

    if metric not in ["accuracy", "loss"]:
        raise ValueError("Metric must be either 'accuracy' or 'loss'")

    try:
        with open(history_dict_path, 'rb') as file:
            history = pickle.load(file)

        epochs = len(history[metric])
        tick_interval = max(1, epochs // 10)

        plt.figure(figsize=(8, 5))
        plt.plot(history[metric], label="Train")
        plt.plot(history[f"val_{metric}"], label="Validation")
        plt.title(f"{title} {metric.capitalize()}")
        plt.xlabel("Epoch")
        plt.xticks(np.arange(0, epochs, tick_interval))
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        if save: plt.savefig(os.path.join(model_folder, f'{model_name}_{metric}.png'))
        if show: plt.show()
    except Exception as e:
        print(f"Error loading history: {e}")


# Example usage
if __name__ == "__main__":
    models = '/home/katka/PycharmProjects/Hier/models'
    model_name = 'model_testing'
    model_folder = os.path.join(models, model_name)  # if each model has its own folder
    plot_model_history(model_folder, model_name, title=model_name, metric="accuracy", save=True)
    plot_model_history(model_folder, model_name, title=model_name, metric="loss", save=True)

    models = '/home/katka/Documents/Trained models/Transformer_encoder_decoder'
    model_name = 'transformer4_asm_ff512'
    model_folder = models  # if models are as keras files in one folder
    plot_model_history(model_folder, model_name, title=model_name, metric="accuracy", save=True)
    plot_model_history(model_folder, model_name, title=model_name, metric="loss", save=True)