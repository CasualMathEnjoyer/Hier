import matplotlib.pyplot as plt
import pickle
import os

import matplotlib
matplotlib.use('TkAgg')

root = "plots/"
os.makedirs(root, exist_ok=True)

def plot_accuracy_history(ax, model_nums, history_dict, save=False):
    try:
        with open(history_dict, 'rb') as file_pi:
            history = pickle.load(file_pi)
            # Plot training & validation accuracy values
            ax.plot(history['accuracy'])
            ax.plot(history['val_accuracy'])
            ax.set_title(f'{model_nums} accuracy')
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Epoch')
            ax.legend(['Train', 'Validation'], loc='upper left')
            # Save the plot with the model name
            plot_filename = f"{root}{model_nums}_accuracy_plot.png"
            if save:
                plt.savefig(plot_filename)
    except Exception as e:
        print(e)

def plot_loss_history(ax, model_nums, history_dict, save=False):
    try:
        with open(history_dict, 'rb') as file_pi:
            history = pickle.load(file_pi)
            # Plot training & validation loss values
            ax.plot(history['loss'])
            ax.plot(history['val_loss'])
            ax.set_title(f'{model_nums} loss')
            ax.set_ylabel('Loss')
            ax.set_xlabel('Epoch')
            ax.legend(['Train', 'Validation'], loc='upper left')
            # Save the plot with the model name
            plot_filename = f"{root}{model_nums}_loss_plot.png"

            if save:
                plt.savefig(plot_filename)
    except Exception as e:
        print(e)

def get_folder_names(folder_path):
    folder_names = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

def get_history_dicts(folder_path, old=False):
    dicts = []
    for item in os.listdir(folder_path):
        if old:
            path_to_file = os.path.join(folder_path, item)
            if os.path.isdir(path_to_file):
                for file in os.listdir(path_to_file):
                    if "HistoryDict" in file:
                        dicts.append(item + "/" + file)
        if "HistoryDict" in item:
            dicts.append(item)
    return dicts

# Example usage:
# model = "my_model4"
# model = "models_LSTM"
# model = "models_endocer_bin_zaloha"
model = "models_LSTM3/models_LSTM2"
models = f'/home/katka/Documents/{model}/'
old = True
# mm_list = get_folder_names(models)
# print(mm_list)

dict_list = get_history_dicts(models, old)
print(dict_list)

for x in range(len(dict_list) // 4):
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Model Training History", fontsize=14)

    for i, model_dict_name in enumerate(dict_list[x*4:x*4 + 4]):
        row = i // 4
        col = i % 4
        # model_file_name = os.path.join(models, model_nums, f'{model_nums}_HistoryDict')
        model_dict_path = os.path.join(models, model_dict_name)  # new keras gets dicts
        save = False
        model_dict_name = model_dict_name.split('_')[-3] + "_" + model_dict_name.split('_')[-2]
        plot_accuracy_history(axs[0, col], model_dict_name, model_dict_path, save)
        plot_loss_history(axs[1, col], model_dict_name, model_dict_path, save)

    for ax in axs.flatten():
        ax.grid(True)  # Add grid to all axes

    plot_filename = f"{root}/models_LSTM3_plots_{x}.png"
    print(plot_filename)
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
    plt.savefig(plot_filename)
    plt.show()
