import matplotlib.pyplot as plt
import matplotlib
import pickle
matplotlib.use('TkAgg')
import os

root = "plots/"
def plot_accuracy_history(plt, model_nums, history_dict, save=False):
    try:
        with open(history_dict, 'rb') as file_pi:
            history = pickle.load(file_pi)
            # Plot training & validation accuracy values
            plt.plot(history['accuracy'])
            plt.plot(history['val_accuracy'])
            plt.set_title(f'{model_nums} accuracy')
            plt.set_ylabel('Accuracy')
            plt.set_xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            # Save the plot with the model name
            plot_filename = f"{root}{model_nums}_accuracy_plot.png"
            if save:
                plt.savefig(plot_filename)

            # plt.show()
            return plt
    except Exception as e:
        print(e)

def plot_loss_history(plt, model_nums, history_dict, save=False):
    try:
        with open(history_dict, 'rb') as file_pi:
            history = pickle.load(file_pi)
            # Plot training & validation loss values
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.set_title(f'{model_nums} loss')
            plt.set_ylabel('Loss')
            plt.set_xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            # Save the plot with the model name
            plot_filename = f"{root}{model_nums}_loss_plot.png"

            if save:
                plt.savefig(plot_filename)

            # plt.show()
            return plt
    except Exception as e:
        print(e)
# Example usage:
# Assuming you have model_name and history_dict variables already defined
# models = 'C:/Users/katka/OneDrive/Dokumenty/models_LSTM'
#
# mm_list = ["em32_dim64", "em64_dim64", "em64_dim128", "em64_dim128", "em64_dim256",
#            "em64_dim512", "em128_dim512", "em128_dim256"]

def get_folder_names(folder_path):
    folder_names = []
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            folder_names.append(item)
    return folder_names

models = 'C:/Users/katka/OneDrive/Dokumenty/model_10'

mm_list = get_folder_names(models)

fig, axs = plt.subplots(2, 3, figsize=(10, 6))
# fig.suptitle("Title centered above all subplots", fontsize=14)

for i, model_nums in enumerate(mm_list):
    # model_file_name = models + f"/transform2seq_LSTM_{model_nums}"
    model_file_name = models + f"/{model_nums}"
    save = False
    accuracy = plot_accuracy_history(axs[0, i], model_nums, model_file_name + '_HistoryDict', save)
    loss = plot_loss_history(axs[1, i], model_nums, model_file_name + '_HistoryDict', save)

for ax in axs.flatten():
    ax.grid(True)  # Add grid to all axes

plot_filename = f"{root}six_plots"
plt.savefig(plot_filename)

plt.tight_layout()
plt.show()