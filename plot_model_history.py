import matplotlib.pyplot as plt
import matplotlib
import pickle
matplotlib.use('TkAgg')

def plot_accuracy_history(model_nums, history_dict):
    try:
        with open(history_dict, 'rb') as file_pi:
            history = pickle.load(file_pi)
            # Plot training & validation accuracy values
            plt.plot(history['accuracy'])
            plt.plot(history['val_accuracy'])
            plt.title(f'{model_nums} accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            # Save the plot with the model name
            plot_filename = f"{model_nums}_accuracy_plot.png"
            plt.savefig(plot_filename)

            plt.show()
    except Exception as e:
        pass

def plot_loss_history(model_nums, history_dict):
    try:
        with open(history_dict, 'rb') as file_pi:
            history = pickle.load(file_pi)
            # Plot training & validation loss values
            plt.plot(history['loss'])
            plt.plot(history['val_loss'])
            plt.title(f'{model_nums} loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
            # Save the plot with the model name
            plot_filename = f"{model_nums}_loss_plot.png"
            plt.savefig(plot_filename)

            plt.show()
    except Exception as e:
        pass
# Example usage:
# Assuming you have model_name and history_dict variables already defined
models = 'C:/Users/katka/OneDrive/Dokumenty/models'

mm_list = ["em32_dim64", "em64_dim64", "em64_dim128", "em64_dim128", "em64_dim256",
           "em64_dim512", "em128_dim512", "em128_dim256"]
for model_nums in mm_list:
    model_file_name = models + f"/transform2seq_LSTM_{model_nums}"
    # plot_accuracy_history(model_nums, model_file_name + '_HistoryDict')
    plot_loss_history(model_nums, model_file_name + '_HistoryDict')
