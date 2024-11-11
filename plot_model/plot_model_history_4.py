import matplotlib.pyplot as plt
import pickle
import os
from matplotlib.backends.backend_pdf import PdfPages

import matplotlib

matplotlib.use('TkAgg')

root = "plots/"
os.makedirs(root, exist_ok=True)

# Example model name mapping dictionary
model_mapping = {
    'model1': 'Model One',
    'model2': 'Model Two',
    # Add other mappings here
}


def plot_accuracy_history(ax, model_name, history_dict):
    try:
        with open(history_dict, 'rb') as file_pi:
            history = pickle.load(file_pi)
            # Plot training & validation accuracy values
            ax.plot(history['accuracy'])
            ax.plot(history['val_accuracy'])
            ax.set_title(f'{model_mapping.get(model_name, model_name)} accuracy')
            ax.set_ylabel('Accuracy')
            ax.set_xlabel('Epoch')
            ax.legend(['Train', 'Validation'], loc='upper left')
    except Exception as e:
        print(e)


def plot_loss_history(ax, model_name, history_dict):
    try:
        with open(history_dict, 'rb') as file_pi:
            history = pickle.load(file_pi)
            # Plot training & validation loss values
            ax.plot(history['loss'])
            ax.plot(history['val_loss'])
            ax.set_title(f'{model_mapping.get(model_name, model_name)} loss')
            ax.set_ylabel('Loss')
            ax.set_xlabel('Epoch')
            ax.legend(['Train', 'Validation'], loc='upper left')
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
model = "models_endocer_bin_zaloha"
models = f'/home/katka/Documents/{model}/'
old = True
# mm_list = get_folder_names(models)
# print(mm_list)

dict_list = get_history_dicts(models, old)
print(dict_list)

pdf_filename = f"{root}/{model}_plots.pdf"
with PdfPages(pdf_filename) as pdf:
    for model_dict_name in dict_list:
        fig, ax = plt.subplots(figsize=(10, 5))
        model_dict_path = os.path.join(models, model_dict_name)  # new keras gets dicts
        model_name = model_dict_name.split('_')[-3] + "_" + model_dict_name.split('_')[-2]

        # Plot accuracy
        plot_accuracy_history(ax, model_name, model_dict_path)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
        pdf.savefig(fig)  # saves the current figure into a pdf page
        plt.show()  # Show the plot
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        # Plot loss
        plot_loss_history(ax, model_name, model_dict_path)
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust layout to make room for the title
        pdf.savefig(fig)  # saves the current figure into a pdf page
        plt.show()  # Show the plot
        plt.close(fig)

print(f"Plots saved to {pdf_filename}")
