import matplotlib.pyplot as plt
import pickle
import os

root = "plots/"
os.makedirs(root, exist_ok=True)

def plot_all_accuracy(histories, save=False):
    plt.figure(figsize=(10, 6))
    for model_nums, history in histories.items():
        plt.plot(history['accuracy'], label=f'{model_nums} Train')
        plt.plot(history['val_accuracy'], label=f'{model_nums} Validation')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    if save:
        plot_filename = f"{root}all_models_accuracy_plot.png"
        plt.savefig(plot_filename)
    plt.show()

def plot_all_loss(histories, save=False):
    plt.figure(figsize=(10, 6))
    for model_nums, history in histories.items():
        plt.plot(history['loss'], label=f'{model_nums} Train')
        plt.plot(history['val_loss'], label=f'{model_nums} Validation')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    if save:
        plot_filename = f"{root}all_models_loss_plot.png"
        plt.savefig(plot_filename)
    plt.show()

def plot_train_accuracy(histories, save=False):
    plt.figure(figsize=(10, 6))
    for model_nums, history in histories.items():
        plt.plot(history['accuracy'], label=f'{model_nums} Train')
    plt.title('Training Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    if save:
        plot_filename = f"{root}train_accuracy_plot.png"
        plt.savefig(plot_filename)
    plt.show()

def plot_val_accuracy(histories, save=False):
    plt.figure(figsize=(10, 6))
    for model_nums, history in histories.items():
        plt.plot(history['val_accuracy'], label=f'{model_nums} Validation')
    plt.title('Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.grid(True)
    if save:
        plot_filename = f"{root}val_accuracy_plot.png"
        plt.savefig(plot_filename)
    plt.show()

def plot_train_loss(histories, save=False):
    plt.figure(figsize=(10, 6))
    for model_nums, history in histories.items():
        plt.plot(history['loss'], label=f'{model_nums} Train')
    plt.title('Training Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    if save:
        plot_filename = f"{root}train_loss_plot.png"
        plt.savefig(plot_filename)
    plt.show()

def plot_val_loss(histories, save=False):
    plt.figure(figsize=(10, 6))
    for model_nums, history in histories.items():
        plt.plot(history['val_loss'], label=f'{model_nums} Validation')
    plt.title('Validation Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    if save:
        plot_filename = f"{root}val_loss_plot.pdf"
        plt.savefig(plot_filename)
    plt.show()

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

dict_list = get_history_dicts(models, old)
print(dict_list)

histories = {}
for model_dict_name in dict_list:
    model_dict_path = os.path.join(models, model_dict_name)
    model_nums = model_dict_name.split('_')[-3] + "_" + model_dict_name.split('_')[-2]
    try:
        with open(model_dict_path, 'rb') as file_pi:
            history = pickle.load(file_pi)
            histories[model_nums] = history
    except Exception as e:
        print(e)

# Plot all accuracies and losses
plot_all_accuracy(histories, save=0)
plot_all_loss(histories, save=0)

# Plot separately train and validation data
plot_train_accuracy(histories, save=0)
plot_val_accuracy(histories, save=0)
plot_train_loss(histories, save=0)
plot_val_loss(histories, save=1)
