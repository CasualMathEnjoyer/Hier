import matplotlib.pyplot as plt
import pickle
import os

root = "plots/"
os.makedirs(root, exist_ok=True)

colors = ['r', 'g', 'c', 'b', 'm', 'y', 'k', 'orange', 'pink']
counter = 0
def plot_history(ax, history_dict, metric, title, ylabel):
    global counter
    try:
        with open(history_dict, 'rb') as file_pi:
            history = pickle.load(file_pi)
            ax.plot(history[metric], linestyle=':', label=title, color=colors[counter]) # i want this
            ax.plot(history[f'val_{metric}'], color=colors[counter])  # and this to have the same color
            ax.set_ylabel(ylabel)
            ax.set_xlabel('Epochs')
            ax.grid(True)
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

def filter_models(dict_list, models_to_exclude):
    return [model for model in dict_list if model.split('/')[-1] not in models_to_exclude]

# Example usage:
model = "my_model4"
model = "models_LSTM3/models_LSTM2"
# model = 'models_endocer_bin_zaloha'
models = f'/home/katka/Documents/{model}/'
old = True

dict_list = get_history_dicts(models, old)
print("Available models:", dict_list)

# Specify models to exclude (example, adjust accordingly)
models_to_exclude = ["attention_is_all_base_HistoryDict",
                     "transformer6_asm_n4_h6_HistoryDict",
                     "transformer5_asm_n4_HistoryDict",]

# models_to_exclude = ["t2b_emb64_ff128_h16_data/t2b_emb64_ff128_h16_HistoryDict"]
filtered_dict_list = filter_models(dict_list, models_to_exclude)

fig, axs = plt.subplots(2, 1, figsize=(10, 10))
# fig.suptitle("Model Training History", fontsize=14)

title_dict = {
    "transform2seq_LSTM_em32_dim64": 1,
    "transform2seq_LSTM_em64_dim64": 2,
    "transform2seq_LSTM_em64_dim128": 3,
    "transform2seq_LSTM_em64_dim256": 4,
    "transform2seq_LSTM_em128_dim256": 5,
    "transform2seq_LSTM_64_dim512": 6,
    "transform2seq_LSTM_em128_dim512": 7,

    "transformer1_n2_h2": 1,
    "transformer2_n4_h4": 2,
    "transformer_asmol": 3,
    "transformer4_asm_ff512": 4,
    "transformer5_asm_n4": 5,
    "transformer6_asm_n4_h6": 6,
    "transformer5_2_2_d_model256": 7,
    "attention_is_all_base": 8,

    "t2b_emb128_data/t2b_emb128": 3,
    "t2b_emb64_h4_data/t2b_emb64_h4": 4,
    "t2b_emb64_data/t2b_emb64": 2,
    "t2b_emb64_ff128_data/t2b_emb64_ff128": 6,
    "t2b_emb64_ff256_h4_data/t2b_emb64_ff256_h4": 8,
    "t2b_emb32_data/t2b_emb32": 1,
    "t2b_emb64_ff128_h4_data/t2b_emb64_ff128_h4": 7,
    "t2b_emb64_ff128_h16_data/t2b_emb64_ff128_h16": 9,
    "t2b_emb64_h4_2_data/t2b_emb64_h4_2": 5,
}


models_names = []
# counter = 5
# for model_dict_name in models_to_exclude:
for model_dict_name in filtered_dict_list:
    # if model_dict_name.split('/')[0] == 'models_LSTM2':
    #     continue
    if counter != 0 and counter != 3 and counter != 1:
        counter += 1
        continue
    model_dict_path = os.path.join(models, model_dict_name)
    model_label = title_dict[model_dict_name[:-12]]
    models_names.append(str(model_label)+" training")
    models_names.append(str(model_label)+" validation")
    # models_names.append(str(model_label))
    plot_history(axs[0], model_dict_path, 'loss', f'{model_label}', 'Loss')
    plot_history(axs[1], model_dict_path, 'accuracy', f'{model_label}', 'Accuracy')
    counter += 1

axs[0].set_title("Loss")
axs[1].set_title("Accuracy")
axs[0].legend(models_names)

print(len(models_names))

# plot_filename = f"{root}/{model}_combined_plot.pdf"
# plot_filename = f"{root}/Encoder_only_val_combined_plot.pdf"
# plot_filename = f"{root}/Encoder_only_val_combined_plot_selected.pdf"
# plot_filename = f"{root}/{model}_combined_plot_excluded.pdf"
plot_filename = f"{root}/models_LSTM3_combined_plot.pdf"
plot_filename = f"{root}/models_LSTM3_combined_plot_selected.pdf"
plt.tight_layout()  # Adjust layout to make room for the title
plt.savefig(plot_filename)
plt.show()
