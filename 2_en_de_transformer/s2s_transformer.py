import nltk.translate.bleu_score
import numpy as np
import random

import sys
import os
import pickle
from tqdm import tqdm
import time
import joblib
import nltk

from metrics_evaluation import metrics as m
from Data import Data
from data_preparation import *

os.environ["KERAS_BACKEND"] = "tensorflow"
import keras
from keras.utils import set_random_seed
from keras import backend as K

import tensorflow as tf

import json

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# DATA AND TRAINING SETTING
new = 0
new_class_dict = 1
caching = 0

batch_size = 15  # 256
epochs = 2
repeat = 0

print("starting transform2seq")

all_models_path = '/home/katka/Documents/Trained models/Transformer_encoder_decoder/'

model_name_short = 'transformer2_n4_h4'

# model_name_short = 'transformer2_n4_h4_sailor'
# assert model_name_short != 'transformer2_n4_h4'

finetune_model = True
finetune_source = "../data/train_src_separated.txt"
finetune_tgt = "../data/train_trl.txt"

# TESTING SETTING
samples = 30
use_custom_testing = False
custom_test_src = "../data/test_src_separated.txt"
custom_test_tgt = "../data/test_trl.txt"

# JSON SETTING
version = "original_data_test"
keras_version = "3.7.0"
result_json_path = f"json_results/transformer_results_{version}_{samples}.json"

# models = []
# for model_name in os.listdir(models_path):
#     if ".keras" in model_name:
#         model_name = model_name[:model_name.index(".keras")]
#         # print(model_name)
#         models.append(model_name)
#
# models = [model_name]

class_data = "processed_data_plk/processed_data_dict.plk"
model_full_path = os.path.join(all_models_path, model_name_short)
history_dict = f"{model_full_path}_HistoryDict"


model_settings_path = 'model_settings.json'
with open(model_settings_path, encoding="utf-8") as f:
    model_settings = json.load(f)


a = random.randrange(0, 2**32 - 1)
a = 12612638
set_random_seed(a)
print("seed = ", a)


# from model_file_2 import model_func
from model_file_2 import *  # for loading
from model_file_mine import *

def load_model_mine(model_name):
    try:
        custom_objects = {
            'EncoderLayer': EncoderLayer,
            'Encoder': Encoder,
            'DecoderLayer': DecoderLayer,
            'Decoder': Decoder,
            'TransformerModel': TransformerModel,
            'MultiHeadAttention': MultiHeadAttention,
            'PositionEmbeddingFixedWeights': PositionEmbeddingFixedWeights,
            'AddNormalization': AddNormalization,
            'FeedForward': FeedForward
        }
        return keras.models.load_model(model_name, custom_objects=custom_objects)  # KERAS 2
    except Exception as e:
        custom_objects = {
            "MyMaskingLayer" : MyMaskingLayer,
            "CustomSinePositionEncoding" : CustomSinePositionEncoding
        }
        return keras.models.load_model(model_name + ".keras", custom_objects=custom_objects)
        # return keras.models.load_model(model_name + ".keras")

def save_model(model, model_file_name):
    try:
        model.save(model_file_name)
        print("Model saved successfully the unsuccessful_attempts way")
    except Exception as e:
        model.save(model_file_name + ".keras")
        print("Model saved using KERAS 3")


# ---------------------------- DATA PROCESSING -------------------------------------------------
if new_class_dict:
    start = time.time()
    print("data preparation...")
    if finetune_model:
        source, target, val_source, val_target = prepare_data(skip_valid=False, files_val=[finetune_source, finetune_tgt], files_additional_train=[finetune_source, finetune_tgt])
    else:
        source, target, val_source, val_target = prepare_data(skip_valid=False)
    to_save_list = [source, target, val_source, val_target]
    end = time.time()
    print("preparation of data took:", end - start)
    def save_object(obj, filename):
        print("Saving data")
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
    save_start = time.time()
    save_object(to_save_list, class_data)
    save_end = time.time()
    print("saving of data took: ", save_end - save_start)
else:
    start = time.time()
    print("loading class data")
    with open(class_data, 'rb') as class_data_dict:
        source, target, val_source, val_target = pickle.load(class_data_dict)
        print("Class data loaded.")
        end = time.time()
    print("loadig took:", end - start)

# --------------------------------- MODEL ---------------------------------------------------------------------------
old_dict = get_history_dict(history_dict, new)
print("model starting...")
if new:
    print("CREATING A NEW MODEL")
    model = model_func(source.vocab_size, target.vocab_size, source.maxlen, target.maxlen, model_settings)
else:
    print("LOADING A MODEL")
    model = load_model_mine(model_full_path)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
# model.summary()
print()

# for i, layer in enumerate(model.layers):
#     print(f"Index: {i}, Name: {layer.name}, Type: {type(layer)}")


if finetune_model:
    print(target.vocab_size, len(target.dict_chars))
    extend_model_embeddings(model, source.vocab_size, target.vocab_size)
    model = adjust_output_layer(model, new_vocab_size=target.vocab_size)
# --------------------------------- TRAINING ------------------------------------------------------------------------
# existuje generator na trenovaci data
print("training")
if repeat*epochs == 0:
    print("Skipping training")
for i in range(repeat):
    history = model.fit(
        (source.padded, target.padded), target.padded_shift_one,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=((val_source.padded, val_target.padded), val_target.padded_shift_one))

    save_model(model, model_full_path)

    new_dict = join_dicts(old_dict, history.history)
    old_dict = new_dict
    with open(history_dict, 'wb') as file_pi:
        pickle.dump(new_dict, file_pi)

    K.clear_session()
print()
# try:
#     model.save(model_file_name + ".keras", save_format="tf")
# except Exception as e:
#     model.export(model_file_name)  # saving for Keras3


# ---------------------------------- TESTING ------------------------------------------------------------------------
import matplotlib.pyplot as plt

def plot_attention_weights(attention_list, input_sentence, output_sentence, n, h, line_num):
    fig = plt.figure(figsize=(16, 8))
    # print("len(attention_list):", len(attention_list))
    for i, attention in enumerate(attention_list):
        # print("i, attention:", i, len(attention))
        # print("i, attention[-1]:", i, len(attention[-1]))
        attention = attention[-1][0].numpy()  # because it's surrounded by brackets
        # print("attention[-1][0].shape :", attention.shape)
        # if i == 1:
        #     continue
        for j, attention_head in enumerate(attention):
            ax = fig.add_subplot(n, h, i*h + j + 1)

            # Plot the attention weights
            ax.matshow(attention_head[:, :len(input_sentence)],
                       cmap='viridis')
            # ax.matshow(attention_head[:-1, 1:len(input_sentence)+1],
            #            cmap='viridis')
            # ax.matshow(attention_head,
            #            cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(input_sentence)))
            ax.set_yticks(range(len(output_sentence)))

            ax.set_xticklabels(input_sentence, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(output_sentence, fontdict=fontdict)

            ax.set_xlabel(f'Head {j + 1}')
    plt.tight_layout()
    folder_name = "plots/attention/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    fig_name = model_full_path.split("/")[-1] + f"_{line_num}"
    plt.savefig(folder_name + fig_name, bbox_inches='tight')
    plt.show()
def visualise_attention(model, encoder_input_data, decoder_input_data, n, h, line_num):
    n_attention_scores = []
    for i in range(n):
        model = keras.Model(inputs=model.input,
                            outputs=[model.output, model.get_layer(f'cross_att{i}').output])
        _, attention_scores = model.call((encoder_input_data, decoder_input_data), training=False)
        n_attention_scores.append(attention_scores)

    input_sentence = []
    for token in encoder_input_data[0]:
        if token == 0:
            break
        if token == 3:  # "_"
            input_sentence.append(" ")
        # elif token == 1:  # remove the <bos> token
        #     pass
        # elif token == 2:  # remove the <eos> token
        #     pass
        else:
            input_sentence.append(test_source.reverse_dict[token])

    output_sentence = []
    for token in decoder_input_data[0]:
        if token == 3:  # "_"
            output_sentence.append(" ")
        # elif token == 1:  # remove the <bos> token
        #     pass
        else:
            output_sentence.append(test_target.reverse_dict[token])

    # placeholder code before i fill in the text
    # input_sentence = [str(i) for i in range(encoder_input_data.shape[1])]
    # output_sentence = [str(i) for i in range(decoder_input_data.shape[1])]

    plot_attention_weights(n_attention_scores, input_sentence, output_sentence, n, h, line_num)

def translate(model, encoder_input, output_maxlen, line_num):
    output_line = [1]
    # i = 1
    i = 0
    while i < output_maxlen:
        prediction = model.call((encoder_input, np.array([output_line])), training=False)  # enc shape: (1, maxlen), out shape: (1, j)
        # next_token_probs = prediction[0, -1, :]  # Prediction is shape (1, i, 63)
        next_token_probs = prediction[0, i, :]  # prediction has the whole sentence every time
        # next_token = np.random.choice(len(next_token_probs), p=next_token_probs)
        next_token = np.argmax(next_token_probs)
        if next_token == 0:
            break
        # Update the output sequence with the sampled token
        output_line.append(next_token)
        i += 1
    # try:
    #     visualise_attention(model, encoder_input, np.array([output_line]), n, h, line_num)
    # except Exception as e:
    #     print(f"Attention failed due to: {e}")
    return output_line

print("Testing data preparation")
test_source = Data(sep, mezera, end_line)
test_target = Data(sep, mezera, end_line)

if use_custom_testing:
    test_in_file_name = custom_test_src
    test_out_file_name = custom_test_tgt

with open(test_in_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_source.file = f.read()
    f.close()
with open(test_out_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_target.file = f.read()
    f.close()

test_source.dict_chars = source.dict_chars
if samples == -1:
    x_test = test_source.split_n_count(False)
else:
    x_test = test_source.split_n_count(False)[:samples]
test_source.padded = test_source.padding(x_test, source.maxlen)

test_target.dict_chars = target.dict_chars
if samples == -1:
    y_test = test_target.split_n_count(False)
else:
    y_test = test_target.split_n_count(False)[:samples]

test_target.padded = test_target.padding(y_test, target.maxlen)
test_target.padded_shift = test_target.padding_shift(y_test, target.maxlen)

valid = list(test_target.padded.astype(np.int32))

assert len(x_test) == len(y_test)
del x_test, y_test

output = []

print("Testing...")
test_source.create_reverse_dict(test_source.dict_chars)
rev_dict = test_target.create_reverse_dict(test_target.dict_chars)


# for model_name in models:
if True:
    model_full_path = os.path.join(all_models_path, model_name_short)
    history_dict = model_full_path + '_HistoryDict'
    testing_cache_filename = model_full_path + '_TestingCache'
    print(model_full_path)

    model = load_model_mine(model_full_path)

    if caching:
        print("Caching is ON")
        tested_dict = load_cached_dict(testing_cache_filename)
    else:
        print("Caching is OFF")
    # Testing Loop
    for j in tqdm(range(len(test_source.padded))):
        i = 1
        encoder_input = np.array([test_source.padded[j]])
        if caching:
            encoder_cache_code = tuple(encoder_input[0])  # cos I can't use np array or list as a hash, [0] removes [around]
            if encoder_cache_code in tested_dict:
                output_line = tested_dict[encoder_cache_code]
            else:
                output_line = translate(model, encoder_input, target.maxlen, j)
                tested_dict[encoder_cache_code] = output_line
        else:
            output_line = translate(model, encoder_input, target.maxlen, j)
            # print(output_line)
        output.append(output_line)
    # End Testing Loop
    if caching:
        cache_dict(tested_dict, testing_cache_filename)

    # PRETY TESTING PRINTING

    from testing_s2s import test_translation, add_to_json

    dict = test_translation(output, valid, rev_dict, sep, mezera, use_custom_rules=False)


    def get_epochs_train_accuracy(history_dict):
        with open(history_dict, 'rb') as file_pi:
            history = pickle.load(file_pi)
            epochs = len(history['accuracy'])
            results = {
                "train_accuracy": history['accuracy'][-1],
                "val_accuracy": history['val_accuracy'][-1],
                "train_loss": history['loss'][-1],
                "val_loss": history['val_loss'][-1]
            }
        return epochs, results


    all_epochs, training_data = get_epochs_train_accuracy(history_dict)

    add_to_json(result_json_path, model_name_short, dict, samples,
                all_epochs, training_data, keras_version)
    print(f"Saved to json for model:{model_name_short}")
    print()