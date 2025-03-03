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
import matplotlib.pyplot as plt


# from model_file_2 import model_func
from model_file_2 import *  # for loading
from model_file_mine import *



model_name = 'transformer2_n4_h4'
models_path = '/home/katka/Documents/Trained models/Transformer_encoder_decoder/'

samples = -1
version = f"{samples}_sample_model4"
keras_version = "3.7.0"
result_json_path = f"transformer_results_{version}.json"

caching = 1


class_data = "processed_data_dict.plk"

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
    fig_name = model_file_name.split("/")[-1] + f"_{line_num}"
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




# LOADING CLASS DATA
start = time.time()
print("loading class data")
with open(class_data, 'rb') as class_data_dict:
    source, target, val_source, val_target = pickle.load(class_data_dict)
    print("Class data loaded.")
    end = time.time()
print("loadig took:", end - start)






print("Testing data preparation")

# test_in_file_name = "../data/sailor_separated_rsc.txt"
# test_out_file_name = "../data/sailor_test_trl.txt"

test_in_file_name = "../data/train_src_separated.txt"
test_out_file_name = "../data/train_trl.txt"

test_source = Data(sep, mezera, end_line)
test_target = Data(sep, mezera, end_line)
#
# test_source.file = "../data/sailor_separated_rsc.txt"
# test_target.file = "../data/sailor_test_trl.txt"

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

    model_file_name = os.path.join(models_path, model_name)
    history_dict = model_file_name + '_HistoryDict'
    testing_cache_filename = model_file_name + '_TestingCache'
    print(model_file_name)

    model = load_model_mine(model_file_name)

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

    dict = test_translation(output, valid, rev_dict, sep, mezera)


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

    add_to_json(result_json_path, model_name, dict, samples,
                all_epochs, training_data, keras_version)
    print(f"Saved to json for model:{model_name}")
    print()