import numpy as np
import random
import keras

import sys
import os
import pickle
from tqdm import tqdm

from keras.utils import set_random_seed
from keras.utils import to_categorical
from keras import backend as K

from metrics_evaluation import metrics as m
from data_file import Data
from data_preparation import *

new = 0
caching = 1

batch_size = 1  # 256
epochs = 20
repeat = 0

print("starting transform2seq")

model_file_name = "models/transform_smol_delete"
history_dict = model_file_name + '_HistoryDict'
testing_cache_filename = model_file_name + '_TestingCache'
print(model_file_name)

h = 2          # Number of self-attention heads
d_k = 32       # Dimensionality of the linearly projected queries and keys
d_v = 32       # Dimensionality of the linearly projected values
d_ff = 512    # Dimensionality of the inner fully connected layer
d_model = 512  # Dimensionality of the model sub-layers' outputs
n = 2          # Number of layers in the encoder stack
params = h, d_k, d_v, d_ff, d_model, n

a = random.randrange(0, 2**32 - 1)
a = 12612638
set_random_seed(a)
print("seed = ", a)


from model_file_2 import model_func
from model_file_2 import *  # for loading

def load_model_mine(model_name):
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
    return keras.models.load_model(model_name, custom_objects=custom_objects)

def save_model_info(model_name, ):
    pass

# ---------------------------- DATA PROCESSING -------------------------------------------------
source, target, val_source, val_target = prepare_data()

# --------------------------------- MODEL ---------------------------------------------------------------------------
old_dict = get_history_dict(history_dict, new)
print("model starting...")
if new:
    print("CREATING A NEW MODEL")
    model = model_func(source.vocab_size, target.vocab_size, source.maxlen, target.maxlen, params)
else:
    print("LOADING A MODEL")
    model = load_model_mine(model_file_name)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
print()

# exit()
# --------------------------------- TRAINING ------------------------------------------------------------------------
print("training")
for i in range(repeat):
    history = model.fit(
        (source.padded, target.padded), target.padded_shift_one,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=((val_source.padded, val_target.padded), val_target.padded_shift_one))

    model.save(model_file_name)

    new_dict = join_dicts(old_dict, history.history)
    old_dict = new_dict
    with open(history_dict, 'wb') as file_pi:
        pickle.dump(new_dict, file_pi)

    K.clear_session()
print()




# ---------------------------------- TESTING ------------------------------------------------------------------------
print("testing...")
print("testing data preparation")

test_x = Data(sep, mezera, end_line)
with open(test_in_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_x.file = f.read()
    f.close()
test_y = Data(sep, mezera, end_line)
with open(test_out_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_y.file = f.read()
    f.close()

samples = 5
test_x.dict_chars = source.dict_chars
x_test = test_x.split_n_count(False)[:samples]
x_test_pad = test_x.padding(x_test, source.maxlen)

test_y.dict_chars = target.dict_chars
y_test = test_y.split_n_count(False)[:samples]
y_test_pad = test_y.padding(y_test, target.maxlen)
y_test_pad_shift = test_y.padding_shift(y_test, target.maxlen)

assert len(x_test) == len(y_test)

def translate(model, encoder_input, output_maxlen):
    output_line = [1]
    i = 1
    while i < output_maxlen:
        prediction = model.call((encoder_input, np.array([output_line])), training=False)
        next_token_probs = prediction[0, -1, :]  # Prediction is shape (1, i, 63)
        # next_token = np.random.choice(len(next_token_probs), p=next_token_probs)
        next_token = np.argmax(next_token_probs)
        if next_token == 0:
            break
        # Update the output sequence with the sampled token
        output_line.append(next_token)
        i += 1
    return output_line

output = []

print("Testing...")
if caching:
    print("Caching is ON")
    tested_dict = load_cached_dict(testing_cache_filename)
else:
    print("Caching is OFF")

for j in tqdm(range(len(y_test_pad[:samples]))):
    i = 1
    encoder_input = np.array([x_test_pad[j]])
    if caching:
        encoder_cache_code = tuple(encoder_input[0])  # cos I can't use np array or list as a hash, [0] removes [around]
        if encoder_cache_code in tested_dict:
            output_line = tested_dict[encoder_cache_code]
        else:
            output_line = translate(model, encoder_input, target.maxlen)
            tested_dict[encoder_cache_code] = output_line
    else:
        output_line = translate(model, encoder_input, target.maxlen)
    output.append(output_line)

if caching:
    cache_dict(tested_dict, testing_cache_filename)


valid = list(target.padded.astype(np.int32))

print("prediction:", output)
# print("target:", valid)

print()
# PRETY TESTING PRINTING
rev_dict = test_y.create_reverse_dict(test_y.dict_chars)

mistake_count = 0
line_lengh = len(valid[0])
for j in range(len(list(output))):
    print("test line number:", j)
    predicted_line = np.array(output[j])
    valid_line = np.array(valid[j])
    zero_index = np.argmax(valid_line == 0)
    valid_line = valid_line[:zero_index]
    min_size = min([predicted_line.shape[0], valid_line.shape[0]])
    max_size = max([predicted_line.shape[0], valid_line.shape[0]])

    mistake_in_line = 0
    if min_size != max_size:
        print("Lines are not the same length")
        mistake_in_line += (max_size - min_size)

    for i in range(min_size):
        if valid[j][i] != output[j][i]:
            mistake_in_line += 1

    print("prediction:", end=" ")
    for char in predicted_line:
        print(rev_dict[char], end=" ")
    print()
    print("valid     :", end=" ")
    for char in valid_line:
        print(rev_dict[char], end=" ")
    print()
    print("mistakes in line:", mistake_in_line)
    print()
    mistake_count += mistake_in_line

print(round(1 - (mistake_count / (line_lengh*len(output))), 5)*100, "% testing accuracy")

