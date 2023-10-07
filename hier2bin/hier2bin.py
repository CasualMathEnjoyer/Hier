# tries to split text into words
# input is a char and output is between zero and one based on whether a space should come after that

# accuracy = 0.8873 with attention
# accuracy = 0.8825 - 0.8841 without
# at_input = [network_outputs, network_outputs] acc = 0.8873
# at_input = [network_inputs, network_outputs]
# 0.9675 !!

# !!! sent len musi byt stejna

# create inspection mechanisms

# IDEA - sentences not split necesarilly correctly - resplit them?

print("starting hier2bin")
import random
import pickle

from keras.optimizers import Adam, SGD
from keras.losses import BinaryCrossentropy, mean_squared_error
from keras.metrics import mse
from keras.models import load_model
from keras.utils import set_random_seed

import numpy as np

from model_file import model_func

a = random.randrange(0, 2**32 - 1)
#a = 250286255
a = 1261263827
set_random_seed(a)
print("seed fixed! ", a)

def create_letter_array(final_file, sentence_split_point='\n', sep=''):
    slovnikk = []
    sent_len = 0
    for line in final_file.split(sentence_split_point):
        if sep != '':
            line = line.split(' ')
        if len(line) > sent_len:
            sent_len = len(line)
        for letter in line:
            if letter not in slovnikk:
                slovnikk.append(letter)
    slovnikk.append('sos')
    slovnikk.append('eos')
    slovnikk.append('OOV')
    return slovnikk, sent_len

def vectorise(final_file, input_dim, slovnik, mezera=' ', sep=''): # TODO predelat do vektorove podoby
    (radky, sent_len, embed_dim) = input_dim
    input_text = np.zeros((radky, sent_len, embed_dim))
    for l, line in enumerate(final_file.split('\n')):
        k = 0
        if sep != '':
            line = line.split(sep)
        for i, letter in enumerate(line):
            if i < sent_len:
                assert letter != mezera
                try:
                    input_text[l][i][slovnik[letter]] = 1
                except KeyError:
                    input_text[l][i][slovnik['OOV']] = 1
                k += 1
        while k < sent_len:
            input_text[l][k][slovnik['eos']] = 1
            # input_text[l][k][slovnik[mezera]] = 1
            k += 1

    for line in input_text:  # kontrola zda zadny vektor neni nulovy
        for vector in line:
            soucet = 0
            for element in vector:
                soucet += element
            assert soucet != 0

    return(input_text)

def model_test(sample_1:str, model_name, input_dims, slovnik:dict, mezera, sep, sent_len):
    model = load_model(model_name)
    sample = vectorise(sample_1, input_dims, slovnik, mezera, sep)
    value = model.predict(sample)  # has to be in the shape of the input for it to predict
    # print(value)
    for num in value[0]:
        if num[0] > 0.5:
            print(1, end=' ')
        else:
            print(0, end='')
    print('')
    i = 0
    if sep != '':
        sample_1 = sample_1.split(sep)
    for i, char in enumerate(sample_1):
        if i < sent_len:
            print(char, end=sep)
            if value[0][i] > 0.5:
                print(mezera, end=sep)
            i+=1
        else:
            print('.', end=sep)
            i += 1
    print('')

def main():
    # OVLADACI PANEL
    train_formating = 1
    model_new = 1
    model_load = 0
    train = 1

    instant_save = 0

    epochs = 100
    num_neurons = 124
    learning_rate = 1e-5
    batch_size = 10

    # input_file_name = "../data/smallervoc_fr_unspaced.txt"
    # final_file_name = "../data/smallervoc_fr.txt"
    # model_file_name = '../data/hier2bin3'
    # space_file_name = '../data/space_file.npy'
    # mezera = ' '
    # sep = ''
    # fixed_sent = 90

    input_file_name = "../data/hier_short.txt"
    final_file_name = "../data/hier_short_sep.txt"
    space_file_name = "../data/space_hier.npy"
    model_file_name = '../data/hier2binH'
    mezera = '_'
    sep = ' '
    fixed_sent = 64

    pikle_slovnik_name = 'hier2bin_slovnik.pkl'

    if train_formating:
        input_file = open(input_file_name, "r", encoding="utf-8").read()
        final_file = open(final_file_name, "r", encoding="utf-8").read()

        list_chars, sent_len = create_letter_array(final_file, sep=sep)   # has to be from final file cos longer then input

        sent_len = fixed_sent

        num_lines = len(final_file.split('\n'))
        dict_chars = {j:i for i,j in enumerate(list_chars)}
        embed_dim = len(dict_chars)

        print('num_lines: ', num_lines)
        print('sent_len: ', sent_len)
        print('embed_dim: ', embed_dim)

        # creating input text
        input_text = vectorise(input_file, (num_lines, sent_len, embed_dim), dict_chars, mezera=mezera, sep=sep)

        # creating output_text
        binary_data = np.load(space_file_name, allow_pickle=True)
        output_text = binary_data

        # for i in range(14):
        #     print(input_text[i], len(input_text[i]))
        #     print(output_text[i], len(output_text[i]))
        #     print('')

        print('input:', len(input_text), 'output:', len(output_text))
        assert len(input_text) == len(output_text)

        print("starting model creation...")

        # model creation and selection
        if model_new:
            model = model_func(sent_len, embed_dim, num_neurons)
            model.compile(loss=BinaryCrossentropy(from_logits=False),
                          optimizer=Adam(learning_rate=learning_rate),
                          metrics=['accuracy', mse])
        elif model_load:
            from keras.models import load_model
            model = load_model(model_file_name)
        else:
            raise Exception("No model selected")
        model.summary()

        # save the dictionary
        with open(pikle_slovnik_name, 'wb') as f:
            pickle.dump(dict_chars, f)

        # training
        if train:
            while True:
                model.fit(input_text, output_text,
                          batch_size=batch_size,
                          epochs=epochs,
                          shuffle=True)
                if instant_save != 1:
                    q = input("continue?")
                    if q == "q":
                        break
                    try:
                        q = int(q)
                        epochs = q
                    except ValueError:
                        pass
                else:
                    break
            print("saving model ...")
            model.save(model_file_name)
            print("model saved")
            #weights = model.layers[1].get_weights()
            #print(weights)

    # loading saved dictionary
    with open('hier2bin_slovnik.pkl', 'rb') as f:
        dict_chars = pickle.load(f)

    with open(input_file_name) as f:
        file = f.read()
    file = file.split('\n')
    for i in range(3):
        # file = file[i].split(sep)
        model_test(file[i], model_file_name, (1, sent_len, embed_dim), dict_chars, mezera, sep, sent_len)

    # model_test("cechatétaitmonanimallepluaimé.", model_file_name, (1, sent_len, embed_dim), dict_chars)
    # model_test("lesétats-unisestparfoisoccupéenjanvier,etilestparfoischaudennovembre.", model_file_name, (1, sent_len, embed_dim), dict_chars)

if __name__ == '__main__':
    main()