# tries to split text into words
# !!! sent len musi byt stejna

# create inspection mechanisms

print("starting hier2hier")
import random
import pickle

from keras.optimizers import Adam, SGD
from keras.losses import BinaryCrossentropy, mean_squared_error
from keras.metrics import Mean, mse
from keras.models import load_model
import numpy as np

from keras.utils import set_random_seed

from model_file import model_func

a = random.randrange(0, 2**32 - 1)
#a = 250286255
a = 1261263827
set_random_seed(a)
print("seed fixed! ", a)

def vectorise(final_file, input_dim, slovnik, mezera=' '):
    (radky, sent_len, embed_dim) = input_dim
    input_text = np.zeros((radky, sent_len, embed_dim))
    for l, line in enumerate(final_file.split('\n')):
        k = 0
        for i, letter in enumerate(line):
            assert letter != mezera
            input_text[l][i][slovnik[letter]] = 1
            k += 1
        while k < sent_len:
            input_text[l][k][slovnik['end_char']] = 1
            k += 1

    for line in input_text:  # kontrola zda zadny vektor neni nulovy
        for vector in line:
            soucet = 0
            for element in vector:
                soucet += element
            assert soucet != 0

    return(input_text)


def add_start_end(input_file, output_file):
    input_texts = []
    target_texts = []
    target_characters = set()
    for input_text in input_file.split('\n'):
        input_text = input_text + "\n"  # "\t" +
        input_texts.append(input_text)
        # for char in input_text:
        #     if char not in input_characters:
        #         input_characters.add(char)
    for target_text in output_file.split('\n'):
        target_text = "\t" + target_text + "\n"
        target_texts.append(target_text)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])
    return input_texts, target_texts, target_token_index

def create_data(input_texts, target_texts, sent_len_enc, sent_len_dec, embed_dim, slovnik):
    encoder_input_data = np.zeros(
        (len(input_texts), sent_len_enc, embed_dim), dtype="float32"
    )
    decoder_input_data = np.zeros(
        (len(input_texts), sent_len_dec, embed_dim), dtype="float32"
    )
    decoder_target_data = np.zeros(
        (len(input_texts), sent_len_dec, embed_dim), dtype="float32"
    )

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text):
            encoder_input_data[i, t, slovnik[char]] = 1.0
        encoder_input_data[i, t + 1:, slovnik[" "]] = 1.0
        for t, char in enumerate(target_text):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, slovnik[char]] = 1.0
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, slovnik[char]] = 1.0
        decoder_input_data[i, t + 1:, slovnik[" "]] = 1.0
        decoder_target_data[i, t:, slovnik[" "]] = 1.0

    return encoder_input_data, decoder_input_data, decoder_target_data
def model_test(sample_1, model_name, input_dims, slovnik):
    model = load_model(model_name)
    lines, sent_len, embed_dim = input_dims

    sample = np.zeros(
        (1, sent_len, embed_dim), dtype="float32"
    )
    for i, c in enumerate(sample_1):
        sample[0, i, slovnik[c]] = 1.0

    print("encoding of the sentence")
    print(sample)
    sample2 = np.zeros(
        (1, sent_len, embed_dim), dtype="float32"
    )
    sample2[0, 0, slovnik['\t']] = 1.0

    value = model.predict([sample, sample2])  # TODO - expand the network and give it appropriate state
    print("predicted values")
    print(value)

    reverse_slovnik = dict((i, char) for char, i in slovnik.items())
    for sent in value:
        for character in sent:
            c = reverse_slovnik[np.argmax(character)]
            print(c, end='')
    print('')



def main():
    # OVLADACI PANEL
    train_formating = 1
    model_new = 1
    model_load = 0
    train = 1

    epochs = 4
    num_neurons = 200
    learning_rate = 1e-5
    batch_size = 128

    # embed_dim = 30
    # num_lines = 14
    # sent_len = 90

    input_file_name = "../data/smallervoc_fr_unspaced.txt"
    final_file_name = "../data/smallervoc_fr.txt"
    model_file_name = '../data/hier2hier2'
    pikle_slovnik_name = 'hier2bin_slovnik.pkl'

    if train_formating:
        input_file = open(input_file_name, "r", encoding="utf-8").read()
        final_file = open(final_file_name, "r", encoding="utf-8").read()

        # adds \t and \n, creates lists
        input_text, output_text, dict_chars = add_start_end(input_file, final_file)

        # sentence counter
        sent_len=0
        for line in output_text:
            if len(line) > sent_len:
                sent_len = len(line)

        num_lines = len(final_file.split('\n'))
        embed_dim = len(dict_chars)

        print('num_lines: ', num_lines)
        print('sent_len: ', sent_len)
        print('embed_dim: ', embed_dim)

        # gives the data a one hot encoding, plus creates the shifted file for decoder input
        input_text, input_decoder, output_text = create_data(input_text, output_text,
                                                             sent_len, sent_len ,
                                                             embed_dim, dict_chars)

        # model creation and selection
        if model_new:
            model = model_func(sent_len, embed_dim, num_neurons)
            model.compile(loss="categorical_crossentropy",
                          optimizer=Adam(learning_rate=learning_rate),
                          metrics=['accuracy'])
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
                model.fit([input_text, input_decoder], output_text,
                          batch_size=batch_size,
                          epochs=epochs,
                          shuffle=True)  # validation_split=0.2
                q = input("continue?")
                if q == "q":
                    break
                try:
                    q = int(q)
                    epochs = q
                except ValueError:
                    pass
            print("saving model ...")
            model.save(model_file_name)
            print("model saved")
            #weights = model.layers[1].get_weights()
            #print(weights)


    #embed_dim = 40
    #sent_len = 114

    # loading saved dictionary
    with open('hier2bin_slovnik.pkl', 'rb') as f:
        dict_chars = pickle.load(f)

    model_test("cechatétaitmonanimallepluaimé.", model_file_name, (1, sent_len, embed_dim), dict_chars)
    model_test("lesétats-unisestparfoisoccupéenjanvier,etilestparfoischaudennovembre.", model_file_name, (1, sent_len, embed_dim), dict_chars)

if __name__ == '__main__':
    main()