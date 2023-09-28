# tries to split text into words
# input is a char and output is between zero and one based on whether a space should come after that

# i put it in main and did not test hetner it works!


import random

print("starting hier2bin")
from keras.layers import LSTM, Input, Dense, TimeDistributed, Bidirectional
from keras.optimizers import Adam, SGD
from keras.losses import BinaryCrossentropy, mean_squared_error
from keras.models import Model, Sequential
from keras.metrics import Mean, mse
import numpy as np

from keras.utils import set_random_seed

a = random.randrange(0, 2**32 - 1)
#a = 250286255
a = 1261263827
set_random_seed(a)
print("seed fixed! ", a)

def create_binary_file(input, spaced_text, output_name, sent_len, spacing=' ', endline='\n'):
    output = []
    spaced_text.split(endline)
    spaces = 0
    for i, sentence in enumerate(input.split(endline)):
        list_sent = np.ones(sent_len)
        for j, letter in enumerate(sentence):
            if spaced_text[j + spaces + 1] != spacing:
                list_sent[j] = 0
            else:
                list_sent[j] = 1
                spaces += 1
        output.append(list_sent)
        #output.write('0') # za tecku nakonci
    output = np.asarray(output)
    np.save(output_name, output)

def create_letter_array(final_file, sentence_split_point='\n'):
    slovnikk = []
    sent_len = 0
    for line in final_file.split(sentence_split_point):
        if len(line) > sent_len:
            sent_len = len(line)
        for letter in line:
            if letter not in slovnikk:
                slovnikk.append(letter)
    return slovnikk, sent_len

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
            input_text[l][k][slovnik[mezera]] = 1
            k += 1

    for line in input_text:  # kontrola zda zadny vektor neni nulovy
        for vector in line:
            soucet = 0
            for element in vector:
                soucet += element
            assert soucet != 0

    return(input_text)

def model_test(sample_1, model_name, input_dims, slovnik):
    from keras.models import load_model
    model = load_model(model_name)
    sample = vectorise(sample_1, input_dims, slovnik)
    value = model.predict(sample)  # has to be in the shape of the input for it to predict
    #print(value)
    o = 0
    for values in value:
        for v in values:
            o += float(v)
    o = float(o/len(values))
    print("prumer = ", o)
    #o = 0.5
    i = 0
    for char in sample_1:
        print(char, end='')
        if value[0][i] > o:
            print(' ', end='')
        i += 1
    while i < len(values):
        print(0, end='')
        if value[0][i] > o:
            print(' ', end='')
        i+=1
    print('')
    print("polovina")
    o = 0.5
    i=0
    for char in sample_1:
        print(char, end='')
        if value[0][i] > o:
            print(' ', end='')
        i += 1
    while i < len(values):
        print(0, end='')
        if value[0][i] > o:
            print(' ', end='')
        i+=1
    print('')

def main():
    # OVLADACI PANEL
    train_formating = 1
    create_space_file = 0
    model_new = 1
    model_load = 0
    train = 1

    epochs = 4
    num_neurons = 200
    learning_rate = 1e-5
    batch_size = 128

    final_file_name = "../data/smallervoc_fr.txt"
    space_file_name = "space_file.npy"
    model_file_name = '../data/hier2bin2'
    pikle_slovnik_name = 'hier2bin_slovnik.pkl'

    if train_formating:
        final_file = open(final_file_name, "r", encoding="utf-8").read()
        input_file = ""
        # remove spaces section
        for char in final_file:
            if char != ' ':
                input_file+=char

        list_chars, sent_len = create_letter_array(final_file)   # has to be from final file cos longer then input
        num_lines = len(final_file.split('\n'))
        #dict_chars = dict.fromkeys(list_chars)
        dict_chars = {j:i for i,j in enumerate(list_chars)}
        embed_dim = len(dict_chars)

        print('num_lines: ', num_lines)
        print('sent_len: ', sent_len)
        print('embed_dim: ', embed_dim)

        # creating input text
        input_text = vectorise(input_file, (num_lines, sent_len, embed_dim), dict_chars)

        # creating output_text
        if create_space_file:
            create_binary_file(input_file, final_file, space_file_name, sent_len)
        binary_data = np.load(space_file_name, allow_pickle=True)
        output_text = binary_data

        assert len(input_text) == len(output_text)

        # getting the model
        if model_new:
            network_inputs = Input(shape=(sent_len, embed_dim))
            network = Bidirectional(LSTM(num_neurons, return_sequences=True, activation='tanh'))
            network_outputs = network(network_inputs)
            network_timestep = TimeDistributed(Dense(1, activation='sigmoid'))
            network_outputs = network_timestep(network_outputs)
            model = Model(inputs=network_inputs, outputs=network_outputs)

            model.compile(loss=BinaryCrossentropy(from_logits=False),
                          optimizer=Adam(learning_rate=learning_rate),
                          metrics=['accuracy'])
            model.summary()

        elif model_load:
            from keras.models import load_model
            model = load_model(model_file_name)
        else:
            print("not valid")
            model = 0

        # save the dictionary
        import pickle
        with open(pikle_slovnik_name, 'wb') as f:
            pickle.dump(dict_chars, f)

        # training
        if train:
            while True:
                model.fit(input_text, output_text,
                          batch_size=batch_size,
                          epochs=epochs,
                          shuffle=True)
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
    import pickle
    with open('hier2bin_slovnik.pkl', 'rb') as f:
        dict_chars = pickle.load(f)

    model_test("cechatétaitmonanimallepluaimé.", model_file_name, (1, sent_len, embed_dim), dict_chars)
    model_test("lesétats-unisestparfoisoccupéenjanvier,etilestparfoischaudennovembre.", model_file_name, (1, sent_len, embed_dim), dict_chars)

if __name__ == '__main__':
    main()