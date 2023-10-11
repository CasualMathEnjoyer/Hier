# tries to split text into words
# input is a char and output is between zero and one based on whether a space should come after that

# accuracy = 0.8873 with attention
# accuracy = 0.8825 - 0.8841 without
# at_input = [network_outputs, network_outputs] acc = 0.8873
# at_input = [network_inputs, network_outputs]
# 0.9675 !!

# !!! sent len musi byt stejna

# create inspection mechanisms

# IDEAS:


print("starting hier2bin")
import random
import pickle

from keras.optimizers import Adam, SGD
from keras.losses import BinaryCrossentropy, mean_squared_error
from keras.metrics import mse
from keras.utils import set_random_seed


from model_file import model_func

a = random.randrange(0, 2**32 - 1)
#a = 250286255
a = 1261263827
set_random_seed(a)
print("seed fixed! ", a)

# PLAN:
# clean & create data (separate soubor)
# reformate data
# upravit format - window a tak podobne
# vektorizavat
# model
# trenovat
# testovat

def main():
    # OVLADACI PANEL
    train_formating = 1
    model_new = 0
    model_load = 1
    train = 1
    testing = 0

    instant_save = 1

    epochs = 1
    num_neurons = 256
    learning_rate = 1e-5
    batch_size = 10
    # TODO - adaptive learning rate

    # input_file_name = "../data/smallervoc_fr_unspaced.txt"
    # final_file_name = "../data/smallervoc_fr.txt"
    # model_file_name = '../data/hier2bin3'
    # space_file_name = '../data/space_file.npy'
    # mezera = ' '
    # sep = ''

    # input_file_name = "../data/hier_short.txt"
    # final_file_name = "../data/hier_short_sep.txt"
    # space_file_name = "../data/space_hier.npy"
    input_file_name = "../data/hier.txt"
    final_file_name = "../data/hier_sep.txt"
    space_file_name = "../data/space_hier_long.npy"
    model_file_name = '../data/hier2binH'
    mezera = '_'
    sep = ' '

    pikle_slovnik_name = 'hier2bin_slovnik.pkl'

    import data_func as df

    if train_formating:
        input_file = open(input_file_name, "r", encoding="utf-8").read() # without spaces
        final_file = open(final_file_name, "r", encoding="utf-8").read() # with spaces

        # formatted_input, formated_binary, dict_chars = df.re_windowing_data_nobinar(final_file, sep, mezera)
        print("formating")
        formatted_input, formated_binary, dict_chars = df.sliding_window(final_file, sep, mezera)
        num_lines = len(formatted_input)
        sent_len = len(formatted_input[0])
        embed_dim = len(dict_chars)

        assert sent_len == len(formated_binary[0])
        assert num_lines == len(formated_binary)

        print('num_lines: ', num_lines)
        print('sent_len: ', sent_len)
        print('embed_dim: ', embed_dim)

        # creating input text
        print("vectorasing")
        input_text = df.vectorise_list(formatted_input, embed_dim, num_lines, sent_len, dict_chars, mezera)
        output_text = formated_binary

        for line in input_text:
            for char in line:
                assert len(char) == embed_dim
        # for i in range(14):
        #     print(input_text[i], len(input_text[i]))
        #     print(output_text[i], len(output_text[i]))
        #     print('')

        print("starting model creation...")
        # assert True == False

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
        if testing:
            # testing
            from model_testing import model_test
            sample, _, _ = df.sliding_window(final_file[:1000], sep, mezera)  # TODO - doesnt look up short sequences
            model_test(sample, model_file_name, len(sample), sent_len, embed_dim, dict_chars, mezera, sep)

if __name__ == '__main__':
    main()