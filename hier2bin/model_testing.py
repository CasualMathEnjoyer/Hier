
from keras.models import load_model
import numpy as np
import pickle

from data_func import vectorise_list

def model_test(sample : list, model_name, n, sent_len, embed_dim, slovnik:dict, mezera, sep):
    model = load_model(model_name)

    # for n in range (len(sample)):
    #     while len(sample[n]) < sent_len:
    #         sample[n] = sample[n].append('OOV')

    sample_v = vectorise_list(sample, embed_dim, n, sent_len, slovnik, mezera)
    value = model.predict(sample_v)  # has to be in the shape of the input for it to predict

    assert len(value) == len(sample_v)
    # print(value)
    for j in range(n):
        for num in value[j]:
            if num[0] > 0.5:
                print(1, end=mezera)
            else:
                print(0, end=sep)
        print('')

        for i, char in enumerate(sample[j]):
            print(char, end=sep)
            if value[j][i][0] > 0.5:
                print(mezera, end=sep)
            i+=1

        print('')


# dict_name = 'hier2bin_slovnik.pkl'
# input_file_name = ''
# input_text = ''
# output_text = ''
# model_file_name = ''
# sent_len = ''
# embed_dim = ''
# mezera = ''
# sep = ''
#
# # loading saved dictionary
# with open(dict_name, 'rb') as f:
#     dict_chars = pickle.load(f)
#
# with open(input_file_name) as f:
#     file = f.read()
# file = file.split('\n')
# for i in range(3):
#     # file = file[i].split(sep)
#     print("string: ", input_text[i])
#     print("goal: ", output_text[i])
#
#     model_test(file[i], model_file_name, (1, sent_len, embed_dim), dict_chars, mezera, sep, sent_len)
#
# # model_test("cechatétaitmonanimallepluaimé.", model_file_name, (1, sent_len, embed_dim), dict_chars)
# # model_test("lesétats-unisestparfoisoccupéenjanvier,etilestparfoischaudennovembre.", model_file_name, (1, sent_len, embed_dim), dict_chars)
