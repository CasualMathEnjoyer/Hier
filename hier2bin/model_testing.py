
from keras.models import load_model
import numpy as np
import pickle

from data_func import vectorise_list

def model_test(sample_1:str, model_name, input_dims, slovnik:dict, mezera, sep, sent_len):
    model = load_model(model_name)
    n, sent_len, embed_dim = input_dims
    # sample = vectorise(sample_1, input_dims, slovnik, mezera, sep)
    sample = sample_1[0].split(sep)
    while len(sample) < sent_len:
        sample.append('OOV')
    sample = vectorise_list([sample], embed_dim, n, sent_len, slovnik, mezera)
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


dict_name = 'hier2bin_slovnik.pkl'
input_file_name = ''
input_text = ''
output_text = ''
model_file_name = ''
sent_len = ''
embed_dim = ''
mezera = ''
sep = ''

# loading saved dictionary
with open(dict_name, 'rb') as f:
    dict_chars = pickle.load(f)

with open(input_file_name) as f:
    file = f.read()
file = file.split('\n')
for i in range(3):
    # file = file[i].split(sep)
    print("string: ", input_text[i])
    print("goal: ", output_text[i])

    model_test(file[i], model_file_name, (1, sent_len, embed_dim), dict_chars, mezera, sep, sent_len)

# model_test("cechatétaitmonanimallepluaimé.", model_file_name, (1, sent_len, embed_dim), dict_chars)
# model_test("lesétats-unisestparfoisoccupéenjanvier,etilestparfoischaudennovembre.", model_file_name, (1, sent_len, embed_dim), dict_chars)
