# https://keras.io/examples/nlp/text_classification_with_transformer/
import numpy as np
from model_file import model_func
from keras.models import load_model
from keras import backend as K

def model_test(sample : list, model_name, n, sent_len, embed_dim, slovnik:dict, mezera, sep):
    model = load_model(model_name)
    # TODO DOWN
    sample_v = tokenize(sample, dict_chars)
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
def sliding_window(output_file: str, sep, space):
    window, step = 64, 20

    if sep != '':
        output_file = output_file.split(sep)
    l = len(output_file)

    re_windowed, re_binar, list_chars = [], [], []
    slide = 0
    while True:
        pos, skipped = 0, 0
        line, line_n = [], []
        # line.append('<bos>')
        while pos < window + skipped:  # slide
            element = output_file[slide * step + pos]
            assert element != ''
            if element != space:
                if "\n" in element:
                    element = element.replace("\n", "")
                if element not in list_chars:
                    list_chars.append(element)
                line.append(element)
                re_binar.append(0)
            else:  # element was a space
                skipped += 1
                re_binar.pop()
                re_binar.append(1)
            pos += 1
        # line.append('<eos>')
        re_windowed.append(line)
        # I take a look at the last like without setting pos to 0 and it is too much so it stops
        slide += 1
        if slide * step + pos > l:
            break

    list_chars.append('OOV')
    dict_chars = {j: i for i, j in enumerate(list_chars)}

    num_line = len(re_windowed)
    re_binar = np.array(re_binar)
    re_binar = np.reshape(re_binar, (num_line, window))
    # print(len(re_binar))
    # print(len(re_binar[0]))
    # for i in range(len(re_windowed)):
    #     # print(re_windowed[i])
    #     # print(re_binar[i])
    #     for j, char in enumerate(re_windowed[i]):
    #         print(char, end=sep)
    #         if re_binar[i][j] == 1:
    #             print(space, end=sep)
    #     print('')

    assert len(re_binar) == len(re_windowed)
    assert len(re_binar[0]) == len(re_windowed[0])

    return re_windowed, re_binar, dict_chars
def tokenize(input_list, dict_chars):
    out = np.zeros((len(input_list), 64))
    for i, line in enumerate(input_list):
        l = np.zeros((64))
        for j, c in enumerate(line):
            l[j] = dict_chars[c]
        out[i] = l
    return(out)

final_file_name = "../data/hier_sep.txt"
model_file_name = "transform2bin"
sep = ' '
mezera = '_'

with open(final_file_name, "r", encoding="utf-8") as f:  # with spaces
    final_file = f.read()

formatted_input, formated_binary, dict_chars = sliding_window(final_file, sep, mezera)
# print(dict_chars)
input_tokens = tokenize(formatted_input, dict_chars)
output_vals = formated_binary

if 0:
    model = model_func()
else:
    from keras.models import load_model
    model = load_model(model_file_name)

# split_point = int(0.8 * len(input_tokens))
# x_test = input_tokens[:split_point]
# x_val = input_tokens[split_point:]
# y_test = output_vals[:split_point]
# y_val = output_vals[split_point:]

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = model.fit(
    input_tokens, output_vals, batch_size=32, epochs=40, validation_split=0.2)
    # validation_data=(x_val, y_val))
K.clear_session()

print("saving model ...")
model.save(model_file_name)
print("model saved")

sample, _, _ = sliding_window(final_file[:1000], sep, mezera)

model_test(sample, model_file_name, len(sample), 64, 32, dict_chars, mezera, sep)