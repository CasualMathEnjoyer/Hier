import pickle

import numpy as np

from transform2bin import load_model_mine, Data


model_file_name = "t2b_emb64_h4"
class_data = model_file_name + "_data/" + model_file_name + "_data.plk"
history_dict = model_file_name + "_data/" + model_file_name + '_HistoryDict'

with open(history_dict, "rb") as file_pi:
    history = pickle.load(file_pi)

def get_testing_data():
    # testing stats
    test_file_name = "../data/src-sep-test.txt"
    with open(test_file_name, "r", encoding="utf-8") as f:  # with spaces
        test_file = f.read()
        f.close()

    with open(class_data, 'rb') as inp:
        d = pickle.load(inp)

    x_test, y_test = d.non_slidng_data(test_file[:1000], False)
    x_valid_tokenized = d.tokenize(x_test)
    prediction, metrics = d.model_test(x_valid_tokenized, y_test, model_file_name)

    pred2 = np.zeros_like(prediction)
    for i in range(len(prediction)):
        for j in range(len(prediction[0])):
            if prediction[i][j] > 0.5:
                pred2[i][j] = 1

    return x_test, y_test, pred2

text, valid, prediction = get_testing_data()

def separate_line(line, bins):
    for i, char in enumerate(line):
        if char != "<pad>":
            if bins[i] == 1:
                print(f"{char} _ ", end="")
            else:
                print(f"{char} ", end="")
    print()

def string_text(line, bins):
    out_string = ''
    found = False
    c = False
    for x, item in enumerate(line):
        out_string += item
        if x == j:
            out_string += "!"
            found = True
        if bins[x] == 1:
            out_string += " _ "
            if found:
                if c == True:
                    break
                c = True
        else:
            out_string += " "
    return out_string

mistake_couneter = 0
splits_words = []
doesnt_split_words = []
for i, line in enumerate(valid):
    for j, bit in enumerate(valid[i]):
        if text[i][j] != "<pad>":
            if valid[i][j] != prediction[i][j]:
                print(f"sentence:{i}")
                print(f"mistake at: {j}")
                print("val: ", end="")
                separate_line(text[i], valid[i])
                print("pre: ", end="")
                separate_line(text[i], prediction[i])
                mistake_couneter += 1
                out_string = string_text(text[i], valid[i])
                if valid[i][j] == 0:
                    word = out_string.split("_")[-2]
                    print(word)
                else:
                    slices = out_string.split("!")
                    one = slices[0].split("_")[-1]
                    two = slices[1].split("_")[1]
                    print(slices[0])
                    print(slices[1])
                    print(one, "!_", two)

print(f"mistakes:{mistake_couneter}")