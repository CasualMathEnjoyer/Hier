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