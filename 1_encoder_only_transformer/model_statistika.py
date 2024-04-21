import pickle
import numpy as np

from transform2bin import load_model_mine, Data

model_file_name = "t2b_emb64_h4"
class_data = model_file_name + "_data/" + model_file_name + "_data.plk"
history_dict = model_file_name + "_data/" + model_file_name + '_HistoryDict'

with open(history_dict, 'rb') as file_pi:
    history = pickle.load(file_pi)

with open(class_data, 'rb') as inp:
    d = pickle.load(inp)

def get_data(file_path):
    # testing stats
    test_file_name = file_path
    with open(test_file_name, "r", encoding="utf-8") as f:  # with spaces
        test_file = f.read()
        f.close()

    x_test, y_test = d.non_slidng_data(test_file[:999], False)
    x_valid_tokenized = d.tokenize(x_test)
    prediction, metrics = d.model_test(x_valid_tokenized, y_test, model_file_name)

    pred2 = np.zeros_like(prediction)
    for i in range(len(prediction)):
        for j in range(len(prediction[0])):
            if prediction[i][j] > 0.5:
                pred2[i][j] = 1

    return x_test, y_test, pred2

text, valid, prediction = get_data("../data/src-sep-test.txt")

def separate_line(line, bins):
    for i, char in enumerate(line):
        if char != "<pad>":
            if bins[i] == 1:
                print(f"{char} _ ", end="")
            else:
                print(f"{char} ", end="")
    print()
def string_text(line, bins, j):
    out_string = ''
    for x, item in enumerate(line):
        if item == "<pad>":
            break
        out_string += item
        if x == j:
            out_string += "!"
        if bins[x] == 1:
            out_string += " _ "
        else:
            out_string += " "
    return out_string

mistake_couneter = 0
words_0 = []
words_1 = []
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
                out_string = string_text(text[i], valid[i], j)
                if valid[i][j] == 0:
                    slices = out_string.split("!")
                    one = slices[0].split(" _ ")[-1]
                    two = slices[1].split(" _ ")[0]
                    # print(one, "!", two)
                    words_0.append((one + two, one + "!" + two, out_string))
                else:
                    slices = out_string.split("!")
                    one = slices[0].split(" _ ")[-1]
                    two = slices[1].split(" _ ")[1]
                    # print(one, "!_", two)
                    words_1.append((one + two, one + "!_" + two, out_string))

print(f"mistakes:{mistake_couneter}")
# format: (spravne, predicted, kontext)
print(words_0)
print(words_1)

with open("../data/src-sep-train.txt", "r", encoding="utf-8") as f:  # with spaces
    file = f.read()
    f.close()

x_test, y_test = d.non_slidng_data(file, False)
# x_valid_tokenized = d.tokenize(x_test)
