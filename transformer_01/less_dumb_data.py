import numpy as np
import pickle
from transform2bin import load_model_mine, Data
def open_files():
    training_file_name = "../data/src-sep-train.txt"
    validation_file_name = "../data/src-sep-val.txt"
    testing_file_name = "../data/src-sep-test.txt"

    with open(training_file_name, "r", encoding="utf-8") as f:  # with spaces
        train_file = f.read()
        f.close()
    with open(validation_file_name, "r", encoding="utf-8") as ff:
        valid_file = ff.read()
        ff.close()
    with open(testing_file_name, "r", encoding="utf-8") as fff:
        test_file = fff.read()
        fff.close()
    return train_file, valid_file, test_file
def create_word_dict(file):
    word_dict = {}
    for line in file.split("\n"):
        line = line.split(" _ ")
        for word in line:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    return word_dict
def add_to_dict(dictionary, word):
    if word == " ":
        return False
    if word[-1] == " ":
        word = word[:-1]
    if word[0] == " ":
        word = word[1:]
    if word not in dictionary:
        dictionary[word] = 1
    else:
        dictionary[word] += 1
def get_data(test_file, model_file_name, d):
    x_test, y_test = d.non_slidng_data(test_file, False)
    x_valid_tokenized = d.tokenize(x_test)
    prediction, metrics = d.model_test(x_valid_tokenized, y_test, model_file_name)

    pred2 = np.zeros_like(prediction)
    for i in range(len(prediction)):
        for j in range(len(prediction[0])):
            if prediction[i][j] > 0.5:
                pred2[i][j] = 1

    return x_test, y_test, pred2
def split_string(line : str, bins) -> str:
    out_string = ''
    for x, item in enumerate(line):
        out_string += item
        if bins[x] == 1:
            out_string += " _ "
        else:
            out_string += " "
    return out_string
def all_valid(text_line, valid_line):
    global correct_dict
    sentence = split_string(text_line, valid_line)
    for word in sentence.split("_"):
        add_to_dict(correct_dict, word)

def find_wrong_indexes(valid, prediction):
    mistake_counter = 0
    all_wrong_indexes = []
    for i, line in enumerate(valid):
        wrong_indexes = []
        for j, bit in enumerate(valid[i]):
            if valid[i][j] != prediction[i][j]:
                wrong_indexes.append(j)
                mistake_counter += 1
        all_wrong_indexes.append(wrong_indexes)
    return mistake_counter, all_wrong_indexes

def mark_mistakes(line : list, index_list : list):
    out_list = []
    i = 0
    for x, word in enumerate(line):
        if i < len(index_list) and x == index_list[i]:
            out_list.append(word + "!")
            i += 1
        else:
            out_list.append(word)
    return out_list

def remove_pads(line):
    for i, item in enumerate(line):
        if item == "<pad>" or item == '':
            return line[:i]
    return line

def find_char_mistakes(line):
    global char_mistake_dict, char_correct_dict
    for char in line:
        if "!" in char:
            char = char[:-1]
            add_to_dict(char_mistake_dict, char)
        else:
            add_to_dict(char_correct_dict, char)

def balance_dicts(dict1, dict2):
    for item in dict1:
        if item not in dict2:
            dict2[item] = 0
    for item in dict2:
        if item not in dict1:
            dict1[item] = 0

def save_to_csv(char_correct_dict, char_mistake_dict):
    import csv
    # TO SAVE TO EXCEL:
    with open('output_chars.csv', 'w') as csvfile:
        csvwriter = csv.writer(csvfile, lineterminator="\n")
        csvwriter.writerow(["char", "count_training", "count_correct", "count_mistakes"])
        balance_dicts(char_correct_dict, char_mistake_dict)
        for i, word in enumerate(char_mistake_dict):
            row = [word, 0, char_correct_dict[word], char_mistake_dict[word]]
            csvwriter.writerow(row)

# PREPARATION
train_file, valid_file, test_file = open_files()
word_dict_train = create_word_dict(train_file)

# DATA THROUGH THE MODEL
model_file_name = "t2b_emb64_h4"
class_data = model_file_name + "_data/" + model_file_name + "_data.plk"
with open(class_data, 'rb') as inp:
    d = pickle.load(inp)
text, valid, prediction = get_data(test_file, model_file_name, d)

mistake_counter, all_wrong_indexes = find_wrong_indexes(valid, prediction)

ssum = 0
for line in all_wrong_indexes:
    ssum += len(line)
assert ssum == mistake_counter

char_mistake_dict, char_correct_dict = {}, {}

# text is a list of list of words
for i, line in enumerate(text):
    line = remove_pads(line)
    for item in line:
        assert item != ''
    # print(line)
    out_line = mark_mistakes(line, all_wrong_indexes[i])
    print(out_line)
    find_char_mistakes(out_line)
    save_to_csv(char_correct_dict, char_mistake_dict)






