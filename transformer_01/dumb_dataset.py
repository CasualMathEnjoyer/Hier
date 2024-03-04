from typing import Tuple, List, Any

import matplotlib.pyplot as plt
import numpy as np
import pickle
from transform2bin import load_model_mine, Data

# TODO
# graf slova v testovacim souboru prelozena dobre vs prelozena spatne
# ABOVE pro slova co rozdeluje (value=0) a nerozdeluje (value=1)
# unify the LACUNAS
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
# -----------------------------------------------------------------------------------------------------
def create_word_dict(train_file):
    word_dict = {}
    for line in train_file.split("\n"):
        line = line.split(" _ ")
        for word in line:
            if word not in word_dict:
                word_dict[word] = 1
            else:
                word_dict[word] += 1
    return word_dict
def create_letter_dict(train_file):
    letter_dict = {}
    for line in train_file.split("\n"):
        line = line.split(" _ ")
        for word in line:
            for pismeno in word.split(" "):
                if pismeno not in letter_dict:
                    letter_dict[pismeno] = 1
                else:
                    letter_dict[pismeno] += 1
    return letter_dict



# -----------------------------------------------------------------------------------------------------

def list_with_once_and_more(word_dict):
    for i, word in enumerate(word_dict):
        # print(word, ":", word_list[word])
        if word_dict[word] == 1:
            just_once.append(word)
        else:
            others.append((word, word_dict[word]))
    print("Data from dataset:")
    print("just once:", len(just_once))
    print("more than once:", len(others))
    print("all:", len(just_once) + len(others))
    return just_once, others

def get_data(file_path):
    # testing stats
    test_file_name = file_path
    with open(test_file_name, "r", encoding="utf-8") as f:  # with spaces
        test_file = f.read()
        f.close()

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
        if item == "<pad>":
            break
        out_string += item
        if bins[x] == 1:
            out_string += " _ "
        else:
            out_string += " "
    return out_string
def split_list(line : str, bins) -> list:
    out_list = []
    for x, item in enumerate(line):
        if item == "<pad>":
            break
        out_list.append(item)
        if bins[x] == 1:
            out_list.append("_")
    return out_list
def mark_mistakes(line : str, index_list : list):
    out_list = []
    i = 0
    while i < len(index_list):
        for x, word in enumerate(line.split(" _ ")):
            out_list.append(line[x])
            if x == index_list[i]:
                out_list.append("!")
    return out_list
def split_and_mark(line, bins, j):
    out_string = ''
    for x, item in enumerate(line):
        if item == "<pad>":
            break
        out_string += item
        if j != None:
            if x == j:
                out_string += "!"
        if bins[x] == 1:
            out_string += " _ "
        else:
            out_string += " "
    return out_string
def mark_mistakes_string(line : str, index_list : list):
    out_str = ''
    i = 0
    for x, word in enumerate(line.split(' ')):
        out_str += word
        if i < len(index_list) - 1:
            if x == index_list[i]:
                out_str += "!"
                i += 1
        out_str += " "
    return out_str
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
def extract_good(text_line, valid_line, wrong_indexes):
    global correct_dict
    x, start, last_space = 0, 0, 0
    for j in wrong_indexes:
        # first find the last mezera in correct text
        while x != j:
            if valid_line[x] == 1:
                last_space = x
            x += 1
        # then split the correctly translated part and save it
        if start < last_space:
            tt = split_string(text_line[start:last_space], valid_line[start:last_space])
            for word in tt.split("_"):
                add_to_dict(correct_dict, word)
        x += 1
        start = x
def find_mistakes(text_line, valid_line, pred_line):
    wrong_indexes = []
    mistake_count = 0
    for j, bit in enumerate(valid_line):
        if text_line[j] == "<pad>":  # if we ran out of characters
            break
        if valid_line[j] != pred_line[j]:
            wrong_indexes.append(j)
            mistake_count += 1
    return wrong_indexes, mistake_count
def fill_dicts(text_line, valid_line, pred_line, wrong_indexes):
    global words_0, words_1, words_0_context, words_1_context
    out_string = split_string(text_line, valid_line)
    out_string = mark_mistakes_string(out_string, wrong_indexes)

    if "_ !" in out_string:  # its just fucking                                       TODO
        return
    slices = out_string.split(" _ ")
    for word_mistaken in slices:
        if "!" in word_mistaken:
            parts = word_mistaken.split("!")
            word_clean = ''
            for part in parts:
                word_clean += part
            if word_clean not in words_0_context:
                words_0_context[word_clean] = [word_mistaken]
            else:
                words_0_context[word_clean] += [word_mistaken]
            assert word_clean != "N35"
            add_to_dict(words_0, word_clean)
        else:
            pass # healthy word found
        # else:  # when model doesnt split
        #     # slices = out_string.split("!")
        #     # one = slices[0].split(" _ ")[-1]
        #     # two = slices[1].split(" _ ")[1]  # todo check
        #     # words_1_context.append((one + " _ " + two, one + "!_" + two, out_string))
        #     # add_to_dict(words_1, one + " _ " + two)
        #     # print(one + " _ " + two)
        #     pass
def all_valid(text_line, valid_line):
    global correct_dict
    tt = split_string(text_line, valid_line)
    for word in tt.split("_"):
        add_to_dict(correct_dict, word)
def find_corrects_n_mistakes(valid, prediction, text):
    global correct_dict
    mistake_counter = 0
    correct_dict = {}
    line_mistakes = []
    for i, line in enumerate(valid):
        # if the entire line correct
        if (valid[i] == prediction[i]).all():
            all_valid(text[i], valid[i])
            line_mistakes.append(0)
        # when mistake
        else:
            wrong_indexes, num_mistakes = find_mistakes(text[i], valid[i], prediction[i])
            line_mistakes.append(num_mistakes)
            mistake_counter += num_mistakes
            extract_good(text[i], valid[i], wrong_indexes)
            fill_dicts(text[i], valid[i], prediction[i], wrong_indexes)
    return mistake_counter

def generate_lists(mistakes : dict, corrects : dict, cap=None, show_corrects=False):
    words, counts_correct, counts_mistakes = [], [], []
    for item in mistakes:
        words.append(item)
        if item in corrects:
            if cap == None:
                counts_correct.append(corrects[item])
            elif corrects[item] > cap:  # horni hranice
                counts_correct.append(cap)
                print("capped word:", item)
            else:
                counts_correct.append(corrects[item])
        else: # word in mistakes but not corrrects
            counts_correct.append(0)
        counts_mistakes.append(mistakes[item])
    if show_corrects:
        for item in corrects: # word in corrects but not in mistakes
            if item not in mistakes:
                words.append(item)
                if cap == None:
                    counts_correct.append(corrects[item])
                elif corrects[item] > cap:  # horni hranice
                    counts_correct.append(cap)
                    print("capped word:", item)
                else:
                    counts_correct.append(corrects[item])
                counts_mistakes.append(0)

    return words, counts_correct, counts_mistakes
def plot(words, counts_training, counts_mistakes=None):
    bar_width = 0.35
    index = np.arange(len(words))
    plt.bar(index, counts_training, width=bar_width, label='corrects')
    if counts_mistakes != None:
        plt.bar(index + bar_width, counts_mistakes, width=bar_width, label='mistakes')

    # Adding labels and title
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.title('ALL testing')
    # plt.xticks(index + bar_width / 2, words, rotation=45, ha='right')

    # Adding legend
    plt.legend()

    # Display the graph
    # plt.show()
    plt.savefig("mygraf.pdf", format="pdf", bbox_inches="tight")


train_file, valid_file, test_file = open_files()

word_dict = create_word_dict(train_file)
letter_dict = create_letter_dict(train_file)

just_once, others = [], []

# print(len(letter_dict))        # todo 1138 pismen
# print(len(word_dict))          # todo 61776 slov

model_file_name = "t2b_emb64_h4"
class_data = model_file_name + "_data/" + model_file_name + "_data.plk"
with open(class_data, 'rb') as inp:
    d = pickle.load(inp)

text, valid, prediction = get_data("../data/src-sep-test.txt")

# globalni promenne
words_0, words_1 = {}, {}
words_0_context, words_1_context = {}, []
correct_dict = {}

mistake_counter = find_corrects_n_mistakes(valid, prediction, text)
print(f"mistakes:{mistake_counter}")

# format: (spravne, predicted, kontext)
# print(words_0)
# print(words_1)
# words_all = words_0 + words_1

# TO PLOT A GRAPH:
# correct_dict are data from testing, word_dict are data from training
all_words, counts_all, counts_mistakes = generate_lists(words_0, correct_dict, show_corrects=False)

for word in all_words:
    if word not in word_dict:
        word_dict[word] = 0
# plot(all_words, counts_all, counts_mistakes)
# -----------------------------------------------------------------------------------------------------
def process_contexts(list):
    list.sort()
    contexts = [list[0]]
    counts = [0]
    for context in list:
        if context == contexts[-1]:
            counts[-1] += 1
        else:
            contexts.append(context)
            counts.append(1)
    return counts, contexts

import csv
# TO SAVE TO EXCEL:
with open('output.csv', 'w') as csvfile:
    csvwriter = csv.writer(csvfile, lineterminator="\n")
    csvwriter.writerow(["word", "in training", "corrects", "mistakes", "mistakes", "mistakes locations"])
    for i, word in enumerate(all_words):
        if word in words_0_context:
            counts, contexts = process_contexts(words_0_context[word])
            row = [word, word_dict[word], counts_all[i], counts_mistakes[i], counts] + contexts
            s = 0
            for n in counts:
                s += n
            assert s == counts_mistakes[i]

            csvwriter.writerow(row)
        else:
            csvwriter.writerow([word, counts_all[i], counts_mistakes[i]])
    sum = 0
    for count in counts_mistakes:
        sum += count
    print(sum, mistake_counter, sum==mistake_counter)  # cant be equal yet, because i dont consider the dvojicky