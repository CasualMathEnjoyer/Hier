import matplotlib.pyplot as plt
import numpy as np
import pickle
from transform2bin import load_model_mine, Data

# TODO
# graf slova v testovacim souboru prelozena dobre vs prelozena spatne
# ABOVE pro slova co rozdeluje (value=0) a nerozdeluje (value=1)
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
train_file, valid_file, test_file = open_files()
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

word_dict = create_word_dict(train_file)
letter_dict = create_letter_dict(train_file)

just_once = []
others = []

# print(len(letter_dict))        # todo 1138 pismen
# print(len(word_dict))          # todo 61776 slov

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

model_file_name = "t2b_emb64_h4"
class_data = model_file_name + "_data/" + model_file_name + "_data.plk"
with open(class_data, 'rb') as inp:
    d = pickle.load(inp)
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
text, valid, prediction = get_data("../data/src-sep-test.txt")

def print_separated(line, bins):
    for i, char in enumerate(line):
        if char != "<pad>":
            if bins[i] == 1:
                print(f"{char} _ ", end="")
            else:
                print(f"{char} ", end="")
    print()

def mark_mistake_string(line, bins, j =None):
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

def funkce_or_sth(valid, prediction, text):
    mistake_couneter = 0
    words_0, words_1 = [], []
    correct_words = {}
    # only considers sentences with mistakes
    for i, line in enumerate(valid):
        if (valid[i] == prediction[i]).all():  # if the entire line correct
            tt = mark_mistake_string(text[i], valid[i])
            for word in tt.split("_"):
                add_to_dict(correct_words, word)

        else:
            wrong_indexes = []
            for j, bit in enumerate(valid[i]):
                if text[i][j] == "<pad>":  # if we ran out of characters
                    break
                if valid[i][j] != prediction[i][j]:
                    wrong_indexes.append(j)
                    # print(f"sentence:{i}")
                    # print(f"mistake at: {j}")
                    # print("val: ", end="")
                    # separate_line(text[i], valid[i])
                    # print("pre: ", end="")
                    # separate_line(text[i], prediction[i])
                    mistake_couneter += 1
                    out_string = mark_mistake_string(text[i], valid[i], j)
                    if valid[i][j] == 0:  # model splits something it shouldnt
                        slices = out_string.split("!")
                        one = slices[0].split(" _ ")[-1]
                        two = slices[1].split(" _ ")[0]
                        # print(one, "!", two)
                        words_0.append((one + two, one + "!" + two, out_string))
                    else:  # model doesnt split
                        slices = out_string.split("!")
                        one = slices[0].split(" _ ")[-1]
                        two = slices[1].split(" _ ")[1]
                        # print(one, "!_", two)
                    words_1.append((one + two, one + "!_" + two, out_string))
            x = 0
            start = 0
            cor_string = ''
            last_space = 0
            for j in wrong_indexes:
                while x != j:
                    cor_string += text[i][x]
                    cor_string += " "
                    if valid[i][x] == 1:
                        last_space = x
                    x += 1
                if start < last_space:
                    tt = mark_mistake_string(text[i][start:last_space], valid[i][start:last_space])
                    for word in tt.split("_"):
                        add_to_dict(correct_words, word)
                x += 1
                start = x
    print("dict")
    print(correct_words)


    return words_0, words_1, mistake_couneter

words_0, words_1, mistake_counter = funkce_or_sth(valid, prediction, text)
print(f"mistakes:{mistake_counter}")
# format: (spravne, predicted, kontext)
print(words_0)
# print(words_1)
words_all = words_0 + words_1

mistakes_dict = {}
for word in words_0:
    add_to_dict(mistakes_dict, word)
print(mistakes_dict)

words = []
counts_all = []
counts_mistakes = []
for item in mistakes_dict:
    if item in word_dict:
        print(item)
        words.append(item)
        if word_dict[item] > 5:  # horni hranice
            counts_all.append(5)
        else:
            counts_all.append(word_dict[item])
        counts_mistakes.append(mistakes_dict[item])
    else:
        print(f"problem word:{item}")

# -----------------------------------------------------------------------------------------------------
def plot(words, counts_training, counts_mistakes=None):
    # dict = word_list
    # counts = list(dict.values())[:100]

    # Creating a bar graph
    bar_width = 0.35
    index = np.arange(len(words))
    plt.bar(index, counts_training, width=bar_width, label='training')
    if counts_mistakes != None:
        plt.bar(index + bar_width, counts_mistakes, width=bar_width, label='testing mistakes')

    # Adding labels and title
    plt.xlabel('Words')
    plt.ylabel('Counts')
    plt.title('Word Counts Comparison')
    plt.xticks(index + bar_width / 2, words, rotation=45, ha='right')

    # Adding legend
    plt.legend()

    # Display the graph
    plt.show()

# plot(list(word_dict.keys())[:100], list(word_dict.values())[:100])
# plot(["haf", "haff", "haff"], [1, 3, 5], [0, 2, 2])
plot(words, counts_all, counts_mistakes)