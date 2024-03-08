
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

lacuna_list = ["LACUNA", "SHADED1", "SHADED2", "SHADED3", "SHADED"]

def replace_lacuna(word):
    for lacuna in lacuna_list:
        word = word.replace(lacuna, "MISSING")
    return word

word_dict = {}
all_count = 0
just_once_count = 0
twice_count = 0
for line in train_file.split("\n"):
    for word in line.split(" _ "):
        word = replace_lacuna(word)
        all_count += 1
        if word not in word_dict:
            word_dict[word] = 1
        else:
            word_dict[word] += 1

for item in word_dict:
    print (item, word_dict[item])

print("all_count:", all_count)
print("unique words:", len(word_dict))
for word in word_dict:
    if word_dict[word] == 1:
        just_once_count += 1
    elif word_dict[word] == 2:
        twice_count += 1
    else:
        pass

print("twice:", twice_count)
print("just_once:", just_once_count)


