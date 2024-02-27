
training_file_name = "../data/src-sep-train.txt"
validation_file_name = "../data/src-sep-val.txt"
testing_file_name = "../data/src-sep-test.txt"

with open(training_file_name, "r", encoding="utf-8") as f:  # with spaces
    final_file = f.read()
    f.close()
with open(validation_file_name, "r", encoding="utf-8") as ff:
    valid_file = ff.read()
    ff.close()
with open(validation_file_name, "r", encoding="utf-8") as fff:
    test_file = fff.read()
    fff.close()

word_list = {}
pismeno_list = []
for line in final_file.split("\n"):
    line = line.split(" _ ")
    for word in line:
        if word not in word_list:
        #     # print(word)
            word_list[word] = 1
        else:
            word_list[word] += 1

        # for pismeno in word.split(" "):
        #     if pismeno not in pismeno_list:
        #         # print(pismeno)
        #         pismeno_list.append(pismeno)

just_once = []
others = []

# print(len(pismeno_list))  # todo 1138 pismen
# print(len(word_list))     # todo 61776 slov

for i, word in enumerate(word_list):
    # print(word, ":", word_list[word])
    if word_list[word] == 1:
        just_once.append(word)
    else:
        others.append((word, word_list[word]))

print("train")
print(len(just_once))
print(len(others))
print(len(just_once)+len(others))

just_once = []
others = []

for line in valid_file.split("\n"):
    line = line.split(" _ ")
    for word in line:
        if word not in word_list:
        #     # print(word)
            word_list[word] = 1
        else:
            word_list[word] += 1

# print(len(word_list))   # 61776 slov
for i, word in enumerate(word_list):
    # print(word, ":", word_list[word])
    if word_list[word] == 1:
        just_once.append(word)
    else:
        others.append((word, word_list[word]))

print()
print("train + valid")
print(len(just_once))
print(len(others))

just_once = []
others = []

for line in test_file.split("\n"):
    line = line.split(" _ ")
    for word in line:
        if word not in word_list:
        #     # print(word)
            word_list[word] = 1
        else:
            word_list[word] += 1

for i, word in enumerate(word_list):
    # print(word, ":", word_list[word])
    if word_list[word] == 1:
        just_once.append(word)
    else:
        others.append((word, word_list[word]))

print()
print("train+valid+test")
print(len(just_once))
print(len(others))
print(len(just_once)+len(others))
