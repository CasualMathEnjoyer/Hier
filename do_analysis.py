
with open("data_3") as file:
    f = file.read()

f = f.split("test line number: ")[1:]

results = f[-1].split("\n\n")[1]
f[-1] = f[-1].split("\n\n")[0]

correct = []
one_mistake = []
one_mistake_insert = []
mistake_dict = {}
for item in f:
    item = item.split("\n")
    if item[1] == 'Lines are not the same length':
        item.remove(item[1])
    # print(item[3])
    mistakes = int(float(item[5].split("ros levens:  ")[-1]))
    if mistakes not in mistake_dict:
        mistake_dict[mistakes] = 1
    else:
        mistake_dict[mistakes] += 1
    if item[3] == "mistakes  :  0":
        # print(item)
        sentence = item[1][13:]
        correct.append(sentence)
    elif item[3] == "mistakes  :  1" and len(item[1][13:]) == len(item[2][13:]):
        sentence = item[1][13:]
        valid_sentence = item[2][13:]
        for o in range(len(valid_sentence)):
            if valid_sentence[o] != sentence[o]:
                bad_char = sentence[o] + " " + valid_sentence[o]
                break
        one_mistake.append([sentence, valid_sentence, bad_char])
    elif item[3] == "mistakes  :  1" and len(item[1][13:]) != len(item[2][13:]):
        sentence = item[1][13:]
        valid_sentence = item[2][13:]
        if len(sentence) > len(valid_sentence):
            bad_char = sentence[-1]
            for o in range(len(valid_sentence)):
                if valid_sentence[o] != sentence[o]:
                    bad_char = sentence[o]
                    break
            one_mistake_insert.append([sentence, valid_sentence, bad_char, "INSERTED"])
        if len(sentence) < len(valid_sentence):
            bad_char = sentence[-1]
            for o in range(len(valid_sentence)):
                if valid_sentence[o] != sentence[o]:
                    bad_char = valid_sentence[o]
                    break
            one_mistake_insert.append([sentence, valid_sentence, bad_char, "DELETED"])
# print(correct)
# print(len(correct))
# print(one_mistake)
# print(len(one_mistake))


import matplotlib.pyplot as plt
# Plotting the data
mistakes = list(mistake_dict.keys())
counts = list(mistake_dict.values())

# plt.bar(mistakes[1:], counts[1:], color='skyblue')
# plt.xlabel('Number of Mistakes')
# plt.ylabel('Count')
# plt.title('Counts of Number of Mistakes in a Sentence')
# # plt.xticks(mistakes)
# plt.savefig('mistakes.pdf', bbox_inches='tight')
# plt.show()

# mistakes = {}
# for item in one_mistake:
#     # if item[2] == "i T":
#     #     print(item[0])
#     #     print(item[1])
#     #     print()
#     if item[2] == "n m":
#         print(item[0])
#         print(item[1])
#         print()
#     if item[2] not in mistakes:
#         mistakes[item[2]] = 1
#     else:
#         mistakes[item[2]] += 1
#
# for line in mistakes:
#     print(line, ":", mistakes[line])
#
# for mistake in mistakes:
#     print(mistake, mistakes[mistake])
#     for item in one_mistake:
#         if item[2] == mistake:
#             print(item[0])
#             print(item[1])

for item in one_mistake_insert:
    print(item[0])
    print(item[1])
    print(item[2], item[3])


with open("data_3_clean") as file:
    f = file.read()
    f = f.split("test line number: ")[1:]

prediction = []
valid = []
for line in f:
    line = line.split("\n")
    sentence = line[1][13:]
    valid_sentence = line[2][13:]
    line[1] = line[1][13:]
    line[2] = line[2][13:]
    prediction.append(sentence)
    valid.append(valid_sentence)
    line.pop()
    print(line)
# print(prediction)
# print(valid)
