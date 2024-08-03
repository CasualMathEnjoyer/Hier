import matplotlib.pyplot as plt

def plot_histogram(mistake_dict):
    ros_mistakes = list(mistake_dict.keys())
    counts = list(mistake_dict.values())

    plt.bar(mistakes[1:], counts[1:], color='skyblue')
    plt.xlabel('Number of Mistakes')
    plt.ylabel('Count')
    plt.title('Counts of Number of Mistakes in a Sentence')
    # plt.savefig('mistakes.pdf', bbox_inches='tight')
    plt.show()


# PREPARE THE DATA
with open("../data_3") as file:
    f = file.read()

with open("../data/src-sep-test.txt") as data:
    source = data.read()

f = f.split("test line number: ")[1:]

results = f[-1].split("\n\n")[1]
f[-1] = f[-1].split("\n\n")[0]


source = source.split("\n")
source = source[:-1]  # cos I remove the last sentence from the testing - probelem! todo
# also i have problems with model generating longer sentences than I have in the valid data set todo

target = []
for i in range(len(f)):
    item_dict = {}
    items = f[i].split("\n")
    # cleaning
    if items[1] == 'Lines are not the same length':
        items.remove(items[1])
    while True:
        if '' in items: items.remove('')
        else: break
    item_dict["prediction"] = items[1][13:]
    item_dict["valid"] = items[2][13:]
    item_dict["source"] = source[i]
    item_dict["mistakes"] = items[3].split("mistakes  :  ")[-1].strip()
    item_dict["levenstein"] = items[4].split("levenstein:  ")[-1].strip()
    item_dict["ros_levens"] = items[5].split("ros levens:  ")[-1].strip()
    item_dict["leven/all"] = items[6].split("leven/all :  ")[-1].strip()
    item_dict["ros/all"] = items[7].split("ros/all :  ")[-1].strip()
    item_dict["line_lengh"] = items[8].split("line lengh:  ")[-1].strip()

    target.append(item_dict)

assert len(source) == len(target)


# MAIN LOOP
correct = []
one_mistake, one_mistake_insert = [], []
mistake_dict = {}
for i, item in enumerate(target):
    ros_mistakes = item["ros_levens"]
    prediction = item["prediction"]
    source = item["source"]
    valid = item["valid"]
    mistakes = item["mistakes"]
    status = None

    if ros_mistakes not in mistake_dict: mistake_dict[ros_mistakes] = 1
    else: mistake_dict[ros_mistakes] += 1

    if mistakes == "0": correct.append(prediction)
    elif mistakes == "1" and len(prediction) == len(valid):
        for o in range(len(valid)):
            if valid[o] != prediction[o]:
                bad_char = prediction[o] + " " + valid[o]
                status = ""
                break
    elif mistakes == "1" and len(prediction) != len(valid):
        if len(prediction) > len(valid):
            bad_char = prediction[-1]  # inicialisation
            status = "INSERTED"
            for o in range(len(valid)):
                if valid[o] != prediction[o]:
                    bad_char = prediction[o]
                    break
        if len(prediction) < len(valid):
            bad_char = prediction[-1]
            status = "DELETED"
            for o in range(len(valid)):
                if valid[o] != prediction[o]:
                    bad_char = valid[o]
                    break
    if status is not None:
        one_mistake.append({"source":source,
                            "prediction":prediction,
                            "valid":valid,
                            "bad_char":bad_char,
                            "bad_locations":[o],
                            "status":status})

# plot_histogram(mistake_dict)

# for i in range(len(one_mistake)):
#     if one_mistake[i]["status"] == "INSERTED":
#         continue
#     print(one_mistake[i]["source"])
#     print(one_mistake[i]["prediction"])
#     print(one_mistake[i]["valid"])
#     print(one_mistake[i]["bad_char"])



def sort_based_on_mistakes(list_of_dict):
    new_dict = {}
    for item in list_of_dict:
        if item["status"] != "":
            continue
        if item["bad_char"] not in new_dict:
            new_dict[item["bad_char"]] = []
        new_dict[item["bad_char"]].append(item)
    # Sorting the dictionary based on the length of the lists
    sorted_keys = sorted(new_dict, key=lambda k: len(new_dict[k]), reverse=True)
    # Creating a new dictionary with sorted keys
    sorted_dict = {key: new_dict[key] for key in sorted_keys}
    return sorted_dict

def print_as_latex(dict):
    latex_item = "\item"
    enter = "\\"
    text_tt = enter + "texttt{"
    end_bracket = "}"
    latex_source = dict['source'].replace('_', '\_')
    latex_prediction = dict['prediction']
    latex_target = dict['valid']
    location = dict["bad_locations"]
    for o in location:
        if o < len(latex_prediction):
            bad_char = latex_prediction[o]
            color = enter + "textcolor{red}{" + str(bad_char) + "}"
            latex_prediction = latex_prediction[:o] + color + latex_prediction[o+1:]
        if o < len(latex_target):
            bad_char = latex_target[o]
            color = enter + "textcolor{red}{" + str(bad_char) + "}"
            latex_target = latex_target[:o] + color + latex_target[o+1:]
    latex_prediction = latex_prediction.replace('_', '\_')
    latex_target = latex_target.replace('_', '\_')
    print(f"{latex_item} source sen : {text_tt}{latex_source}{end_bracket} {enter}{enter}")
    print(f"prediction : {text_tt}{latex_prediction}{end_bracket} {enter}{enter}")
    print(f"target sen : {text_tt}{latex_target}{end_bracket}")


sorted_one_mistake = sort_based_on_mistakes(one_mistake)
first_time = True
for l in sorted_one_mistake.keys():
    if "_" in l:
        l.replace('_', '\_')
    if len(sorted_one_mistake[l]) > 1:
        print("\\" + "subsection{", l, "}")
        print("This error occurred:", len(sorted_one_mistake[l]), "times")
        print("\\" + "begin{itemize}")
        for item in sorted_one_mistake[l][:6]:
            print_as_latex(item)
        print("\\" + "end{itemize}")
    elif first_time:
        print("\\" + "subsection{single occurrence}")
        print("\\" + "begin{itemize}")
        first_time = False
    else:
        for item in sorted_one_mistake[l]:
            print_as_latex(item)
print("\\" + "end{itemize}")



exit()
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


with open("../data_3_clean") as file:
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
