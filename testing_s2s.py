import nltk
from metrics_evaluation import metrics as m
import numpy as np
from Levenshtein import distance
from metrics_evaluation.rosm_lev import LevenshteinDistance as rosmLev
ros_distance = rosmLev()
from Tokens import Token

output_prediction_file = '../data/SAILOR_output_prediction.txt'

def process_custom_rules(text_line):
    text_line = text_line.replace("a H a", "o H o")
    text_line = text_line.replace("i", "j")
    text_line = text_line.replace("s p", "z p")
    text_line = text_line.replace("s n", "z n")
    text_line = text_line.replace("s f", "z f")
    text_line = text_line.replace("s b", "z b")
    text_line = text_line.replace("s S", "z S")
    return text_line

def test_translation(output, valid : list, rev_dict : dict, sep, mezera, use_custom_rules=False):
    """ input translated dataset as list of list of tokens"""
    mistake_count, all_chars, all_levenstein, all_line_lengh = 0, 0, 0, 0
    all_ros_levenstein = 0
    line_lengh = len(valid[0])
    num_lines = len(valid)
    output_list_words, valid_list_words = [], []  # i could take the valid text from y_test but whatever
    output_list_chars, valid_list_chars = [], []
    predicted_lines = []
    for j in range(len(list(output))):
        if j > len(valid)-1:
            print("Less items in valid then in prediction")
            break
        print("test line number:", j)
        print("predicted ", output[j])
        # print("valid_line", list(valid[j]))
        if Token.eos in output[j]:
            if output[j][0] == 1 and output[j][-1] == 2:
                predicted_line = np.array(output[j][1:-1])
            else:
                print("problem line:", output[j])
                for i in range(len(output[j])):
                    if output[j][i] == 2:
                        # raise ValueError("predicted")
                        predicted_line = np.array(output[j][1:i])
        else:
            predicted_line = np.array(output[j])

        if Token.eos in valid[j]:
            for i in range(len(valid[j])):
                if valid[j][i] == 2:
                    valid_line = np.array(valid[j][1:i])
                    break
        print("predicted ", predicted_line)
        print("valid_line", valid_line)

        if 0 in valid_line:  # aby to neusekavalo vetu
            zero_index = np.argmax(valid_line == 0)
            valid_line = valid_line[:zero_index]
        min_size = min([predicted_line.shape[0], valid_line.shape[0]])
        max_size = max([predicted_line.shape[0], valid_line.shape[0]])
        true_line_leng = valid_line.shape[0]

        mistake_in_line = 0
        if min_size != max_size:
            print("Lines are not the same length")
            mistake_in_line += (max_size - min_size)

        for i in range(min_size):
            if valid[j][i] != output[j][i]:
                mistake_in_line += 1

        output_text_line, valid_text_line = "", ""
        output_list_line, valid_list_line = [], []
        for char in predicted_line:
            output_text_line += (rev_dict[char] + sep)
            output_list_line.append(rev_dict[char])
        for char in valid_line:
            valid_text_line += (rev_dict[char] + sep)
            valid_list_line.append(rev_dict[char])
        if use_custom_rules: output_text_line = process_custom_rules(output_text_line)
        output_list_words.append(output_text_line)
        valid_list_words.append(valid_text_line)
        output_list_chars.append(output_list_line)
        valid_list_chars.append([valid_list_line])  # to be accepted by BLEU scocre

        levenstein = distance(output_text_line, valid_text_line)
        ros_levenstein = ros_distance.compute(output_text_line, valid_text_line)

        print("prediction: ", output_text_line)
        print("valid     : ", valid_text_line)
        print("mistakes  : ", mistake_in_line)
        print("levenstein: ", levenstein)
        print("ros levens: ", ros_levenstein)
        print("leven/all : ", levenstein / true_line_leng)
        print("ros/all : ", ros_levenstein / true_line_leng)
        print("line lengh: ", true_line_leng)
        print()
        mistake_count += mistake_in_line
        all_levenstein += levenstein
        all_ros_levenstein += ros_levenstein
        all_line_lengh += true_line_leng
        all_chars += max_size
        predicted_lines.append(output_text_line)

    pred_words_split_mezera, valid_words_split_mezera, valid_words_split_mezeraB = [], [], []
    for i in range(len(output_list_words)):
        pred_words_split_mezera.append(output_list_words[i].split(mezera))
        valid_words_split_mezera.append(valid_list_words[i].split(mezera))
        valid_words_split_mezeraB.append([valid_list_words[i].split(mezera)])

    round_place = 7

    word_accuracy = m.on_words_accuracy(pred_words_split_mezera, valid_words_split_mezera)
    character_accuracy = (1 - (mistake_count / all_chars))
    average_levenstein = all_levenstein / num_lines
    levenstein_per_length = all_levenstein / all_line_lengh
    # one_minus_levenstein_per_length = (1 - (all_levenstein / all_line_lengh))
    average_ros_levenstein = all_ros_levenstein / num_lines
    ros_per_lengh = round(all_ros_levenstein / all_line_lengh, round_place)
    # bleu_score_words = nltk.translate.bleu_score.corpus_bleu(valid_words_split_mezeraB, pred_words_split_mezera)
    # one_minus_ros_per_length = round((1 - (all_ros_levenstein / all_line_lengh)), round_place)
    # bleu_score_chars = nltk.translate.bleu_score.corpus_bleu(valid_list_chars, output_list_chars)

    print("word_accuracy:", round(word_accuracy, round_place))
    print("character accuracy:", round(character_accuracy, round_place))
    print(f"num lines:{num_lines}, all line lengh:{all_line_lengh}")

    print("all levenstein: ", all_levenstein)
    print("all_ros_levenstein: ", all_ros_levenstein)

    print("average Levenstein (per num lines): ", round(average_levenstein, round_place))
    print("average Ros Levenstein (per num lines): ", round(average_ros_levenstein, round_place))
    print("levenstein/all_length: ", round(levenstein_per_length, round_place))
    # print("1 - levenstein/all length: ", round(one_minus_levenstein_per_length, round_place))
    print("ROS_leve/all_length: ", ros_per_lengh)
    # print("1 - ros_leve/all length: ", one_minus_ros_per_length)
    # print("BLEU SCORE words:", bleu_score_words)
    # print("BLEU SCORE chars:", bleu_score_chars)
    print()

    with open(output_prediction_file, 'w') as f:
        for line in predicted_lines:
            f.write(line + '\n')
        f.close()

    return {
        "word_accuracy": round(word_accuracy, round_place),
        "character_accuracy": round(character_accuracy, round_place),
        "num_lines": num_lines,
        "all_line_lengh": all_line_lengh,

        "all_levenstein": all_levenstein,
        "all_ros_levenstein": all_ros_levenstein,

        "levenstein_per_num_lines": round(average_levenstein, round_place),
        "ros_levenstein_per_num_lines": round(average_ros_levenstein, round_place),
        "levenstein_per_length": round(levenstein_per_length, round_place),
        # "one_minus_levenstein_per_length": round(one_minus_levenstein_per_length, round_place),
        "ROS_per_lengh": ros_per_lengh,
        # "one_minus_ros_per_lenght": one_minus_ros_per_length,
        # "bleu_score_words": bleu_score_words,
        # "bleu_score_chars": bleu_score_chars
    }


import json
import os


def add_to_json(result_json_path, model_name: str, results: dict, sample_size: int,
                all_epochs: int, training_data : dict, keras_version: str):
    # Create an empty dictionary to hold the data
    data = {}

    # Check if the file exists
    if os.path.exists(result_json_path):
        # Load the existing data
        with open(result_json_path, 'r') as file:
            data = json.load(file)

    # Initialize model data structure if it doesn't exist
    if model_name not in data:
        data[model_name] = {}

    # Add/update entry for the model version
    data[model_name] = {
        "results": results,
        "sample_size": sample_size,
        "all_epochs": all_epochs,
        "training_data": training_data,
        "keras_version": keras_version
    }

    # Write the data back to the file
    with open(result_json_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Added/Updated entry for model: {model_name}")

# Example usage
# result_json_path = "results.json"
# model_name = "my_model"
# results = {"word_accuracy": 99.9, "character_accuracy": 99.8, "average_levenstein": 1.2, "all_line_length": 1000, "all_levenstein": 50, "levenstein_per_length": 0.05, "one_minus_levenstein_per_length": 99.95, "bleu_score_words": 0.85, "bleu_score_chars": 0.88}
# sample_size = 1000
# version = "7.0"
# all_epochs = 10
# training_data = {
#     "acc": 29,
#     "loss": 444
# }
# keras_version = "2.4.0"
# add_to_json(result_json_path, model_name, results, sample_size, version, all_epochs, training_data, keras_version)
