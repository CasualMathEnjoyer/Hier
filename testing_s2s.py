import nltk
from metrics_evaluation import metrics as m
import numpy as np
import json
import os
import string
import sys

from metrics_evaluation.rosm_lev import LevenshteinDistance as rosmLev
ros_distance = rosmLev()
from Utils.Tokens import Token

# Import metrics libraries for sacrebleu and rouge
try:
    import datasets
    load_metric = datasets.load_metric
    HAS_METRICS = True
except ImportError:
    HAS_METRICS = False
    print("Warning: datasets library not available. ROUGE-L and SacreBLEU will not be calculated.", file=sys.stderr)

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


def add_to_json(result_json_path, model_name: str, results: dict, sample_size: int,
                all_epochs: int, training_data: dict, keras_version: str):
    data = {}
    if os.path.exists(result_json_path):
        with open(result_json_path, 'r') as file:
            data = json.load(file)

    if model_name not in data:
        data[model_name] = {}

    data[model_name] = {
        "results": results,
        "sample_size": sample_size,
        "all_epochs": all_epochs,
        "training_data": training_data,
        "keras_version": keras_version
    }

    with open(result_json_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f"Added/Updated entry for model: {model_name}")


def cut_bos_and_eos(line : np.array):
    bos_index, eos_index = None, None
    if np.any(line == Token.eos):
        eos_index = np.where(line == Token.eos)[0][0]
        line = line[:eos_index]
    if np.any(line == Token.bos):
        bos_index = np.where(line == Token.bos)[0][0]
        line = line[bos_index + 1:]

    if bos_index and eos_index:
        if bos_index >= eos_index: print(f"BOS index {bos_index} >= EOS index {eos_index}")
        if bos_index != 0: print(f"Unusual BOS index: {bos_index}")
    return line

def count_mistakes(valid_line, predicted_line):
    min_size = min([predicted_line.shape[0], valid_line.shape[0]])
    max_size = max([predicted_line.shape[0], valid_line.shape[0]])
    mistake_in_line = 0
    if min_size != max_size:
        print("  -> len(predicted) != len(valid), difference: ", max_size - min_size)
        mistake_in_line += (max_size - min_size)

    for i in range(min_size):
        if valid_line[i] != predicted_line[i]:
            mistake_in_line += 1
    return mistake_in_line


def calculate_gold(predictions, targets):
    """
    Calculate GOLD metric: number of sentences translated with no errors at all.
    A sentence is "gold" if prediction exactly matches target (after stripping).
    """
    gold_count = 0
    
    for pred, tgt in zip(predictions, targets):
        if pred.strip() == tgt.strip():
            gold_count += 1
    
    return gold_count


def calculate_rouge_and_bleu(predictions, targets):
    """
    Calculate ROUGE-L and SacreBLEU metrics at token level (word level).
    Matches the implementation from calculate_metrics_ramses_format.py:
    - Converts character-separated format to word-level format
    - Removes spaces between characters, replaces underscores with spaces
    - Then normalizes: strip punctuation, lowercase, split into tokens
    """
    if not HAS_METRICS:
        return None, None
    
    if len(predictions) == 0:
        return None, None
    
    # Load metrics
    rouge_metric = load_metric("rouge")
    sacrebleu_metric = load_metric("sacrebleu")
    
    # Process and add batches - convert to word-level format then normalize
    for pred, tgt in zip(predictions, targets):
        # Step 1: Remove spaces between characters to get word-level format
        # "x a a _ = k _ a n x" -> "xaa_=k_anx"
        pred_words = pred.replace(" ", "")
        tgt_words = tgt.replace(" ", "")
        
        # Step 2: Replace underscores with spaces
        # "xaa_=k_anx" -> "xaa =k anx"
        pred_words = pred_words.replace("_", " ")
        tgt_words = tgt_words.replace("_", " ")
        
        # Step 3: Normalize same way as calculate_metrics_ramses_format.py
        # Strip punctuation, lowercase, split into tokens
        pred_tokens = pred_words.strip(string.punctuation).lower().split()
        tgt_tokens = tgt_words.strip(string.punctuation).lower().split()
        
        # Pass tokenized lists directly
        # datasets library accepts both lists and strings, but lists give different results
        sacrebleu_metric.add_batch(
            predictions=[pred_tokens],
            references=[[tgt_tokens]]
        )
        
        # ROUGE also accepts tokenized lists
        rouge_metric.add_batch(
            predictions=[pred_tokens],
            references=[[tgt_tokens]]
        )
    
    # Compute metrics
    rouge_result = rouge_metric.compute()
    sacrebleu_result = sacrebleu_metric.compute()
    
    # Extract scores
    sacrebleu = sacrebleu_result["score"]
    
    rougeL_value = rouge_result["rougeL"]
    if isinstance(rougeL_value, (int, float)):
        # evaluate library returns float (0-1)
        rouge_l = 100 * float(rougeL_value)
    else:
        # datasets library returns object with .mid.fmeasure
        rouge_l = 100 * rougeL_value.mid.fmeasure
    
    return rouge_l, sacrebleu

def test_translation(transformer_output, valid : list, rev_dict : dict, sep, mezera, use_custom_rules=False):
    mistake_count = 0
    max_all_chars, all_valid_chars, all_pred_chars = 0, 0, 0
    all_ros_levenstein = 0

    output_lines_strings, valid_lines_strings = [], []  # i could take the valid text from y_test but whatever

    predicted_text_to_save = []
    for j in range(len(list(transformer_output))):
        if j > len(valid)-1:
            print(" !!! -> num_predictions > num_valids")
            break

        print("test line number:", j)
        print(f"output[j]:", transformer_output[j])

        predicted_line = cut_bos_and_eos(np.array(transformer_output[j]))
        valid_line = cut_bos_and_eos(valid[j])

        print("predicted:", predicted_line)
        print("valid    :", valid_line)

        mistake_in_line = count_mistakes(valid_line, predicted_line)

        pred_line_string = ""
        for char in predicted_line:
            pred_line_string += (rev_dict[char] + sep)

        valid_line_string = ""
        for char in valid_line:
            valid_line_string += (rev_dict[char] + sep)

        if use_custom_rules: pred_line_string = process_custom_rules(pred_line_string)

        output_lines_strings.append(pred_line_string)
        valid_lines_strings.append(valid_line_string)

        true_line_leng = valid_line.shape[0]
        pred_line_len = predicted_line.shape[0]

        ros_levenstein = ros_distance.compute(pred_line_string, valid_line_string)


        print("len(valid): ", true_line_leng)
        print("len(pred) : ", pred_line_len)
        print("prediction: ", pred_line_string)
        print("valid     : ", valid_line_string)
        print("mistakes  : ", mistake_in_line)
        print("RLEV      : ", int(ros_levenstein))
        print("RLEV / len valid: ", round(ros_levenstein / true_line_leng, 4))
        print("RLEV / len pred : ", round(ros_levenstein / pred_line_len, 4))
        print()

        mistake_count += mistake_in_line
        all_ros_levenstein += ros_levenstein
        all_valid_chars += true_line_leng
        all_pred_chars += pred_line_len
        max_all_chars += max([predicted_line.shape[0], valid_line.shape[0]])
        predicted_text_to_save.append(pred_line_string)

    pred_words_split_mezera = []
    valid_words_split_mezera = []
    for sentence_num in range(len(output_lines_strings)):
        pred_words_split_mezera.append(output_lines_strings[sentence_num].split(mezera))
        valid_words_split_mezera.append(valid_lines_strings[sentence_num].split(mezera))

    round_place = 7
    num_lines = len(valid)

    word_accuracy = round(m.on_words_accuracy(pred_words_split_mezera, valid_words_split_mezera), round_place)
    character_accuracy = round((1 - (mistake_count / all_valid_chars)), round_place)

    avg_RLEV_per_line = round(all_ros_levenstein / num_lines, round_place)
    avg_RLEV_per_valid_char = round(all_ros_levenstein / all_valid_chars, round_place)
    avg_RLEV_per_pred_char = round(all_ros_levenstein / all_pred_chars, round_place)

    # Calculate new metrics: sacrebleu, rouge L, and gold
    print("Calculating additional metrics (sacrebleu, rouge L, gold)...")
    rouge_l, sacrebleu = calculate_rouge_and_bleu(output_lines_strings, valid_lines_strings)
    gold_count = calculate_gold(output_lines_strings, valid_lines_strings)

    header = "RESULTS"
    separator = "-" * (30 + round_place + 3)
    rows = [
        f"{'num_lines':<30} | {num_lines}",
        f"{'all_valid_chars':<30} | {all_valid_chars}",
        f"{'all_pred_chars':<30} | {all_pred_chars}",
        f"{'max_all_chars':<30} | {max_all_chars}",
        f"{'word_accuracy':<30} | {word_accuracy}",
        f"{'character_accuracy':<30} | {character_accuracy}",
        f"{'all_ros_levenstein':<30} | {all_ros_levenstein}",
        f"{'avg_RLEV_per_line':<30} | {avg_RLEV_per_line}",
        f"{'avg_RLEV_per_valid_char':<30} | {avg_RLEV_per_valid_char}",
        f"{'avg_RLEV_per_pred_char':<30} | {avg_RLEV_per_pred_char}",
    ]
    
    # Add new metrics to rows
    if rouge_l is not None:
        rows.append(f"{'rouge_l':<30} | {round(rouge_l, round_place)}")
    if sacrebleu is not None:
        rows.append(f"{'sacrebleu':<30} | {round(sacrebleu, round_place)}")
    rows.append(f"{'gold':<30} | {gold_count}")

    print(header)
    print(separator)
    for row in rows:
        print(row)
    print(separator)
    print()


    with open(output_prediction_file, 'w') as f:
        for line in predicted_text_to_save:
            f.write(line + '\n')
        f.close()

    result_dict = {
        "num_lines": num_lines,
        "all_valid_chars": all_valid_chars,
        "all_pred_chars": all_pred_chars,
        "max_all_chars": max_all_chars,

        "word_accuracy": word_accuracy,
        "character_accuracy": character_accuracy,

        "all_ros_levenstein": all_ros_levenstein,

        "avg_RLEV_per_line": avg_RLEV_per_line,
        "avg_RLEV_per_valid_char": avg_RLEV_per_valid_char,
        "avg_RLEV_per_pred_char": avg_RLEV_per_pred_char,
    }
    
    # Add new metrics
    if rouge_l is not None:
        result_dict["rouge_l"] = round(rouge_l, round_place)
    if sacrebleu is not None:
        result_dict["sacrebleu"] = round(sacrebleu, round_place)
    result_dict["gold"] = gold_count
    
    return result_dict
