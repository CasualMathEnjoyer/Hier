import nltk
from metrics_evaluation import metrics as m
import numpy as np
from Levenshtein import distance
distance("lewenstein", "levenshtein")
def test_translation(output, valid : list, rev_dict : dict, sep, mezera):
    """ input translated dataset as list of list of tokens"""
    mistake_count, all_chars, all_levenstein, all_line_lengh = 0, 0, 0, 0
    line_lengh = len(valid[0])
    num_lines = len(valid)
    output_list_words, valid_list_words = [], []  # i could take the valid text from y_test but whatever
    output_list_chars, valid_list_chars = [], []
    for j in range(len(list(output))):
        print("test line number:", j)
        predicted_line = np.array(output[j])
        valid_line = np.array(valid[j])
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
        output_list_words.append(output_text_line)
        valid_list_words.append(valid_text_line)
        output_list_chars.append(output_list_line)
        valid_list_chars.append([valid_list_line])  # to be accepted by BLEU scocre
        levenstein = distance(output_text_line, valid_text_line)
        print("prediction: ", output_text_line)
        print("valid     : ", valid_text_line)
        print("mistakes  : ", mistake_in_line)
        print("levenstein: ", levenstein)
        print("leven/all : ", levenstein / true_line_leng)
        print("line lengh: ", true_line_leng)
        print()
        mistake_count += mistake_in_line
        all_levenstein += levenstein
        all_line_lengh += true_line_leng
        all_chars += max_size

    pred_words_split_mezera, valid_words_split_mezera, valid_words_split_mezeraB = [], [], []
    for i in range(len(output_list_words)):
        pred_words_split_mezera.append(output_list_words[i].split(mezera))
        valid_words_split_mezera.append(valid_list_words[i].split(mezera))
        valid_words_split_mezeraB.append([valid_list_words[i].split(mezera)])

    word_accuracy = m.on_words_accuracy(pred_words_split_mezera, valid_words_split_mezera)

    word_accuracy = m.on_words_accuracy(pred_words_split_mezera, valid_words_split_mezera)
    character_accuracy = (1 - (mistake_count / all_chars)) * 100
    average_levenstein = all_levenstein / num_lines
    levenstein_per_length = all_levenstein / all_line_lengh
    one_minus_levenstein_per_length = (1 - (all_levenstein / all_line_lengh)) * 100
    bleu_score_words = nltk.translate.bleu_score.corpus_bleu(valid_words_split_mezeraB, pred_words_split_mezera)
    bleu_score_chars = nltk.translate.bleu_score.corpus_bleu(valid_list_chars, output_list_chars)

    print("word_accuracy:", round(word_accuracy * 100, 5), "%")
    print("character accuracy:", round(character_accuracy, 5), "%")
    print("average Levenstein: ", average_levenstein)
    print("all line length: ", all_line_lengh, ", all levenstein: ", all_levenstein)
    print("levenstein/all length: ", levenstein_per_length)
    print("1 - levenstein/all length: ", round(one_minus_levenstein_per_length, 5), "%")
    print("BLEU SCORE words:", bleu_score_words)
    print("BLEU SCORE chars:", bleu_score_chars)

    return {
        "word_accuracy": round(word_accuracy * 100, 5),
        "character_accuracy": round(character_accuracy, 5),
        "average_levenstein": average_levenstein,
        "all_line_length": all_line_lengh,
        "all_levenstein": all_levenstein,
        "levenstein_per_length": levenstein_per_length,
        "one_minus_levenstein_per_length": round(one_minus_levenstein_per_length, 5),
        "bleu_score_words": bleu_score_words,
        "bleu_score_chars": bleu_score_chars
    }
