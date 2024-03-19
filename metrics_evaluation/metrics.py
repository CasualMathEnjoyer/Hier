import numpy as np

def calc_accuracy(predicted, valid, num_sent, sent_len):
    val_all = 0
    for i in range(num_sent):
        # print("prediction:", self.one_hot_to_token([value[i]]))
        # print("true value:", self.one_hot_to_token([valid_one[i]]))
        val = 0
        for j in range(sent_len):
            if predicted[i][j] != valid[i][j]:  # because valid has weird shape with dim 1 in the middle
                val += 1
        # print("difference:", val, "accuracy:", 1-(val/sent_len))
        val_all += val
    return round(1-(val_all/(sent_len*num_sent)), 2)  # formating na dve desetina mista
def calculate_precision_recall_f1(y_true, y_pred, label):
    true_positive = np.sum((y_true == label) & (y_pred == label))
    false_positive = np.sum((y_true != label) & (y_pred == label))
    false_negative = np.sum((y_true == label) & (y_pred != label))
    print(" TP:", true_positive, " FP:", false_positive, " FN:", false_negative)

    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else None
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else None

    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def calculate_metrics(true_labels, predicted_labels, classes):
    """
    Calculate recall, precision, and F1 score for multiple classes.

    Parameters:
        true_labels (list): True labels of the samples.
        predicted_labels (list): Predicted labels of the samples.
        classes (list): List of unique classes.

    Returns:
        metrics (dict): A dictionary containing recall, precision, and F1 score for each class,
                        as well as overall recall and precision.
    """
    metrics = {'overall_precision': 0, 'overall_recall': 0}

    true_labels = np.array(true_labels)
    predicted_labels = np.array(predicted_labels)

    # Calculate class-wise metrics
    present_classes = []
    for cls in classes:
        true_positive = np.sum((true_labels == cls) & (predicted_labels == cls))
        false_positive = np.sum((true_labels != cls) & (predicted_labels == cls))
        false_negative = np.sum((true_labels == cls) & (predicted_labels != cls))
        if true_positive != 0 or false_positive != 0 or false_negative != 0:
            present_classes.append(cls)

        precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
        recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        metrics[cls] = {'precision': precision, 'recall': recall, 'f1_score': f1_score}

        metrics['overall_precision'] += precision
        metrics['overall_recall'] += recall

    num_classes = len(present_classes)
    metrics['overall_precision'] /= num_classes
    metrics['overall_recall'] /= num_classes
    print("HERE:", metrics['overall_precision'], metrics['overall_recall'])

    return metrics
def f1_precision_recall(file, y_true, y_pred):
    dict = file.dict_chars
    # if input one hot, use below:
    # y_true = np.array(target.one_hot_to_token(y_true))
    # y_pred = np.array(target.one_hot_to_token(y_pred))
    unique_labels = np.array(list(dict.values()))
    # print("labels:", unique_labels)
    # print("d labs:", np.array(list(dict.values())))

    file.create_reverse_dict(file.dict_chars)
    # for label in unique_labels:
    #     print(file.reverse_dict[label], end=":")
    #     precision, recall, _ = calculate_precision_recall_f1(y_true, y_pred, label)
    #     total_precision += precision
    #     total_recall += recall
    #     # print("char:", target.reverse_dict[label], "- f1:", round(2*precision*recall/(precision+recall), 5) if (precision+recall) > 0 else "zero")
    #     # TODO  zero
    metrics = calculate_metrics(y_true, y_pred, unique_labels)
    macro_precision = metrics["overall_precision"]
    macro_recall = metrics["overall_recall"]
    print("precision:", metrics["overall_precision"])
    print("recall:   ", metrics["overall_recall"])

    macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0

    # return f1, precision and recall formated na dve desetinna mista
    # return float(f'{macro_f1:.2f}'), float(f'{macro_precision:.2f}'), float(f'{macro_recall:.2f}')
    return round(macro_f1, 2), round(macro_precision, 2), round(macro_recall, 2)

def on_words_accuracy(prediction_list, valid_list):
    all_value = 0
    all_sent = 0
    for i in range(len(prediction_list)):
        value_i = 0
        len_sent = len(prediction_list[i])
        for j in range(len_sent):
            try:
                if prediction_list[i][j] != valid_list[i][j]:
                    if prediction_list[i][j] == ' ' and valid_list[i][j] == '':  # fix - the last item after split
                        pass
                    else:
                        value_i += 1
            except IndexError:  # the lengh is not matching so thats wrong
                value_i += 1
        try:
            if "." in prediction_list[i][len_sent - 1]:
                value_i -= 1  # protoze tecka tam jakoby je
                prediction_list[i][len_sent - 1] = valid_list[i][len_sent - 1]
        except IndexError:
            pass
        # print(1 - (value_i/len_sent))
        all_value += value_i
        all_sent += len_sent
    return 1-(all_value/all_sent)