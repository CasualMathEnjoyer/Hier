from uu import Error

import numpy as np
import random
from model_file import model_func
import keras
from keras.utils import set_random_seed
from keras import backend as K
import pickle
import os

print("starting TESTING transform2bin")

a = random.randrange(0, 2**32 - 1)
a = 1261263827
set_random_seed(a)


model_folder_path = '/home/katka/Documents/Trained models/Transformer_encoder_only/'
model_name = 't2b_emb64_h4'
model_file_name = f"{model_folder_path}{model_name}"

# test_file_name = "../data/src-sep-test.txt"
# test_file_name = "../data/sailor_test_src.txt"
# target_sep_file = "../data/sailor_separated_rsc.txt"

test_file_name = "../data/train_src.txt"
target_sep_file = "../data/train_src_separated.txt"

test_file_name = "../data/test_src.txt"
target_sep_file = "../data/test_src_separated.txt"

sep = ' '
mezera = '_'
endline = "\n"

folder_path = model_file_name + "_data"
class_data = folder_path + "/" + model_name + "_data.plk"
history_dict = folder_path + "/" + model_name + '_HistoryDict'

optimizer = "adam"
# loss_function = "binary_focal_crossentropy"
loss_function = "binary_crossentropy"


# HYPER PARAMETERS:
embed_dim = 128
num_heads = 2
ff_dim = 64         # Hidden layer size in feed forward network inside transformer
maxlen = 128
step = 64

class Data():
    vocab_size = 0         # gets inicialised to the size of dict - if i want to extend with more data
                           # i need to have initially greater vocab size
    embed_dim = embed_dim  # Embedding size for each token
    num_heads = num_heads  # Number of attention heads
    ff_dim = ff_dim        # Hidden layer size in feed forward network inside transformer

    final_file, valid_file = '', ''

    x_train, y_train = '', ''
    x_train_tok = ''
    x_valid, y_valid = '', ''
    x_valid_tok = ''

    dict_chars = {}

    step = step

    maxlen = maxlen

    def __init__(self, sep, mezera, endline):
        super().__init__()
        self.sep = sep
        self.space = mezera
        self.endline = endline

    def tokenize_window(self, input_list):
        num_lines = len(input_list)
        out = np.zeros((num_lines, self.maxlen))
        unk_counter = 0
        # assert self.dict_chars != None
        for i, line in enumerate(input_list):
            l = np.zeros((self.maxlen))
            for j, c in enumerate(line):
                try:
                    l[j] = self.dict_chars[c]
                except KeyError:
                    l[j] = self.dict_chars["OOV"]
                    unk_counter += 1
            out[i] = l
        print("unknown chars in text: ", unk_counter)
        return(out)
    def tokenize(self, input_list):
        num_lines = len(input_list)
        out = np.zeros((num_lines, self.maxlen))
        unk_counter = 0
        # assert self.dict_chars != None
        for i, line in enumerate(input_list):
            l = np.zeros((self.maxlen))
            for j, c in enumerate(line):
                try:
                    l[j] = self.dict_chars[c]
                except KeyError:
                    l[j] = self.dict_chars["OOV"]
                    unk_counter += 1
            out[i] = l
        print("unknown chars in text: ", unk_counter)
        return(out)
    def model_test(self, sample_v, valid, model_name):
        model = load_model_mine(model_name)

        prediction = model.predict(sample_v)  # has to be in the shape of the input for it to predict

        assert len(valid) == len(prediction)
        for i in range(len(valid)):
            assert len(valid[i]) == len(prediction[i])

        valid.resize(prediction.shape)  # resize to the sahpe of prediction

        # metrics
        accuracy_metric = keras.metrics.BinaryAccuracy()  # binary includes threshold=0.5
        accuracy_metric.update_state(valid, prediction)
        acc = accuracy_metric.result().numpy()

        precision_metrics = keras.metrics.Precision(thresholds=0.5)
        precision_metrics.update_state(valid, prediction)
        prec = precision_metrics.result().numpy()

        recall_metrics = keras.metrics.Recall(thresholds=0.5)
        recall_metrics.update_state(valid, prediction)
        rec = recall_metrics.result().numpy()

        f1 = F1_score(prediction, valid.astype('float32')).numpy()

        pred2 = np.zeros_like(prediction)
        for i, line in enumerate(valid):
            for j, char in enumerate(line):
                if prediction[i][j] > 0.5:
                    pred2[i][j] = 1

        def edit_distance(valid, pred):
            score = 0
            for i, line in enumerate(valid):
                for j, char in enumerate(line):
                    if valid[i][j] != pred[i][j]:
                        score += 1
            l = len(valid)
            return score/l

        ed = edit_distance(valid, pred2)


        # confusion matrix:
        from sklearn.metrics import confusion_matrix

        y_pred_classes = (prediction > 0.5).astype(int)
        valid2 = valid.astype(int)
        y_pred_flat = np.array(y_pred_classes).flatten()
        valid_flat = np.array(valid2).flatten()
        print("Unique values in pred2:", np.unique(y_pred_classes))
        conf_matrix = confusion_matrix(valid_flat, y_pred_flat)

        if __name__ == "__main__":
            print("Accuracy:", acc)
            print("Precision:", prec)
            print("Recall:", rec)
            print("F1 score:", f1)
            print("Edit distance:", ed)
            print("Confusion Matrix:\n", conf_matrix)


        return prediction, [acc, prec, rec, f1, ed]
    def model_use(self, sample_v, model_name):
        model = load_model_mine(model_name)
        prediction = model.predict(sample_v)
        return prediction
    def print_separation(self, text, prediction):
        output = ''
        print(len(text), len(prediction))
        for j in range(len(text)):
            # print(text[j])
            # print(prediction[j])
            for i, char in enumerate(text[j]):
                if char != "<pad>":
                    # print(char, end=self.sep)
                    output += char
                    output += self.sep
                if prediction[j][i][0] > 0.5:
                # if prediction[j][i] > 0.5:
                #     print(self.space, end=self.sep)
                    output += self.space
                    output += self.sep
                i+=1
            output += self.endline
            # print('')
        return output
    def sliding_window(self, output_file: str):  # chunks the data into chunks
        if self.sep != '':
            output_file = output_file.split(self.sep)
        l = len(output_file)

        re_windowed, re_binar, list_chars = [], [], []
        slide, num_space, num_nonspaces = 0, 0, 0
        list_chars.append('OOV')

        while True:
            pos, skipped = 0, 0
            line, line_n = [], []
            # line.append('<bos>')
            while pos < self.maxlen + skipped:  # slide
                element = output_file[slide * self.step + pos]
                assert element != ''
                if element != self.space:
                    if "\n" in element:
                        element = element.replace("\n", "")
                    if element not in list_chars:
                        list_chars.append(element)
                    line.append(element)
                    re_binar.append(0)
                    num_nonspaces += 1
                else:  # element was a space
                    skipped += 1
                    re_binar.pop()
                    re_binar.append(1)
                    num_space += 1
                pos += 1
            # line.append('<eos>')
            re_windowed.append(line)
            # I take a look at the last like without setting pos to 0 and it is too much so it stops
            slide += 1
            if slide * self.step + pos > l:
                # print ("spaces:", num_space)
                # print ("non spaces :", num_nonspaces)
                break

        dict_chars = {j: i for i, j in enumerate(list_chars)}

        num_line = len(re_windowed)
        re_binar = np.array(re_binar)
        re_binar = np.reshape(re_binar, (num_line, self.maxlen))

        assert len(re_binar) == len(re_windowed)
        assert len(re_binar[0]) == len(re_windowed[0])

        if not bool(self.dict_chars):  # empty dicts evaluate as false
            self.dict_chars = dict_chars
        # print(self.dict_chars)
        return re_windowed, re_binar
    def non_slidng_data(self, output_file: str, create_dict: bool):  # chunks the data into chunks
        output_file_pad = []
        binar, list_chars = [], []

        list_chars.append('<pad>')  # code 0
        list_chars.append('OOV')

        if self.endline != '':
            output_file = output_file.split(self.endline)
        num_lines = len(output_file)

        binar = np.zeros((num_lines, self.maxlen))
        assert binar[0].size == self.maxlen

        output_without_space = np.zeros_like(binar)

        for i, line in enumerate(output_file):
            if self.sep != '':
                splitted = line.split(self.sep)
            else:
                splitted = list(line)
            num_mezer = 0
            for j, element in enumerate(splitted):
                if element not in list_chars:
                    list_chars.append(element)
                if element == self.space:
                    assert j > 0
                    if j-1-num_mezer < self.maxlen:
                        binar[i][j-1-num_mezer] = 1  # o jedno predchozi character je nastaveny jako posledni
                    num_mezer += 1  # num mezer in the whole unpadded unshortened text
                    # others dont get processed in terms of spaces

            # remove mezery
            new_splitted = [i for i in splitted if i != self.space]
            # print(new_splitted)
            # print(self.space)
            # assert 0
            assert num_mezer == len([i for i in splitted if i == self.space])

            # pad sentence
            if len(new_splitted) > self.maxlen:
                new_splitted = new_splitted[:self.maxlen]
            else:
                while len(new_splitted) < self.maxlen:
                    new_splitted.append('<pad>')

            output_file_pad.append(new_splitted)

        del output_file
        l = len(output_file_pad[0])

        for line in output_file_pad:
            assert len(line) == l

        if create_dict:
            dict_chars = {j: i for i, j in enumerate(list_chars)}
            self.dict_chars = dict_chars
            self.vocab_size = len(dict_chars)
            print(self.vocab_size)

        return output_file_pad, binar

def F1_score(y_true, y_pred):  # taken from unsuccessful_attempts keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    # print("precision:", precision.numpy(), "recall:", recall.numpy())
    return f1_val
def load_model_mine(model_name):
    return keras.models.load_model(model_name, custom_objects={"F1_score": F1_score})
def join_dicts(dict1, dict2):
    dict = {}
    if dict1 == {}:
        dict = dict2

    if dict1.keys() == dict2.keys():
        pass
    else:
        print(dict1.keys(), " != ", dict2.keys())

    for i in range(len(dict1.keys())):
        history_list = list(dict1.keys())
        # print(history_list[i])
        ar = []
        for item in dict1[history_list[i]]:
            ar.append(item)
        for item in dict2[history_list[i]]:
            ar.append(item)
        dict[history_list[i]] = ar
    # print(dict)
    return dict
# -------------------------------- DATA ---------------------------------------------------------------------------
def model_run():
    if not os.path.exists(folder_path):
        raise FileNotFoundError

    print("loading class data")
    with open(class_data, 'rb') as inp:
        d = pickle.load(inp)
    # --------------------------------- MODEL ---------------------------------------------------------------------------
    print("model loading...")
    model = load_model_mine(model_file_name)

    model.compile(optimizer=optimizer,
                  loss=loss_function,
                  metrics=["accuracy", "Precision", "Recall", F1_score])
    model.summary()

    # ---------------------------------- TESTING ------------------------------------------------------------------------
    print("testing...")

    with open(test_file_name, "r", encoding="utf-8") as f:  # with spaces
        test_file = f.read()
        f.close()


    # for masking layer
    x_test, y_test = d.non_slidng_data(test_file, False)
    # print(len(x_test), len(y_test))

    # print(x_test[0])

    x_valid_tokenized = d.tokenize(x_test)
    print(len(x_valid_tokenized))
    prediction, metrics = d.model_test(x_valid_tokenized, y_test, model_file_name)
    print(len(prediction))
    output = d.print_separation(x_test, prediction)
    # print(output)
    with open(target_sep_file, 'w') as f:
        f.write(output)
        f.close()

if __name__ == "__main__":
    model_run()