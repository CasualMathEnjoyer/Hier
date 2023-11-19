# https://keras.io/examples/nlp/text_classification_with_transformer/
# https://netron.app/

import numpy as np
import random
from model_file import model_func
import keras
from keras.utils import set_random_seed
from keras import backend as K

print("starting transform2bin")

# TODO implement the K cross sections thing thing for data processing
# TODO fix data at the end of file - flip around?
# TODO - save class specs into json so you dont have to process the data all the time

# check this library: https://github.com/evidentlyai/evidently

a = random.randrange(0, 2**32 - 1)
# a = 1261263827
set_random_seed(a)
print("seed = ", a)

# I use the file with spaces to generate both the string without spaces and an array with 0 and 1

# about 25% of model are spaces
# precision = to minimise false alarms
# precision = true positive/(true positive + false positive)
# recall = to minimise missed spaces
# recall = TP/(TP+FN)

# False Positive = false alarm -> wanted to space it but there shouldnt be a space
# False Negative = missed space -> should be spaced but it didnt

# v datasetu momentale 203 znaku zastoupeno pouze jednou

# model_file_name = "transform2bin_4"
# training_file_name = "../data/src-sep-train.txt"
# validation_file_name = "../data/src-sep-val.txt"
# test_file_name = "../data/src-sep-test.txt"
# sep = ' '
# mezera = '_'

# model_file_name = "transform2bin_french"
# training_file_name = "../data/smallvoc_fr.txt"
# validation_file_name = "../data/smallvoc_fr.txt"
# test_file_name = "../data/smallvoc_fr.txt"
model_file_name = "transform2bin_eng"
training_file_name = "../data/smallvoc_en.txt"
validation_file_name = "../data/smallvoc_en.txt"
test_file_name = "../data/smallvoc_en.txt"
sep = ''
mezera = ' '

new = 0  # whether it creates a model (1) or loads a model (0)

batch_size = 128
epochs = 2
repeat = 0  # full epoch_num=epochs*repeat

class Data():
    vocab_size = 1138
    embed_dim = 32      # Embedding size for each token
    num_heads = 2       # Number of attention heads
    ff_dim = 64         # Hidden layer size in feed forward network inside transformer

    final_file, valid_file = '', ''
    dict_chars = {}

    window, step = 64, 20

    def __init__(self, sep, mezera):
        super().__init__()
        self.sep = sep
        self.space = mezera
    def tokenize(self, input_list):
        out = np.zeros((len(input_list), self.window))
        unk_counter = 0
        # assert self.dict_chars != None
        for i, line in enumerate(input_list):
            l = np.zeros((self.window))
            for j, c in enumerate(line):
                try:
                    l[j] = self.dict_chars[c]
                except KeyError:
                    l[j] = self.dict_chars["OOV"]
                    unk_counter += 1
            out[i] = l
        print("unknown chars in text: ", unk_counter)
        return(out)
    def model_test(self, sample:list, valid, model_name, sample_len):
        model = load_model_mine(model_name)

        sample_v = self.tokenize(sample)
        value = model.predict(sample_v)  # has to be in the shape of the input for it to predict

        for j in range(sample_len):
            for i, char in enumerate(sample[j]):
                print(char, end=self.sep)
                if value[j][i][0] > 0.5:
                    print(self.space, end=self.sep)
                i+=1
            print('')

        assert len(valid) == len(value)
        valid.resize(value.shape)
        print("F1 score:", F1_score(value, valid.astype('float32')).numpy())

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
            while pos < self.window + skipped:  # slide
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
        re_binar = np.reshape(re_binar, (num_line, self.window))

        assert len(re_binar) == len(re_windowed)
        assert len(re_binar[0]) == len(re_windowed[0])

        if not bool(self.dict_chars):  # empty dicts evaluate as false
            self.dict_chars = dict_chars
        # print(self.dict_chars)
        return re_windowed, re_binar

def F1_score(y_true, y_pred):  # taken from old keras source code
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

# -------------------------------- DATA ---------------------------------------------------------------------------
print("data preparation...")
d = Data(sep, mezera)
with open(training_file_name, "r", encoding="utf-8") as f:  # with spaces
    d.final_file = f.read()
    f.close()
with open(validation_file_name, "r", encoding="utf-8") as ff:
    d.valid_file = ff.read()
    ff.close()

x_train, y_train = d.sliding_window(d.final_file)
x_valid, y_valid = d.sliding_window(d.valid_file)

x_train_tokenized = d.tokenize(x_train)
x_valid_tokenized = d.tokenize(x_valid)

# --------------------------------- MODEL ---------------------------------------------------------------------------
print("model starting...")
if new:
    model = model_func(d.vocab_size, d.window, d.embed_dim, d.num_heads, d.ff_dim)
else:
    model = load_model_mine(model_file_name)

model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy", "Precision", "Recall", F1_score])

# --------------------------------- TRAINING ------------------------------------------------------------------------
for i in range(repeat):
    history = model.fit(
        x_train_tokenized, y_train, batch_size=batch_size, epochs=epochs,
        validation_data=(x_valid_tokenized, y_valid))
    model.save(model_file_name)
    K.clear_session()

# ---------------------------------- TESTING ------------------------------------------------------------------------
print("testing...")

with open(test_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_file = f.read()
    f.close()

sample_x, sample_y = d.sliding_window(test_file[:9600])
d.model_test(sample_x, sample_y, model_file_name, len(sample_x))