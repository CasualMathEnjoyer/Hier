# https://keras.io/examples/nlp/text_classification_with_transformer/
import numpy as np
import random
from model_file import model_func
import keras
from keras.utils import set_random_seed
from keras import backend as K

print("starting transform2bin")

a = random.randrange(0, 2**32 - 1)
# a = 1261263827
set_random_seed(a)
print("seed = ", a)

# about 25% of model are spaces
# precision = to minimise false alarms
# precision = true positive/(true positive + false positive)
# recall = to minimise missed spaces
# recall = TP/(TP+FN)

# False Positive = false alarm -> wanted to space it but there shouldnt be a space
# False Negative = missed space -> should be spaced but it didnt

training_file_name = "../data/src-sep-train.txt"
validation_file_name = "../data/src-sep-val.txt"
model_file_name = "transform2bin_4"
sep = ' '
mezera = '_'

batch_size = 64
epochs = 1
repeat = 1  # full epoch_num=epochs*repeat

class Data():
    vocab_size = 1138
    maxlen = 64
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 64  # Hidden layer size in feed forward network inside transformer

    def __init__(self, final_file, valid_file, sep, mezera, dict_chars=None):
        super().__init__()
        if dict_chars is None:
            dict_chars = {}
        self.final_file = final_file
        self.valid_file = valid_file
        self.sep = sep
        self.mezera = mezera
        self.dict_chars = dict_chars

    def tokenize(self, input_list):
        out = np.zeros((len(input_list), 64))
        unk_counter = 0
        # assert self.dict_chars != None
        for i, line in enumerate(input_list):
            l = np.zeros((64))
            for j, c in enumerate(line):
                try:
                    l[j] = self.dict_chars[c]
                except KeyError:
                    l[j] = self.dict_chars["OOV"]
                    unk_counter += 1
            out[i] = l
        print("unknown chars in text: ", unk_counter)
        return(out)
    def model_test(self, sample : list, model_name, n, sent_len, embed_dim, slovnik:dict, mezera, sep):
        model = load_model_mine(model_name)
        # TODO DOWN
        sample_v = self.tokenize(sample)
        value = model.predict(sample_v)  # has to be in the shape of the input for it to predict

        assert len(value) == len(sample_v)
        # print(value)
        for j in range(n):
            for num in value[j]:
                if num[0] > 0.5:
                    print(1, end=mezera)
                else:
                    print(0, end=sep)
            print('')

            for i, char in enumerate(sample[j]):
                print(char, end=sep)
                if value[j][i][0] > 0.5:
                    print(mezera, end=sep)
                i+=1

            print('')
    def sliding_window(self, output_file: str):
        window, step = 64, 20
        sep = self.sep
        space = self.mezera

        if sep != '':
            output_file = output_file.split(sep)
        l = len(output_file)

        re_windowed, re_binar, list_chars = [], [], []
        list_chars.append('OOV')

        slide, num_space, num_nonspaces= 0, 0, 0
        while True:
            pos, skipped = 0, 0
            line, line_n = [], []
            # line.append('<bos>')
            while pos < window + skipped:  # slide
                element = output_file[slide * step + pos]
                assert element != ''
                if element != space:
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
            if slide * step + pos > l:
                # print ("spaces:", num_space)
                # print ("non spaces :", num_nonspaces)
                break

        dict_chars = {j: i for i, j in enumerate(list_chars)}

        num_line = len(re_windowed)
        re_binar = np.array(re_binar)
        re_binar = np.reshape(re_binar, (num_line, window))

        assert len(re_binar) == len(re_windowed)
        assert len(re_binar[0]) == len(re_windowed[0])

        if not bool(self.dict_chars):  # empty dicts evaluate as false
            self.dict_chars = dict_chars

        return re_windowed, re_binar, dict_chars

def F1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val
def load_model_mine(model_name):
    return keras.models.load_model(model_name, custom_objects={"F1_score": F1_score})

# -------------------------------- DATA ---------------------------------------------------------------------------
print("data preparation...")
with open(training_file_name, "r", encoding="utf-8") as f:  # with spaces
    final_file = f.read()
with open(validation_file_name, "r", encoding="utf-8") as ff:
    valid_file = ff.read()

d = Data(final_file, valid_file, sep, mezera)

x_train, y_train, dict_chars = d.sliding_window(final_file)
x_valid, y_valid, dict_val = d.sliding_window(valid_file)

x_train_tokenized = d.tokenize(x_train)
x_valid_tokenized = d.tokenize(x_valid)

# --------------------------------- MODEL ---------------------------------------------------------------------------
print("model starting...")
if 0:
    model = model_func(d.vocab_size, d.maxlen, d.embed_dim, d.num_heads, d.ff_dim)
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

sample, _, _ = d.sliding_window(final_file[:1000])
d.model_test(sample, model_file_name, len(sample), 64, 32, dict_chars, mezera, sep)