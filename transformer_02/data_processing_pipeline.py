import numpy as np
import random
from model_file import model_func
import keras
from keras.utils import set_random_seed
from keras.utils import to_categorical
from keras import backend as K

# TODO - check for lines of all zeros in tokens
# TODO - cropping sentences might be a problem!

from sklearn.metrics import f1_score


print("starting transform2seq")

a = random.randrange(0, 2**32 - 1)
# a = 1261263827
set_random_seed(a)
print("seed = ", a)


# model_file_name = "transform2seq_1"
# training_file_name = "../data/src-sep-train.txt"
# target_file_name = "../data/tgt-train.txt"
# # validation_file_name = "../data/src-sep-val.txt"
# ti_file_name = "../data/src-sep-test.txt"  # test input file
# tt_file_name = "../data/tgt-test.txt"  # test target
# sep = ' '
# mezera = '_'
# end_line = '\n'

model_file_name = "transform2seq_fr-eng_1"
training_file_name = "../data/smallvoc_fr_.txt"
target_file_name = "../data/smallvoc_en_.txt"
# validation_file_name = "../data/src-sep-val.txt"
ti_file_name = "../data/smallervoc_fr_.txt"  # test input file
tt_file_name = "../data/smallervoc_en_.txt"  # test target
sep = ' '
mezera = '_'
end_line = '\n'

new = 1

batch_size = 128
epochs = 2
repeat = 2  # full epoch_num=epochs*repeat

class Data():
    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads

    maxlen = 0
    file = ''
    dict_chars = {}
    vocab_size = 0

    def __init__(self, sep, mezera, end_line):
        super().__init__()
        self.sep = sep
        self.space = mezera
        self.end_line = end_line

    def array_to_token(self, input_array):
        if input_array.size == 0:
            # Handle empty array case
            return np.array([])
        max_index = np.argmax(input_array)
        # result_array = np.zeros_like(input_array)  # so it is the same shape
        # result_array[max_index] = 1
        return max_index
    def create_reverse_dict(self, dictionary):
        reverse_dict = {}
        for key, value in dictionary.items():
            reverse_dict.setdefault(value, key)  # assuming values and keys unique
        return reverse_dict
    def model_test(self, sample, valid_shift, valid, model_name, sample_len):  # input = padded array of tokens
        model = load_model_mine(model_name)
        rev_dict = self.create_reverse_dict(self.dict_chars)
        value = model.predict((sample, valid_shift))  # has to be in the shape of the input for it to predict
        # TODO - do we put just the validated stuff in it or do we want to unpack the encoder?
        print("value.shape=", value.shape)
        print("valid.shape=", valid.shape)
        # valid jsou tokeny -> one hot


        assert sample_len == len(valid)
        dim = len(valid[0])
        print ("dim:", dim)
        value_one = np.zeros_like(value)
        valid_one = np.zeros_like(value)
        for i in range(sample_len):
            for j in range(len(value[i])):
                # input one-hot-ization
                token1 = int(valid[i][j])
                valid_one[i][j][token1] = 1
                # output tokenization
                token2 = self.array_to_token(value[i][j])
                value_one[i][j][token2] = 1
                print(rev_dict[token2], end=' ')
            print()

        assert len(valid) == len(value)
        # valid.resize(value.shape)
        print("F1 score:", F1_score(value, valid_one.astype('float32')).numpy())
        print("F1 score value_one:", F1_score(value_one, valid_one.astype('float32')).numpy())
    def split_n_count(self, create_dic):  # creates a list of lists of TOKENS and a dictionary
        maxlen, complete = 0, 0
        output = []
        len_list = []
        dict_chars = {"OVV": 0, "<bos>": 1, "<eos>": 2, "_": 3, "<pad>": 4}
        for line in self.file.split(self.end_line):
            line = ["<bos>"] + line.split(self.sep) + ["<eos>"]
            ll = len(line)
            len_list.append(len(line))
            if ll > maxlen:
                maxlen = ll
            complete += ll
            l = []
            for c in line:  # leave mezery !!
                if c != '':
                    if create_dic:
                        if c not in dict_chars:
                            dict_chars[c] = len(dict_chars)
                        l.append(dict_chars[c])
                    else:
                        if c in self.dict_chars:
                            l.append(self.dict_chars[c])
                        else:
                            l.append(self.dict_chars["OVV"])
            output.append(l)
        # for line in output:
        #     print(line)
        print("maxlen:", maxlen)
        print("average:", complete / len(self.file.split('\n')))
        likelyhood = 39 / 40
        weird_median = sorted(len_list)[int(len(len_list) * likelyhood)]
        print('"median" with:', likelyhood,":", weird_median)  # mene nez 2.5% ma sequence delsi, nez 100 znaku
        # maxlen: 1128
        # average: 31.42447596485441
        self.maxlen = weird_median
        if create_dic:
            self.dict_chars = dict_chars
            self.vocab_size = len(dict_chars)
            print("dict chars:", self.dict_chars)
            print("vocab size:", self.vocab_size)
        return output
    def padding(self, input_list, lengh):
        input_list_padded = np.zeros((len(input_list), lengh))  # maybe zeros?
        for i, line in enumerate(input_list):
            if len(line) > lengh: # shorten
                input_list_padded[i] = np.array(line[:lengh])
            elif len(line) < lengh:  # padd, # 4 is the code for padding
                input_list_padded[i] = np.array(line + [4 for i in range(lengh - len(line))])
            else:
                pass
        print(input_list_padded)
        return input_list_padded
    def padding_shift(self, input_list):
        input_list_padded = np.zeros((len(input_list), self.maxlen))  # maybe zeros?
        for i, line in enumerate(input_list):
            if len(line) > self.maxlen: # shorten
                input_list_padded[i] = np.array(line[1 : self.maxlen + 1])
            elif len(line) < self.maxlen:  # padd, # 4 is the code for padding
                input_list_padded[i] = np.array(line[1:] + [4 for i in range(self.maxlen - len(line) + 1)])
            else:
                pass
        print(input_list_padded)
        return input_list_padded

# TODO - better F1
# def F1_score(y_true, y_pred):
#     num_classes = target.vocab_size
#     y_true = K.one_hot(K.cast(y_true, 'int32'), num_classes)
#     y_pred = K.one_hot(K.argmax(y_pred), num_classes)
#
#     true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
#     possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
#     predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
#
#     precision = true_positives / (predicted_positives + K.epsilon())
#     recall = true_positives / (possible_positives + K.epsilon())
#
#     f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
#     return f1_val
def F1_score(y_true, y_pred): #taken from old keras source code  # TODO transform
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    # print("precision:", precision.numpy(), "recall:", recall.numpy())
    return f1_val

    # return f1_score(y_true, y_pred, average=None)
def load_model_mine(model_name):
    from model_file import PositionalEmbedding, TransformerEncoder, TransformerDecoder
    return keras.models.load_model(model_name, custom_objects={"F1_score": F1_score,
                                                               'PositionalEmbedding': PositionalEmbedding,
                                                               'TransformerEncoder': TransformerEncoder,
                                                               'TransformerDecoder': TransformerDecoder
    })

print()
print("data preparation...")
source = Data(sep, mezera, end_line)
target = Data(sep, mezera, end_line)
with open(training_file_name, "r", encoding="utf-8") as f:  # with spaces
    source.file = f.read()
    f.close()
with open(target_file_name, "r", encoding="utf-8") as ff:
    target.file = ff.read()
    ff.close()

print("first file:")
x_train = source.split_n_count(True)
x_train_pad = source.padding(x_train, source.maxlen)
print()
print("second file:")
y_train = target.split_n_count(True)
y_train_pad = target.padding(y_train, target.maxlen)
y_train_pad_one = to_categorical(y_train_pad)
y_train_pad_shift = target.padding_shift(y_train)
print(y_train_pad_one)
print()

print(x_train_pad.shape)
print(y_train_pad.shape)
print(y_train_pad_shift.shape)
print(y_train_pad_one.shape)

# x_train, y_train = d.sliding_window(d.final_file)
# x_valid, y_valid = d.sliding_window(d.target_file)
#
# x_train_tokenized = d.tokenize(x_train)
# x_valid_tokenized = d.tokenize(x_valid)

# --------------------------------- MODEL ---------------------------------------------------------------------------
print("model starting...")
if new:
    model = model_func(source.vocab_size, target.vocab_size, source.maxlen, target.maxlen)
else:
    model = load_model_mine(model_file_name)

model.compile(optimizer="adam", loss="categorical_crossentropy",
              metrics=["accuracy", "Precision", "Recall", F1_score])

# --------------------------------- TRAINING ------------------------------------------------------------------------
for i in range(repeat):
    history = model.fit(
        (x_train_pad, y_train_pad_shift), y_train_pad_one, batch_size=batch_size, epochs=epochs)
        # validation_data=(x_valid_tokenized, y_valid))
    model.save(model_file_name)
    K.clear_session()

# ---------------------------------- TESTING ------------------------------------------------------------------------
print("testing...")
test_x = Data(sep, mezera, end_line)
with open(ti_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_x.file = f.read()
    f.close()
test_y = Data(sep, mezera, end_line)
with open(tt_file_name, "r", encoding="utf-8") as f:  # with spaces
    test_y.file = f.read()
    f.close()

samples = 10
test_x.dict_chars = source.dict_chars  # mohla bych prepsat file v source a jen znova rozbehnout funkci
print("source: ", source.dict_chars)
x_test = test_x.split_n_count(False)[:10]  # ale tohle je lepsi
x_test_pad = test_x.padding(x_test, source.maxlen)

test_y.dict_chars = target.dict_chars
print("target: ", target.dict_chars)
y_test = test_y.split_n_count(False)[:10]
y_test_pad = test_y.padding(y_test, target.maxlen)
y_test_pad_shift = test_y.padding(y_test, target.maxlen)

lengh = len(x_test)
print(len(x_test))
print(len(y_test))
assert len(x_test) == len(y_test)

test_y.model_test(x_test_pad, y_test_pad_shift, y_test_pad, model_file_name, lengh)