# https://keras.io/examples/nlp/text_classification_with_transformer/
import numpy as np
from model_file import model_func
from keras.models import load_model
from keras import backend as K

# about 25% of model are spaces
# precision = to minimise false alarms
# precision = true positive/(true positive + false positive)
# recall = to minimise missed spaces
# recall = TP/(TP+FN)

# False Positive = false alarm -> wanted to space it but there shouldnt be a space
# False Negative = missed space -> should be spaced but it didnt
def model_test(sample : list, model_name, n, sent_len, embed_dim, slovnik:dict, mezera, sep):
    model = load_model(model_name)
    # TODO DOWN
    sample_v = tokenize(sample, dict_chars)
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
def sliding_window(output_file: str, sep, space):
    window, step = 64, 20

    if sep != '':
        output_file = output_file.split(sep)
    l = len(output_file)

    re_windowed, re_binar, list_chars = [], [], []
    slide = 0
    num_space = 0
    num_nonspaces = 0
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

    list_chars.append('OOV')
    dict_chars = {j: i for i, j in enumerate(list_chars)}

    num_line = len(re_windowed)
    re_binar = np.array(re_binar)
    re_binar = np.reshape(re_binar, (num_line, window))
    # print(len(re_binar))
    # print(len(re_binar[0]))
    # for i in range(len(re_windowed)):
    #     # print(re_windowed[i])
    #     # print(re_binar[i])
    #     for j, char in enumerate(re_windowed[i]):
    #         print(char, end=sep)
    #         if re_binar[i][j] == 1:
    #             print(space, end=sep)
    #     print('')

    assert len(re_binar) == len(re_windowed)
    assert len(re_binar[0]) == len(re_windowed[0])

    return re_windowed, re_binar, dict_chars
def tokenize(input_list, dict_chars):
    out = np.zeros((len(input_list), 64))
    for i, line in enumerate(input_list):
        l = np.zeros((64))
        for j, c in enumerate(line):
            try:
                l[j] = dict_chars[c]
            except KeyError:
                l[j] = dict_chars["OOV"]
        out[i] = l
    return(out)

training_file_name = "../data/src-sep-train.txt"
validation_file_name = "../data/src-sep-val.txt"
model_file_name = "transform2bin_3"
sep = ' '
mezera = '_'

vocab_size = 1138  # TODO - to be adjusted
maxlen = 64
embed_dim = 32  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer

batch_size = 64
epochs = 4
repeat = 2  # full epoch_num=epochs*repeat

with open(training_file_name, "r", encoding="utf-8") as f:  # with spaces
    final_file = f.read()
with open(validation_file_name, "r", encoding="utf-8") as ff:
    valid_file = ff.read()
formatted_input, formated_binary, dict_chars = sliding_window(final_file, sep, mezera)
# print(dict_chars)
x_val, y_val, dict_val = sliding_window(valid_file, sep, mezera)

input_tokens = tokenize(formatted_input, dict_chars)
validation_tokens = tokenize(x_val, dict_chars)
output_vals = formated_binary

if 1:
    model = model_func(vocab_size, maxlen, embed_dim, num_heads, ff_dim)
else:
    from keras.models import load_model
    model = load_model(model_file_name)

def F1_score(y_true, y_pred): #taken from old keras source code
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2*(precision*recall)/(precision+recall+K.epsilon())
    return f1_val

model.compile(optimizer="adam", loss="binary_crossentropy",
              metrics=["accuracy", "Precision", "Recall", F1_score])
for i in range(repeat):
    history = model.fit(
        input_tokens, output_vals, batch_size=batch_size, epochs=epochs,
        validation_data=(validation_tokens, y_val))
    K.clear_session()

    # print("saving model ...")
    model.save(model_file_name)
    # print("model saved")

sample, _, _ = sliding_window(final_file[:1000], sep, mezera)

model_test(sample, model_file_name, len(sample), 64, 32, dict_chars, mezera, sep)