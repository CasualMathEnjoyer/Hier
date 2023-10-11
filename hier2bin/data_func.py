
import numpy as np
"""
def create_letter_array(final_file: str, sentence_split_point='\n', sep=''):
    slovnikk = []
    sent_len = 0
    for line in final_file.split(sentence_split_point):
        if sep != '':
            line = line.split(' ')
        if len(line) > sent_len:
            sent_len = len(line)
        for letter in line:
            if letter not in slovnikk:
                slovnikk.append(letter)
    slovnikk.append('sos')
    slovnikk.append('eos')
    slovnikk.append('OOV')
    return slovnikk, sent_len
def listify_data(input_file: str, charsep):
    input_list, list_chars = [], []
    for i, line in enumerate(input_file.split('\n')):
        ll = []
        for item in line.split(charsep):
            ll.append(item)
            if item not in list_chars:
                list_chars.append(item)
        ll.pop()  # removes last item as it was an empty string
        input_list.append(ll)
    print(input_list)

    if ' ' not in list_chars: list_chars.append(' ')
    list_chars.append('<bos>')
    list_chars.append('<eos>')
    list_chars.append('OOV')
    dict_chars = {j: i for i, j in enumerate(list_chars)}

    return input_list, dict_chars
def data_windowing(input_list: list, binary: list):
    window, core, left, right = 128, 64, 32, 32
    assert len(input_list) == len(binary)
    assert len(input_list[0]) == len(binary[0])

    # loop to make it 1D
    flatten_input = [j for sub in input_list for j in sub]
    flatten_binar = [j for sub in binary for j in sub]

    assert len(flatten_input) == len(flatten_binar)

    # simple sentence destroyer
    i = 0
    reform_input, reform_binar = [], []
    while True:
        line, line_n = [], []
        while (len(line) < window):
            if i >= len(flatten_input):
                break
            line.append(flatten_input[i])
            line_n.append(flatten_binar[i])
            i += 1
        reform_input.append(line)
        reform_binar.append(line_n)
        if i > len(flatten_input):
            break

    # for i in range(len(reform_input)):
    #     print(reform_input[0])
    #     print(reform_binar[0])
    #     print('')

    reform_binar = np.array(reform_binar)

    return reform_input, reform_binar

    # windows around sentences
def re_windowing_data(input_file: str, binar: np.array, sep):
    window, core, left, right = 128, 64, 32, 32
    assert right + left + core == window
    binar = binar.flatten()
    input_split = input_file.split(sep)
    l = len(input_split)
    re_windowed, re_binar, list_chars = [], [], []
    i, pos_n = 0, 0
    while True:
        pos, skipped = 0, 0
        line, line_n = [], []
        # line.append('<bos>')
        while pos < 128 + skipped:
            element = input_split[i * window + pos]
            if element != '':
                if "\n" in element:
                    element = element.replace("\n", "")
                line.append(element)
                line_n.append(binar[pos_n])

                if element not in list_chars:
                    list_chars.append(element)
                pos_n += 1
            else:
                skipped += 1
            pos += 1
        # line.append('<eos>')
        re_windowed.append(line)
        re_binar.append(line_n)
        i += 1
        if i * window + pos > l:
            break
    list_chars.append('OOV')
    dict_chars = {j: i for i, j in enumerate(list_chars)}

    assert len(re_binar) == len(re_windowed)
    assert len(re_binar[0]) == len(re_windowed[0])
    re_binar = np.array(re_binar)

    return re_windowed, re_binar, dict_chars
def vectorise(final_file, input_dim, slovnik, mezera=' ', sep=''):
    (radky, sent_len, embed_dim) = input_dim
    input_text = np.zeros((radky, sent_len, embed_dim))
    for l, line in enumerate(final_file.split('\n')):
        k = 0
        if sep != '':
            line = line.split(sep)
        for i, letter in enumerate(line):
            if i < sent_len:
                assert letter != mezera
                try:
                    input_text[l][i][slovnik[letter]] = 1
                except KeyError:
                    input_text[l][i][slovnik['OOV']] = 1
                k += 1
        while k < sent_len:
            input_text[l][k][slovnik['eos']] = 1
            # input_text[l][k][slovnik[mezera]] = 1
            k += 1

    for line in input_text:  # kontrola zda zadny vektor neni nulovy
        for vector in line:
            soucet = 0
            for element in vector:
                soucet += element
            assert soucet != 0

    return (input_text)
def create_binary_simple(spaced_text : str, spacing, endline, charsplit):
    output, spaced_text_char, unspaced_text_char = [], [], []
    spaced_text = spaced_text.split(endline)
    # unspaced_text = unspaced_text.split(endline)
    output = []

    for i, line in enumerate(spaced_text):
        line = line.split(charsplit)
        ll = []
        for j, char in enumerate(line):
            if char != spacing:
                ll.append(0)
            else:
                ll.pop()
                ll.append(1)
        ll.pop() # there is one more zero at the end absolutely not necessary
        output.append(ll)

    return output
# # create a vector of input file
# print("create a vector of input file")
# list_input, dict_chars = df.listify_data(input_file, sep)
#
# # creating binary list file
# print("creating binary list file")
# list_binary = df.create_binary_simple(final_file, mezera, '\n', sep)
#
# # formatting input and output
# print("formatting input and output")
# formatted_input, formated_binary = df.data_windowing(list_input, list_binary)
"""

# it doesnt use the last elements - need to fix
def re_windowing_data_nobinar(output_file: str, sep, space):
    window, core, left, right = 128, 64, 32, 32
    assert right + left + core == window

    output_file = output_file.split(sep)
    print(output_file)
    l = len(output_file)
    print(l)
    re_windowed, re_binar, list_chars = [], [], []
    i = 0

    while True:
        pos, skipped = 0, 0
        line, line_n = [], []
        # line.append('<bos>')
        while pos < window + skipped:
            element = output_file[i * window + pos]
            if element != space:
                if "\n" in element:
                    element = element.replace("\n", "")
                if element not in list_chars:
                    list_chars.append(element)
                line.append(element)
                re_binar.append(0)
            else:  # element was a space
                skipped += 1
                re_binar.pop()
                re_binar.append(1)
            pos += 1
        # line.append('<eos>')
        re_windowed.append(line)
        # I take a look at the last like without setting pos to 0 and it is too much so it stops
        i += 1
        if i * window + pos > l:
            break
    list_chars.append('OOV')
    dict_chars = {j: i for i, j in enumerate(list_chars)}

    num_line = len(re_windowed)
    re_binar = np.array(re_binar)
    re_binar = np.reshape(re_binar, (num_line, window))

    assert len(re_binar) == len(re_windowed)
    assert len(re_binar[0]) == len(re_windowed[0])

    return re_windowed, re_binar, dict_chars

def sliding_window(output_file: str, sep, space):
    window, step = 64, 20

    if sep != '':
        output_file = output_file.split(sep)
    l = len(output_file)

    re_windowed, re_binar, list_chars = [], [], []
    slide = 0
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
            else:  # element was a space
                skipped += 1
                re_binar.pop()
                re_binar.append(1)
            pos += 1
        # line.append('<eos>')
        re_windowed.append(line)
        # I take a look at the last like without setting pos to 0 and it is too much so it stops
        slide += 1
        if slide * step + pos > l:
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
def vectorise_list(file: list, embed_dim, radky, sent_len, dictionary, mezera):
    # takes list and creates array with the same dimensions of the embedded list

    input_text = np.zeros((radky, sent_len, embed_dim))
    for i, line in enumerate(file):
        for j, letter in enumerate(line):
            assert letter != mezera
            try:
                input_text[i][j][dictionary[letter]] = 1
            except KeyError:
                input_text[i][j][dictionary['OOV']] = 1

    for line in input_text:  # kontrola zda zadny vektor neni nulovy
        for vector in line:
            soucet = 0
            for element in vector:
                soucet += element
            assert soucet != 0

    return input_text
