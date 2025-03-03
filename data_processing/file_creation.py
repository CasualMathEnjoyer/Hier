def remove_spaces(input_file_name, output_file_name, space, sep):
    # removes spaces in a file to prepare it for training
    f = open(input_file_name, "r", encoding="utf-8").read()
    sf = open(output_file_name, "w",  encoding="utf-8")
    # remove spaces section
    if sep == ' ':
        for i, char in enumerate(f):
            if char != space:
                sf.write(char)
    elif sep == '':
        for char in f:
            if char != space:
                sf.write(char)
    else:
        Exception('not selected valid separator')
    sf.close()

import numpy as np


def create_binary_file(unspaced_text, spaced_text, output_name, sent_len, spacing=' ', endline='\n', charsplit=''):
    output = []
    spaced_text = spaced_text.split(endline)
    unspaced_text = unspaced_text.split(endline)
    spaced_text_char = []
    unspaced_text_char = []

    # creates lists of independent characters in each sentence
    if charsplit == '':
        for i in range(len(unspaced_text)):
            item2_s = []
            item2_u = []
            for c in unspaced_text[i]:
                item2_u.append(c)
            for c in spaced_text[i]:
                item2_s.append(c)
            # print(item2_u)
            # print(item2_s)
            unspaced_text_char.append(item2_u)
            spaced_text_char.append(item2_s)
    else:
        for i in range(len(unspaced_text)):
            item2_u = unspaced_text[i].split(charsplit)
            item2_s = spaced_text[i].split(charsplit)
            # print(item2_u)
            # print(item2_s)

            it2_u = [s for s in item2_u if s != '']
            unspaced_text_char.append(it2_u)
            it2_s = [s for s in item2_s if s != '']
            spaced_text_char.append(it2_s)

    # print(spaced_text_char)
    # print(unspaced_text_char)

    # TODO - check below
    # unspaced_text_char = [s for s in unspaced_text_char if s != []]  # this shouldn't be needed
    # spaced_text_char = [s for s in spaced_text_char if s != []]

    # creating the 0 and 1
    for i in range(len(spaced_text_char)):
        spaces = 0
        # unspaced_text = unspaced_text.split(' ')
        list_sent = np.ones(sent_len)

        list_sent[0] = 0

        for j in range(1, len(spaced_text_char[i])):
            if j < sent_len + spaces + 0:
                assert j - spaces < sent_len
                if spaced_text_char[i][j] != spacing:
                    list_sent[j - spaces] = 0
                else:
                    spaces += 1
                    list_sent[j - spaces] = 1
                    j += 1
            else:
                pass

        print('')
        print('i=', i, ' len=', len(unspaced_text_char[i]))
        print(unspaced_text_char[i])
        print(spaced_text_char[i])
        print(list_sent)
        output.append(list_sent)
        #output.write('0') # za tecku nakonci
    output = np.asarray(output)
    np.save(output_name, output)

def clean_data(input_file : str):
    output = ''
    for i, char in enumerate(input_file):
        try:
            if char != ' ' or input_file[i+1] != ' ':
                output += char
        except IndexError:
            pass
    return output

def main():
    # input_file_name = "smallervoc_fr_unspaced.txt"
    # final_file_name = "smallervoc_fr.txt"
    # space_file_name = "../data/space_file.npy"
    # spacing = ' '
    # sep = ''
    # sent_len = 90

    # input_file_name = "../data/hier.txt"
    # final_file_name = "../data/hier_sep.txt"
    # space_file_name = "../data/space_hier_long.npy"

    input_file_name = "hier_short.txt"
    final_file_name = "hier_short_sep.txt"
    space_file_name = "space_hier.npy"
    spacing = '_'
    sep = ' '

    remove_spaces(final_file_name, input_file_name, spacing, sep)

    input_file = open(input_file_name, "r", encoding="utf-8").read()
    cleared_data = clean_data(input_file)
    input_file = open(input_file_name, "w", encoding="utf-8")
    input_file.write(cleared_data)
    input_file.close()

    input_file = open(input_file_name, "r", encoding="utf-8").read()
    final_file = open(final_file_name, "r", encoding="utf-8").read()

    # create_binary_file(input_file, final_file, space_file_name, sent_len, spacing=spacing,  charsplit=sep)  # change sent_len if needed!

    pass

if __name__ == '__main__':
    main()

# QUESTIONS:
# preskalovat data tak, aby kazda sentence mela stejnou delku
# ted je neuronka biased smerem k 0 nebo 1 v zavislosti na volbe filler znaku
# pridavat dvojky jako filler (respektive preskalovat na 0, 0.5 a 1)

# NAPAD
# slo by jenom vzit hieroglyfy, udelat pro znaky embedding a potom udelat aplikaci
# ktera by pri psani nabizela jakoby autofil nebo autocorrect