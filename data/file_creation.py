def remove_spaces(input_file_name, output_file_name, space):
    # removes spaces in a file to prepare it for training
    f = open(input_file_name, "r", encoding="utf-8").read()
    sf = open(output_file_name, "w",  encoding="utf-8")
    # remove spaces section
    for char in f:
        if char != space:
            sf.write(char)
    sf.close()

import numpy as np
def create_binary_file(unspaced_text, spaced_text, output_name, sent_len, spacing=' ', endline='\n'):
    output = []
    spaced_text = spaced_text.split(endline)
    print(spaced_text)
    for i, sentence in enumerate(unspaced_text.split(endline)):
        spaces = 0
        # unspaced_text = unspaced_text.split(' ')
        list_sent = np.zeros(sent_len)
        for j, letter in enumerate(sentence):
            if j < len(sentence) - 1:
                if spaced_text[i][j + spaces + 1] != spacing:
                    list_sent[j] = 0
                else:
                    list_sent[j] = 1
                    spaces += 1
        print(sentence)
        print(list_sent)
        output.append(list_sent)
        #output.write('0') # za tecku nakonci
    output = np.asarray(output)
    np.save(output_name, output)

def main():
    input_file_name = "smallervoc_fr_unspaced.txt"
    final_file_name = "smallervoc_fr.txt"
    space_file_name = "space_file.npy"
    spacing = ' '
    # input_file_name = "src-ctest.txt"
    # final_file_name = "src-sep-ctest.txt"
    # space_file_name = "space_hier.npy"
    # spacing = '_'

    remove_spaces(final_file_name, input_file_name, spacing)

    input_file = open(input_file_name, "r", encoding="utf-8").read()
    final_file = open(final_file_name, "r", encoding="utf-8").read()

    create_binary_file(input_file, final_file, space_file_name, 90)  # change sent_len if needed!
    pass

if __name__ == '__main__':
    main()