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
def create_binary_file(input, spaced_text, output_name, sent_len, spacing=' ', endline='\n'):
    output = []
    spaced_text.split(endline)
    spaces = 0
    for i, sentence in enumerate(input.split(endline)):
        list_sent = np.ones(sent_len)
        for j, letter in enumerate(sentence):
            if spaced_text[j + spaces + 1] != spacing:
                list_sent[j] = 0
            else:
                list_sent[j] = 1
                spaces += 1
        output.append(list_sent)
        #output.write('0') # za tecku nakonci
    output = np.asarray(output)
    np.save(output_name, output)

def main():
    remove_spaces("smallervoc_fr.txt", "smallervoc_fr_unspaced.txt", ' ')

    input_file_name = "smallervoc_fr_unspaced.txt"
    final_file_name = "smallervoc_fr.txt"
    space_file_name = "space_file.npy"
    input_file = open(input_file_name, "r", encoding="utf-8").read()
    final_file = open(final_file_name, "r", encoding="utf-8").read()

    create_binary_file(input_file, final_file, space_file_name, 90)  # change sent_len if needed!
    pass

if __name__ == '__main__':
    main()