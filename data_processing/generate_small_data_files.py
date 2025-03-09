import os

def save_first_n_lines(train_in_file_name, train_out_file_name, output_folder, n):
    """
    This function loads the given input and output files, reads the first n lines
    from both, and saves them to new files with the suffix '_n' before the '.txt' extension.
    It ensures no empty lines are added, especially any extra empty line appearing in the output.
    """
    # Reading the first n lines from the input file
    with open(train_in_file_name, 'r') as infile:
        input_lines = [line.rstrip('\n') for _, line in zip(range(n), infile)]  # Remove any trailing newline

    # Reading the first n lines from the output file
    with open(train_out_file_name, 'r') as outfile:
        output_lines = [line.rstrip('\n') for _, line in zip(range(n), outfile)]  # Remove any trailing newline

    # Get the base filenames without directory paths
    input_base_filename = os.path.basename(train_in_file_name)
    output_base_filename = os.path.basename(train_out_file_name)

    # Adding '_n' before the '.txt' extension
    input_filename_with_n = input_base_filename.replace('.txt', f'_{n}.txt')
    output_filename_with_n = output_base_filename.replace('.txt', f'_{n}.txt')

    # Saving the first n lines to new files in the specified output folder
    with open(os.path.join(output_folder, input_filename_with_n), 'w') as infile:
        infile.write('\n'.join(input_lines))

    with open(os.path.join(output_folder, output_filename_with_n), 'w') as outfile:
        outfile.write('\n'.join(output_lines))

# Example usage
train_in_file_name = "../data/src-sep-train.txt"
train_out_file_name = "../data/tgt-train.txt"
output_folder = "/home/katka/PycharmProjects/Hier/data/ramses_smaller"

for i in range(1, 68):
    n = i * 1000
    save_first_n_lines(train_in_file_name, train_out_file_name, output_folder, n)
