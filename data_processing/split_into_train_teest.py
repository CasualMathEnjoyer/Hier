# Script to split sentences from source and target files into training and testing datasets
import random

# Function to split data into training and testing datasets
def split_data(source_file, target_file, train_ratio=0.8):
    # Read the source and target files
    with open(source_file, 'r', encoding='utf-8') as src_file:
        source_lines = src_file.readlines()

    with open(target_file, 'r', encoding='utf-8') as tgt_file:
        target_lines = tgt_file.readlines()

    if len(source_lines) != len(target_lines):
        raise ValueError("Source and target files must have the same number of lines.")

    # Combine lines to maintain order during shuffling
    combined_lines = list(zip(source_lines, target_lines))

    # Shuffle the combined lines
    random.shuffle(combined_lines)

    # Determine split index
    split_index = int(len(combined_lines) * train_ratio)

    # Split into training and testing datasets
    train_data = combined_lines[:split_index]
    test_data = combined_lines[split_index:]

    return train_data, test_data

# Function to save data to files
def save_data(data, source_output, target_output):
    source_data, target_data = zip(*data)

    with open(source_output, 'w', encoding='utf-8') as src_out:
        src_out.writelines(source_data)

    with open(target_output, 'w', encoding='utf-8') as tgt_out:
        tgt_out.writelines(target_data)

# Input and output file paths
source_file = '../data/sailor_test_src.txt'
target_file = '../data/sailor_test_trl.txt'

train_source_output = 'train_src.txt'
train_target_output = 'train_trl.txt'
test_source_output = 'test_src.txt'
test_target_output = 'test_trl.txt'

# Split the data and save the results
try:
    train_data, test_data = split_data(source_file, target_file)

    save_data(train_data, train_source_output, train_target_output)
    save_data(test_data, test_source_output, test_target_output)

    print("Data successfully split and saved.")
except Exception as e:
    print(f"Error: {e}")