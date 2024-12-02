import json
import os
from lib2to3.fixes.fix_input import context

# Path to the JSON file
file_path = '/home/katka/Documents/Hieroglyphs_sailor/preTranslations_Sailor.json'
output_path = '../data'

# Load the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Initialize lists to store extracted data
lines_with_dash = []
first_lists = []
second_lists = []

# Extract required data
for entry in data:
    if len(entry) > 4:  # Ensure the entry has enough elements
        lines_with_dash.append(entry[2])  # Line with '-'
        first_lists.append(entry[3])     # First list
        second_lists.append(entry[4])    # Second list

# removing last line as im unsure whether it's correctly formated
lines_with_dash.pop()
first_lists.pop()
second_lists.pop()

# Save the extracted data into separate files
with open(os.path.join(output_path, 'sailor_test_src.txt'), 'w') as src_file:
    line = "\n".join(lines_with_dash)
    line = line.replace("-", " ")
    # print(line)
    src_file.write(line)

with open(os.path.join(output_path, 'sailor_test_trl.txt'), 'w') as trl_file:
    for lst in first_lists:
        print(lst)
        pop_line = False
        line = "_".join(lst)

        print(line)
        line = line.replace("", " ")
        line = line[1:] # remove first space added in the line above
        line += '_ '  # adding last space at the end to be consistent with training data
        print(line)
        print()
        trl_file.write(line + "\n")

with open(os.path.join(output_path, 'sailor_test_trl2.txt'), 'w') as trl2_file:
    for lst in second_lists:
        trl2_file.write(" ".join(lst) + "\n")

print("Files saved: sailor_test_src.txt, sailor_test_trl.txt, sailor_test_trl2.txt")
