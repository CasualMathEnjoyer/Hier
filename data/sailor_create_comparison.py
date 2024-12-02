import json


def process_files(file_paths, output_file, annotations):
    # Read all files and align their line counts
    file_lines = [open(file_path, 'r').readlines() for file_path in file_paths]

    # Normalize the number of lines
    min_lines = min(len(lines) for lines in file_lines)
    file_lines = [lines[:min_lines] for lines in file_lines]

    # Remove empty trailing lines if present
    file_lines = [[line.strip() for line in lines if line.strip()] for lines in file_lines]

    # Create JSON structure with annotations
    combined_data = []
    for i in range(min_lines):
        line_data = {"line_number": i + 1}
        line_data.update({
            annotations[j]: file_lines[j][i] for j in range(len(file_paths))
        })
        combined_data.append(line_data)

    # Write JSON to output file
    with open(output_file, 'w') as outfile:
        json.dump(combined_data, outfile, indent=4)


# Define file paths, annotations, and output file
file_paths = [
    'sailor_test_src.txt',
    'sailor_separated_rsc.txt',
    'sailor_test_trl.txt',
    'SAILOR_output_prediction.txt'
]
# file_paths = [
#     'sailor_test_trl.txt',
#     'SAILOR_output_prediction.txt'
# ]
annotations = [
    "src_gardiner",
    "src_sep_pred",
    "target_transl",
    "predic_transl"
]
# annotations = [
#     "target_transl",
#     "predic_transl"
# ]
output_file = "sailor_combined_output.json"

# Execute the function
process_files(file_paths, output_file, annotations)
