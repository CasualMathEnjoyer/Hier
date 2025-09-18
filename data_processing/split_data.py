# This script takes a path to a txt file and creates a new file in the same folder
# containing the first 30% of the original file without splitting lines
# and without leaving an empty line at the end.
# The new file is named "<original_name>_30.txt".

import os
import sys

def create_30_percent_file(file_path: str):
    if not os.path.isfile(file_path) or not file_path.lower().endswith(".txt"):
        raise ValueError("Provided path must be a valid .txt file")

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total_chars = sum(len(line) for line in lines)
    cutoff_chars = int(total_chars * 0.3)

    collected_lines = []
    current_chars = 0
    for line in lines:
        if current_chars + len(line) > cutoff_chars and collected_lines:
            break
        collected_lines.append(line)
        current_chars += len(line)

    if collected_lines and collected_lines[-1].endswith("\n"):
        collected_lines[-1] = collected_lines[-1].rstrip("\n")

    new_file_path = os.path.join(
        os.path.dirname(file_path),
        f"{os.path.splitext(os.path.basename(file_path))[0]}_30.txt"
    )

    with open(new_file_path, "w", encoding="utf-8") as f:
        f.writelines(collected_lines)

    return new_file_path

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <path_to_txt_file>")
        sys.exit(1)

    try:
        output_file = create_30_percent_file(sys.argv[1])
        print(f"Created: {output_file}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
