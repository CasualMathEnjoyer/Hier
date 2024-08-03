# with open("data_3_clean") as d:
#     data = d.read().split("test line number: ")
#
# for i in range(len(data)):
#     data[i] = data[i].split("\n")
#     print(data[i])


import pandas as pd

# Load the clean document
file_path = '../data_3_clean'
with open(file_path, 'r') as file:
    lines = file.readlines()

# Extract predictions and valid translations
predictions = []
valid_translations = []
for i in range(0, len(lines), 2):
    predictions.append(lines[i].strip())
    valid_translations.append(lines[i + 1].strip())

# Function to find single-character substitution errors
def find_single_char_substitution_errors(predictions, valid_translations):
    substitution_errors = []

    for pred, valid in zip(predictions, valid_translations):
        if len(pred) != len(valid):
            continue  # Ignore lines with different lengths

        errors = []
        for i in range(len(pred)):
            if pred[i] != valid[i]:
                if (i > 0 and pred[i-1] != valid[i-1]) or (i < len(pred) - 1 and pred[i+1] != valid[i+1]):
                    continue  # Skip if there are adjacent mistakes
                errors.append((i, pred[i], valid[i]))

        if errors:
            substitution_errors.append({
                'Prediction': pred,
                'Valid Translation': valid,
                'Errors': errors
            })

    return substitution_errors

# Analyze the data for substitution errors
substitution_errors = find_single_char_substitution_errors(predictions, valid_translations)

# Create a DataFrame for the substitution errors
error_df = pd.DataFrame(substitution_errors)

# Display the DataFrame
print(error_df)

error_df.to_csv('substitution_errors.csv', index=False)