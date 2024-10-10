import json
import os

# Load the tokenized data from the JSON file
tokens_file_path = '../uni/data/tokens.json'

try:
    with open(tokens_file_path, 'r', encoding='utf-8') as tokens_file:
        tokens_data = json.load(tokens_file)
except FileNotFoundError:
    raise FileNotFoundError(f"The specified JSON file does not exist: {tokens_file_path}")
except json.JSONDecodeError:
    raise ValueError("Error decoding JSON. Please check the file format.")

# Prepare training data
training_data = []

for faculty, tokens in tokens_data.items():
    # Create input-output pairs or structure as needed
    # Here, we're creating a simple pair where input is the faculty name, and output is the tokens
    training_data.append({
        'input': faculty,
        'output': tokens
    })

# Save the training data to a JSON file
training_data_file_path = '../uni/data/training_data.json'

with open(training_data_file_path, 'w', encoding='utf-8') as training_file:
    json.dump(training_data, training_file, ensure_ascii=False, indent=4)

print(f"Training data saved successfully to: {training_data_file_path}")
