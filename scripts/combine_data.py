import os
import json

# Directory with processed files
processed_dir = '../uni/data/processed/'
combined_file = '../uni/data/combined_data.json'

# Combine processed files into a dictionary
combined_data = {}

# Iterate through each processed file
for filename in os.listdir(processed_dir):
    if filename.endswith('_processed.txt'):  # Adjust this according to your processed file naming convention
        faculty_name = filename[:-14]  # Extract faculty name from the filename
        with open(os.path.join(processed_dir, filename), 'r', encoding='utf-8') as infile:
            content = infile.read()  # Read the content of the processed file
            combined_data[faculty_name] = content  # Add faculty name as key and content as value

# Save the combined data to a JSON file
with open(combined_file, 'w', encoding='utf-8') as outfile:
    json.dump(combined_data, outfile, ensure_ascii=False, indent=4)

print(f"Combined processed files into {combined_file}.")
