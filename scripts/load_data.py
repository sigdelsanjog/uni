import json

# Load the combined data from the JSON file
combined_file_path = '../uni/data/combined_data.json'

with open(combined_file_path, 'r', encoding='utf-8') as json_file:
    data = json.load(json_file)

# Print the keys (faculty names) and a snippet of their content
for faculty, content in data.items():
    print(f"Faculty: {faculty}\nContent Snippet: {content[:100]}...\n")  # Print first 100 characters
