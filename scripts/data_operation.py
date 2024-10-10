import os
import re

# Set the base directory
base_directory = '../uni/data/raw/'  # Adjust this path as necessary

# Resolve the absolute path for clarity
absolute_base_directory = os.path.abspath(base_directory)
print(f"Base directory set to: {absolute_base_directory}")

# Initialize a list to hold faculty names
faculties = []  # This should be a list, not a dictionary
# Initialize a dictionary for faculty data
faculties_data = {}

# Check if the base directory exists
if not os.path.exists(absolute_base_directory):
    raise FileNotFoundError(f"The specified path does not exist: {absolute_base_directory}")

# List all files in the base directory
print("Checking files in the directory:")
for filename in os.listdir(absolute_base_directory):
    print(f"Found file: {filename}")  # Print each file found

    # Check if the file is a text file
    if filename.endswith('.txt'):
        faculty_name = filename[:-4]  # Get faculty name without .txt
        faculties.append(faculty_name)  # Correctly append to the list
        
        # Read the content of the file
        with open(os.path.join(absolute_base_directory, filename), 'r', encoding='utf-8') as file:
            faculties_data[faculty_name] = file.read()  # Store in dictionary

# Log total faculties processed
print(f"Total faculties processed: {len(faculties)}")
print(f"Faculties: {faculties}")

# Preprocess the text for each faculty
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Preprocess each faculty's data
for faculty_name in faculties:
    faculties_data[faculty_name] = preprocess_text(faculties_data[faculty_name])

print("Preprocessing complete.")

# Save processed data to a new directory
output_directory = '../uni/data/processed/'  # Adjust this as necessary
os.makedirs(output_directory, exist_ok=True)

def save_processed_data(faculties_data, output_directory):
    for faculty_name, text in faculties_data.items():
        with open(os.path.join(output_directory, f"{faculty_name}_processed.txt"), 'w', encoding='utf-8') as file:
            file.write(text)

# Save the processed data
save_processed_data(faculties_data, os.path.abspath(output_directory))
print(f"Processed data saved to {output_directory}.")
