import nltk
from nltk.tokenize import word_tokenize
import string
import json
import os

# Download the necessary NLTK resources if not already downloaded
nltk.download('punkt')

# Load the combined data from the JSON file
combined_file_path = '../uni/data/combined_data.json'

try:
    with open(combined_file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
except FileNotFoundError:
    raise FileNotFoundError(f"The specified JSON file does not exist: {combined_file_path}")
except json.JSONDecodeError:
    raise ValueError("Error decoding JSON. Please check the file format.")

# Function to preprocess the text
def preprocess_text(text):
    # Normalize text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    return tokens

# Initialize a dictionary to store tokens
tokens_data = {}

# Preprocess content for each faculty
for faculty, content in data.items():
    # Ensure content is a string before processing
    if isinstance(content, str):
        tokens = preprocess_text(content)
        tokens_data[faculty] = tokens  # Save tokens under the faculty's name
        print(f"Tokens for {faculty}: {tokens[:10]}...")  # Show first 10 tokens
    else:
        print(f"Content for {faculty} is not a string. Skipping this faculty.")

# Save the tokens to a JSON file
tokens_file_path = '../uni/data/tokens.json'
try:
    with open(tokens_file_path, 'w', encoding='utf-8') as tokens_file:
        json.dump(tokens_data, tokens_file, ensure_ascii=False, indent=4)
    print(f"Tokens saved successfully to: {tokens_file_path}")
except Exception as e:
    print(f"An error occurred while saving tokens: {e}")
