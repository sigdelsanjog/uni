import nltk
from nltk.tokenize import word_tokenize
import string
import json
import os
import pickle  # Import pickle for saving the tokenizer
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

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

# Initialize a dictionary to store tokens and padded sequences
tokens_data = {}
all_texts = []

# Function to preprocess the text
def preprocess_text(text):
    # Normalize text to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    return tokens

# Preprocess content for each faculty
for faculty, content in data.items():
    # Ensure content is a string before processing
    if isinstance(content, str):
        tokens = preprocess_text(content)
        tokens_data[faculty] = tokens  # Save tokens under the faculty's name
        all_texts.append(content)  # Collect all texts for tokenization
        print(f"Tokens for {faculty}: {tokens[:10]}...")  # Show first 10 tokens
    else:
        print(f"Content for {faculty} is not a string. Skipping this faculty.")

# Tokenization using Keras Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)

# Convert each faculty's tokens to sequences
for faculty in tokens_data:
    tokens_data[faculty] = tokenizer.texts_to_sequences([data[faculty]])[0]

# Define maximum length for padding
max_length = max(len(seq) for seq in tokens_data.values())

# Pad sequences
for faculty in tokens_data:
    tokens_data[faculty] = pad_sequences([tokens_data[faculty]], maxlen=max_length, padding='post')[0]

# Save the tokens and tokenizer to a JSON file
tokens_file_path = '../uni/data/tokens.json'
try:
    # Convert ndarray to list for JSON serialization
    tokens_data_list = {faculty: tokens.tolist() for faculty, tokens in tokens_data.items()}
    
    with open(tokens_file_path, 'w', encoding='utf-8') as tokens_file:
        json.dump(tokens_data_list, tokens_file, ensure_ascii=False, indent=4)
    print(f"Tokens saved successfully to: {tokens_file_path}")
except Exception as e:
    print(f"An error occurred while saving tokens: {e}")

# Save the tokenizer for future use
with open('../uni/data/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
