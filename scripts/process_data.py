import nltk
from nltk.tokenize import word_tokenize
import string
import json
import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Download NLTK resources if needed
nltk.download('punkt')

# Load the combined data from JSON
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
numeric_sequences = {}
all_texts = []

# Function to preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    return tokens

# Preprocess content for each faculty
for faculty, content in data.items():
    if isinstance(content, str):
        tokens = preprocess_text(content)
        tokens_data[faculty] = tokens  # Store original tokens
        all_texts.append(content)  # Collect texts for tokenization
        print(f"Tokens for {faculty}: {tokens[:10]}...")  # Preview tokens
    else:
        print(f"Content for {faculty} is not a string. Skipping this faculty.")

# Tokenization using Keras Tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_texts)

# Convert each faculty's tokens to numeric sequences
for faculty in tokens_data:
    numeric_sequences[faculty] = tokenizer.texts_to_sequences([data[faculty]])[0]

# Define maximum length for padding
max_length = max(len(seq) for seq in numeric_sequences.values())

# Pad sequences
for faculty in numeric_sequences:
    numeric_sequences[faculty] = pad_sequences([numeric_sequences[faculty]], maxlen=max_length, padding='post')[0]

# Save both original tokens and numeric sequences to a JSON file
tokens_file_path = '../uni/data/tokens.json'
try:
    tokens_data_combined = {
        faculty: {
            "original_tokens": tokens_data[faculty],
            "numeric_tokens": numeric_sequences[faculty].tolist()
        }
        for faculty in tokens_data
    }
    
    with open(tokens_file_path, 'w', encoding='utf-8') as tokens_file:
        json.dump(tokens_data_combined, tokens_file, ensure_ascii=False, indent=4)
    print(f"Tokens (both original and numeric) saved successfully to: {tokens_file_path}")
except Exception as e:
    print(f"An error occurred while saving tokens: {e}")

# Save the tokenizer for future use
with open('../uni/data/tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
