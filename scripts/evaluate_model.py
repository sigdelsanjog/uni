import json
import numpy as np
import tensorflow as tf
import logging
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from mtm import TransformerModel, PositionalEncoding

# Set up logging
logging.basicConfig(filename='../uni/logs/evaluate_model.log', level=logging.INFO, 
                    format='%(asctime)s %(levelname)s %(message)s')

# Load the trained model with custom objects
model_path = '../uni/models/my_llm_model.keras'  # Update to .keras format
logging.info(f"Loading model from {model_path}")

with tf.keras.utils.custom_object_scope({
    "PositionalEncoding": PositionalEncoding,
    "TransformerModel": TransformerModel  # Include the TransformerModel
}):
    model = tf.keras.models.load_model(model_path)

# Load the saved tokenizer
tokenizer_path = '../uni/data/tokenizer.pkl'
with open(tokenizer_path, 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load token mappings for debugging or reverse lookup
tokens_file = '../uni/data/tokens.json'
with open(tokens_file, 'r') as f:
    tokens_data = json.load(f)

logging.info(f"Loaded token data from {tokens_file}")

# Function to preprocess input text using Keras Tokenizer
def preprocess_input(text, max_length=1000):
    # Debugging: Print the original text
    print(f"Original input text: {text}")
    
    # Tokenize and pad the input using the Keras tokenizer
    input_sequence = tokenizer.texts_to_sequences([text])
    input_padded = pad_sequences(input_sequence, maxlen=max_length, padding='post')

    # Debugging: Print the tokenized and padded input
    print(f"Tokenized and padded input: {input_padded}")

    return input_padded

# Evaluation function
def evaluate_model(test_input, expected_output=None):
    try:
        input_data = preprocess_input(test_input)
        predictions = model.predict(input_data)

        # Get predicted indices with the highest probability
        predicted_indices = np.argmax(predictions[0], axis=-1)

        # Debugging: Print predicted indices
        print(f"Predicted indices: {predicted_indices}")

        # Convert predicted indices back to words using Keras tokenizer's reverse mapping
        predicted_tokens = tokenizer.sequences_to_texts([predicted_indices])

        # Join the predicted tokens to form the predicted output
        result = ' '.join(predicted_tokens)
        logging.info(f"Input: {test_input}")
        logging.info(f"Predicted Output: {result}")

        # ToDo: Compare with expected output and calculate accuracy

        if expected_output:
            logging.info(f"Expected Output: {expected_output}")
            accuracy = np.mean([a == b for a, b in zip(result.split(), expected_output.split())]) * 100
            print(f"Accuracy: {accuracy:.2f}%")

            return result

    except Exception as e:
        logging.error(f"Error during evaluation: {str(e)}")
        raise

# Load sample test data for evaluation
test_samples = [
    {
        "input": "What is the academic experience of Sanjog Sigdel?",
        "output": "Sanjog Sigdel is a lecturer since September 2024 in the Department of Computer Science and Engineering at Kathmandu University, involved in various subjects including Artificial Intelligence and Database Management Systems."
    },
    {
        "input": "What are Sanjog Sigdel's research interests?",
        "output": "Sanjog Sigdel's research interests include gait analysis and eHealth architecture, with multiple publications in relevant fields."
    },
    {
        "input": "What subjects does Sanjog Sigdel teach?",
        "output": "Sanjog Sigdel teaches Digital Logic, Operating Systems, Artificial Intelligence, Database Management Systems, Data Mining, Web Technology, and Software Project Management."
    },
    {
        "input": "What is Sanjog Sigdel's current position?",
        "output": "Sanjog Sigdel is currently a lecturer in the Department of Computer Science and Engineering at Kathmandu University since September 2024."
    },
    {
        "input": "Can you list some of Sanjog Sigdel's publications?",
        "output": "Some of Sanjog Sigdel's publications include 'A Summary on Above the Clouds: A Berkeley View of Cloud Computing' and 'Gait Analysis in Early Identification of Cardiovascular Diseases Using MPU6050 Sensor.'"
    }
]

# Evaluate the model on each test sample
for sample in test_samples:
    predicted_output = evaluate_model(sample["input"], sample.get("output"))

    # Log the result
    print(f"Input: {sample['input']}")
    print(f"Predicted Output: {predicted_output}")
    if "output" in sample:
        print(f"Expected Output: {sample['output']}")
