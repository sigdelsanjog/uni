import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

# Import EncoderLayer and PositionalEncoding from mini_transformer_model
from mtm import TransformerModel, EncoderLayer, PositionalEncoding

# Load the tokens data from tokens.json (Numeric tokens)
tokens_file_path = '../uni/data/tokens.json'

with open(tokens_file_path, 'r', encoding='utf-8') as tokens_file:
    tokens_data = json.load(tokens_file)

# Extract input and output data
input_data = []
output_data = []

# Assuming tokens.json structure: { 'faculty1': {'original_tokens': [...], 'numeric_tokens': [...]}, ... }
for faculty, tokens in tokens_data.items():
    input_data.append(tokens['numeric_tokens'][:-1])  # Use all except the last token for input
    output_data.append(tokens['numeric_tokens'][1:])  # Use the tokens shifted by one for output

# Determine the maximum sequence length for padding
max_length = 1000  # Set a fixed length (can adjust based on analysis)

# Pad the input and output sequences to ensure uniform length
input_data = pad_sequences(input_data, maxlen=max_length, padding='post')
output_data = pad_sequences(output_data, maxlen=max_length, padding='post')

# Find the maximum token index in output_data
max_token_value = max(max(seq) for seq in output_data)
print(f"Maximum token value in output data: {max_token_value}")

# Adjust the vocab_size based on the maximum token value
vocab_size = max_token_value + 1  # +1 to include the highest token index

# Ensure output data is converted to one-hot encoding if needed (multi-class classification)
output_data = np.array([to_categorical(seq, num_classes=vocab_size) for seq in output_data])

# Print the shapes of input_data and output_data
print(f"Shape of Input Data: {input_data.shape}")
print(f"Shape of Output Data: {output_data.shape}")

# Model parameters
num_layers = 2
d_model = 64
num_heads = 4
dff = 128
input_vocab_size = vocab_size  # Use the adjusted vocabulary size
target_vocab_size = vocab_size
max_seq_len = max_length

# Initialize the Transformer model
transformer_model = TransformerModel(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=input_vocab_size,
    target_vocab_size=target_vocab_size,
    max_seq_len=max_seq_len
)

# Compile the model
transformer_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Custom callback to use TQDM for a progress bar
class CustomTqdmCallback(Callback):
    def on_epoch_begin(self, epoch, logs=None):
        self.epochs = self.params['epochs']
        self.epoch_progress_bar = tqdm(total=self.epochs, desc=f"Epoch {epoch+1}/{self.epochs}", position=0, leave=True)

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_progress_bar.update(1)
        self.epoch_progress_bar.close()

    def on_batch_end(self, batch, logs=None):
        self.epoch_progress_bar.set_postfix(logs)

# Configure dataset using tf.data for efficient multi-threaded loading
batch_size = 1
dataset = tf.data.Dataset.from_tensor_slices((input_data, output_data))
dataset = dataset.shuffle(buffer_size=len(input_data)).batch(batch_size)

# Train the model with a progress bar
transformer_model.fit(dataset, epochs=20, callbacks=[CustomTqdmCallback()], verbose=1)

# Save the trained model
model_save_path = '../uni/models/my_llm_model.keras'
transformer_model.save(model_save_path)

print("Model training complete and saved.")
