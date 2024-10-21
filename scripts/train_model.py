import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

# Import EncoderLayer and PositionalEncoding from mini_transformer_model
from mtm import TransformerModel, EncoderLayer, PositionalEncoding

# Load the training data
training_data_file_path = '../uni/data/training_data.json'

with open(training_data_file_path, 'r', encoding='utf-8') as training_file:
    training_data = json.load(training_file)

# Tokenization
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts([data['input'] for data in training_data])

# Convert input text to sequences
input_data = tokenizer.texts_to_sequences([data['input'] for data in training_data])

# Prepare output data
output_data = [data['output'] for data in training_data]

# Flatten the output_data if necessary
flattened_output_data = []
for outputs in output_data:
    flattened_output_data.extend(outputs)

# Create a unique set of classes for outputs
unique_classes = sorted(set(flattened_output_data))
class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}

# Convert output labels to indices for each input
output_indices = []
for outputs in output_data:
    output_indices.append([class_to_index[label] for label in outputs])

# Determine the maximum length for consistent input and output sequences
max_length = 1000  # Set a fixed length to ensure uniformity
input_data = [seq[:max_length] for seq in input_data]
output_indices = [seq[:max_length] for seq in output_indices]

# Pad the input and output sequences to ensure uniform length
input_data = pad_sequences(input_data, maxlen=max_length, padding='post')
output_indices = pad_sequences(output_indices, maxlen=max_length, padding='post')

# Convert to one-hot encoding if multi-class classification
output_data = np.array([to_categorical(seq, num_classes=len(unique_classes)) for seq in output_indices])

# Print the shapes of input_data and output_data
print(f"Shape of Input Data: {input_data.shape}")
print(f"Shape of Output Data: {output_data.shape}")

# Model parameters
num_layers = 2
d_model = 64
num_heads = 4
dff = 128
input_vocab_size = len(tokenizer.word_index) + 1
target_vocab_size = len(unique_classes)
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
transformer_model.save('../uni/models/my_llm_model.keras')

print("Model training complete and saved.")
