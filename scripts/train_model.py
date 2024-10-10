import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, TimeDistributed
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

# Load the training data
training_data_file_path = '../uni/data/training_data.json'

with open(training_data_file_path, 'r', encoding='utf-8') as training_file:
    training_data = json.load(training_file)

# Check the length of the training data
print(f"Number of training samples: {len(training_data)}")

# Create a mapping of words to indices
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

# Create a unique set of classes
unique_classes = sorted(set(flattened_output_data))
class_to_index = {cls: idx for idx, cls in enumerate(unique_classes)}
print("Class to Index Mapping:", class_to_index)

# Convert output labels to indices for each input
output_indices = []
for outputs in output_data:
    output_indices.append([class_to_index[label] for label in outputs])

# Determine the maximum length for consistent input and output sequences
max_length = 10000  # Set a fixed length to ensure uniformity
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

# Define a simplified model architecture
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=8, input_length=max_length))
model.add(LSTM(16, return_sequences=True))
model.add(TimeDistributed(Dense(len(unique_classes), activation='softmax')))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

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
model.fit(dataset, epochs=10, callbacks=[CustomTqdmCallback()], verbose=0)

# Save the model
model.save('../uni/models/my_llm_model.h5')

print("Model training complete and saved.")
