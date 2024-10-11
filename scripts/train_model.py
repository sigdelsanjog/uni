import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm

# Import EncoderLayer and PositionalEncoding from mini_transformer_model
from mtm import EncoderLayer, PositionalEncoding

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

# Define the Transformer model architecture
import tensorflow as tf
from mtm import PositionalEncoding  # Ensure PositionalEncoding is correctly imported
# Import EncoderLayer if it's defined in another file

class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, max_seq_len, rate=0.1, trainable=True, dtype='float32', name="transformer_model"):
        super(TransformerModel, self).__init__(name=name, trainable=trainable, dtype=dtype)  # Pass dtype, name, and trainable to the superclass

        self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len=max_seq_len)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(rate)

        self.final_layer = tf.keras.layers.Dense(target_vocab_size, activation='softmax')

        # Store hyperparameters to return in get_config
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.dff = dff
        self.input_vocab_size = input_vocab_size
        self.target_vocab_size = target_vocab_size
        self.max_seq_len = max_seq_len
        self.rate = rate

    def call(self, x, training=False):
        seq_len = tf.shape(x)[1]
        
        # Add embedding and positional encoding
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(tf.shape(x)[-1], tf.float32))  # Scaling the embeddings
        x = self.pos_encoding(x)  # (batch_size, input_seq_len, d_model)

        x = self.dropout(x, training=training)

        # Pass through each encoder layer
        for enc_layer in self.enc_layers:
            x = enc_layer(x, training=training)  # Pass training as a keyword argument

        # Final classification layer
        x = self.final_layer(x)

        return x  # Final encoded output

    def get_config(self):
        # This method is required for the model to be saved and loaded correctly
        config = super(TransformerModel, self).get_config()
        config.update({
            'num_layers': self.num_layers,
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'dff': self.dff,
            'input_vocab_size': self.input_vocab_size,
            'target_vocab_size': self.target_vocab_size,
            'max_seq_len': self.max_seq_len,
            'rate': self.rate,
            'trainable': self.trainable,
            'dtype': self._dtype_policy.name,  # Use dtype from TensorFlow's dtype policy
            'name': self.name
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)  # Pass all config parameters directly to the constructor


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
