import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, LayerNormalization, Dropout

# Positional Encoding Class
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.positional_encoding = self.get_positional_encoding(max_len, d_model)

    def get_positional_encoding(self, max_len, d_model):
        positions = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        positional_encoding = np.zeros((max_len, d_model))
        positional_encoding[:, 0::2] = np.sin(positions * div_term)
        positional_encoding[:, 1::2] = np.cos(positions * div_term)
        return tf.constant(positional_encoding[np.newaxis, ...], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.positional_encoding[:, :tf.shape(inputs)[1], :]

# Multi-Head Attention Class
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.depth = d_model // num_heads

        self.query_dense = Dense(d_model)
        self.key_dense = Dense(d_model)
        self.value_dense = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth)."""
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, query, key, value, mask=None):
        batch_size = tf.shape(query)[0]

        query = self.query_dense(query)
        key = self.key_dense(key)
        value = self.value_dense(value)

        query = self.split_heads(query, batch_size)
        key = self.split_heads(key, batch_size)
        value = self.split_heads(value, batch_size)

        attention_weights = tf.matmul(query, key, transpose_b=True)
        attention_weights /= tf.math.sqrt(tf.cast(self.depth, tf.float32))

        if mask is not None:
            attention_weights += (mask * -1e9)

        attention_weights = tf.nn.softmax(attention_weights, axis=-1)
        attention_output = tf.matmul(attention_weights, value)

        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        output = tf.reshape(attention_output, (batch_size, -1, self.num_heads * self.depth))
        return self.dense(output)

# Feed-Forward Network
def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        Dense(dff, activation='relu'),
        Dense(d_model)
    ])

# Encoder Layer Class
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

    def call(self, x, training, mask=None):
        attn_output = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
