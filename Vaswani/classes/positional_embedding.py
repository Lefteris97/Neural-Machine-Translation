import numpy as np
import tensorflow as tf


def positional_encoding(length, depth):  # length: the length of the sentences | depth: the depth that represents the dimensionality of the encoding
    depth = depth/2

    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth

    angle_rates = 1 / (10000 ** depths)
    angle_radians = positions * angle_rates

    pos_encoding = np.concatenate([np.sin(angle_radians), np.cos(angle_radians)], axis=-1)

    return tf.cast(pos_encoding, dtype=tf.float32)


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, vocab_size, model_dim):  # model_dim: the dimensionality of the embedding layer
        super().__init__()
        self.model_dim = model_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, model_dim, mask_zero=True)
        self.pos_encoding = positional_encoding(length=2048, depth=model_dim)

    def compute_mask(self, *args, **kwargs):
        return self.embedding.compute_mask(*args, **kwargs)

    def call(self, x):
        x = self.embedding(x)

        length = tf.shape(x)[1]

        # Scale the embeddings to maintain variance relative to the dimension of the model
        x *= tf.math.sqrt(tf.cast(self.model_dim, tf.float32))

        # Add the positional encodings to the embeddings
        x = x + self.pos_encoding[tf.newaxis, :length, :]

        return x
