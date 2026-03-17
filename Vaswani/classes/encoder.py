import tensorflow as tf
from .attentions import GlobalSelfAttention
from .feed_forward import FeedForward
from .positional_embedding import PositionalEmbedding

# Class for the multiple layers of the encoder
class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, model_dim, num_heads, ff_dim, dropout_rate=0.1):
        super().__init__()
        self.gsa = GlobalSelfAttention(
            num_heads=num_heads,
            key_dim=model_dim,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(model_dim, ff_dim)

    def call(self, x):
        x = self.gsa(x)
        x = self.ffn(x)

        return x


class Encoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, model_dim, num_heads, ff_dim, vocab_size, dropout_rate=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.pos_embedding = PositionalEmbedding(vocab_size, model_dim)
        self.encoder_layers = [EncoderLayer(
            model_dim=model_dim,
            num_heads=num_heads,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate
        ) for _ in range(num_layers)]
        self.dropout = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x):
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.encoder_layers[i](x)

        return x
