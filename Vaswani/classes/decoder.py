import tensorflow as tf
from .attentions import CausalSelfAttention
from .attentions import CrossAttention
from .feed_forward import FeedForward
from .positional_embedding import PositionalEmbedding


class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, *, model_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(DecoderLayer, self).__init__()
        self.csa = CausalSelfAttention(
            num_heads=num_heads,
            key_dim=model_dim,
            dropout=dropout_rate
        )
        self.ca = CrossAttention(
            num_heads=num_heads,
            key_dim=model_dim,
            dropout=dropout_rate
        )
        self.ffn = FeedForward(model_dim, ff_dim)

    def call(self, x, context):
        x = self.csa(x=x)
        x = self.ca(x=x, context=context)
        x = self.ffn(x)

        return x


class Decoder(tf.keras.layers.Layer):
    def __init__(self, *, num_layers, model_dim, num_heads, ff_dim, vocab_size, dropout_rate=0.1):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.model_dim = model_dim
        self.pos_embedding = PositionalEmbedding(vocab_size, model_dim)
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.decoder_layers = [DecoderLayer(
                model_dim=model_dim,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout_rate=dropout_rate
        ) for _ in range(num_layers)]

    def call(self, x, context):
        x = self.pos_embedding(x)
        x = self.dropout(x)

        for i in range(self.num_layers):
            x = self.decoder_layers[i](x, context)

        return x
