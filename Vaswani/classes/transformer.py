import tensorflow as tf
from .encoder import Encoder
from .decoder import Decoder


class Transformer(tf.keras.Model):
    # num_heads for base model is 8 and for big model is 16
    # ff_dim for base model is 2048 and for big model is 4096
    def __init__(self, *, input_vocab_size, target_vocab_size, model_dim, num_layers=6, num_heads=16, ff_dim=4096, dropout_rate= 0.1):
        super().__init__()
        self.encoder = Encoder(num_layers=num_layers, model_dim=model_dim, num_heads=num_heads, ff_dim=ff_dim,
                               vocab_size=input_vocab_size, dropout_rate=dropout_rate)
        self.decoder = Decoder(num_layers=num_layers, model_dim=model_dim, num_heads=num_heads, ff_dim=ff_dim,
                               vocab_size=target_vocab_size, dropout_rate=dropout_rate)
        self.final_layer = tf.keras.layers.Dense(target_vocab_size)

    def call(self, inputs):
        context, x = inputs
        context = self.encoder(context)
        x = self.decoder(x, context)
        logits = self.final_layer(x)

        try:
            # Drop the keras mask, so it doesn't scale the losses/metrics.
            del logits._keras_mask
        except AttributeError:
            pass

        return logits
