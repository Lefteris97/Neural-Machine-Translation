import tensorflow as tf


# Class for the common functionalities of the attentions used in a transformer
class BaseAttention(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
        self.ln = tf.keras.layers.LayerNormalization()
        self.add = tf.keras.layers.Add()


# for the Decoder-Encoder Attention
class CrossAttention(BaseAttention):
    def call(self, x, context):
        mha_output = self.mha(query=x, key=context, value=context)

        x = self.add([x, mha_output])
        x = self.ln(x)

        return x


# for the Encoder Self Attention
class GlobalSelfAttention(BaseAttention):
    def call(self, x):
        mha_output = self.mha(query=x, key=x, value=x)

        x = self.add([x, mha_output])
        x = self.ln(x)

        return x


# For the Decoder Self Attention
# Similar to Global Self Attention but with masking so the model doesn't see the last word
class CausalSelfAttention(BaseAttention):
    def call(self, x):
        mha_output = self.mha(query=x, key=x, value=x, use_causal_mask=True)

        x = self.add([x, mha_output])
        x = self.ln(x)

        return x
