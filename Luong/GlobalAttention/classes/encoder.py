import tensorflow as tf


class Encoder(tf.keras.Model):

    def __init__(self, vocab_size, timesteps_num, embedding_dim, encoder_dim, **kwards):
        super(Encoder, self).__init__(**kwards)
        self.encoder_dim = encoder_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=timesteps_num)
        self.rnn = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(encoder_dim, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        )

    def call(self, x, state):
        x = self.embedding(x)
        output, forward_state, backward_state = self.rnn(x, initial_state=state)
        state = tf.concat([forward_state, backward_state], axis=-1)
        return output, state

    def init_state(self, batch_size):
        return [tf.zeros((batch_size, self.encoder_dim)) for _ in range(2)]  # One state for each direction