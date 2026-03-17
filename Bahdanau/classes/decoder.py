import tensorflow as tf


class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        # query: Decoder's hidden state, shape == (batch_size, hidden size)
        # values: Encoder output, shape == (batch_size, max_length, hidden_size)

        # Expand query so it can be added to values
        query_with_time_axis = tf.expand_dims(query, 1)

        # Calculate the attention score
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))

        # Apply softmax to get the attention weights
        attention_weights = tf.nn.softmax(score, axis=1)

        # Compute the context vector as the weighted sum of the encoder outputs
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, timesteps_num, embedding_dim, decoder_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.decoder_dim = decoder_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=timesteps_num)
        self.rnn = tf.keras.layers.GRU(decoder_dim, return_sequences=True, return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.attention = BahdanauAttention(decoder_dim)

        # Projection layer to handle the doubled dimensions from bidirectional encoder
        self.state_projection = tf.keras.layers.Dense(self.decoder_dim)

    def call(self, x, state, enc_output):
        # Embed the input token
        x = self.embedding(x)

        # Project the state if it has doubled dimension
        if state.shape[-1] == 2 * self.decoder_dim:
            state = self.state_projection(state)

        # Pass the embedded input through the RNN
        rnn_output, state = self.rnn(x, initial_state=state)

        # Apply attention
        context_vector, attention_weights = self.attention(state, enc_output)

        # Expand context_vector to have the same time dimension as rnn_output
        context_vector = tf.expand_dims(context_vector, 1)

        # Repeat context_vector along the time axis
        context_vector = tf.repeat(context_vector, repeats=rnn_output.shape[1], axis=1)

        # Concatenate context vector and RNN output
        context_and_output = tf.concat([context_vector, rnn_output], axis=-1)

        # Generate logits
        logits = self.dense(context_and_output)

        return logits, state

