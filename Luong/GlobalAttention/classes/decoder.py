import tensorflow as tf


class GlobalAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(GlobalAttention, self).__init__()
        # Create a dense layer to project decoder_hidden to match encoder_dim
        self.Wa = tf.keras.layers.Dense(units)

    def call(self, query, values):
        # query: Decoder's hidden state, shape == (batch_size, hidden size)
        # values: Encoder output, shape == (batch_size, max_length, hidden_size)

        # Project the query (decoder state) to the encoder output dimensions
        query_with_time_axis = tf.expand_dims(query, 1)  # Shape: (batch_size, 1, hidden size)
        query_projected = self.Wa(query_with_time_axis)  # Shape: (batch_size, 1, encoder_dim)

        # Compute alignment scores using dot product
        scores = tf.matmul(values, query_projected, transpose_b=True)  # Shape: (batch_size, max_length, 1)
        alignment_scores = tf.squeeze(scores, axis=-1)  # Shape: (batch_size, max_length)

        # Compute attention weights using softmax
        attention_weights = tf.nn.softmax(alignment_scores, axis=1)  # Shape: (batch_size, max_length)

        # Calculate context vector as the weighted sum of encoder outputs
        context_vector = tf.reduce_sum(tf.expand_dims(attention_weights, -1) * values, axis=1)

        return context_vector, attention_weights



### NO Input Feeding
# class Decoder(tf.keras.Model):
#     def __init__(self, vocab_size, timesteps_num, embedding_dim, decoder_dim, **kwargs):
#         super(Decoder, self).__init__(**kwargs)
#         self.decoder_dim = decoder_dim
#         self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=timesteps_num)
#         self.rnn = tf.keras.layers.GRU(decoder_dim, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
#         self.dense = tf.keras.layers.Dense(vocab_size)
#         self.attention = GlobalAttention(units=2*decoder_dim)  # 2048 to match the encoder output
#
#         # Define the projection layer to handle the bidirectional states
#         self.state_projection = tf.keras.layers.Dense(decoder_dim)
#
#     def call(self, x, state, enc_output):
#         # Embed the input token
#         x = self.embedding(x)
#
#         # Project the state if it has doubled dimension
#         if state.shape[-1] == 2 * self.decoder_dim:
#             state = self.state_projection(state)
#
#         # Pass the embedded input through the RNN
#         rnn_output, state = self.rnn(x, initial_state=state)
#
#         # Apply attention
#         context_vector, attention_weights = self.attention(state, enc_output)
#
#         # Expand context_vector to have the same time dimension as rnn_output
#         context_vector = tf.expand_dims(context_vector, 1)
#
#         # Repeat context_vector along the time axis
#         context_vector = tf.repeat(context_vector, repeats=rnn_output.shape[1], axis=1)
#
#         # Concatenate context vector and RNN output
#         context_and_output = tf.concat([context_vector, rnn_output], axis=-1)
#
#         # Generate logits
#         logits = self.dense(context_and_output)
#
#         return logits, state


### WITH Input Feeding
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, timesteps_num, embedding_dim, decoder_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.decoder_dim = decoder_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=timesteps_num)
        self.rnn = tf.keras.layers.GRU(decoder_dim, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.attention = GlobalAttention(units=2*decoder_dim)  # 2048 to match the encoder output

        # Define the projection layer to handle the bidirectional states
        self.state_projection = tf.keras.layers.Dense(decoder_dim)
        # Define the projection layer to handle the concatenated x dimensionality
        self.concat_x_projection = tf.keras.layers.Dense(embedding_dim)

    def call(self, x, state, enc_output, prev_context_vector=None):
        # Embed the input token
        x = self.embedding(x)

        # Project the state if it has doubled dimension
        if state.shape[-1] == 2 * self.decoder_dim:
            state = self.state_projection(state)

        # Input Feeding - Concatenate previous context vector with the input
        if prev_context_vector is not None:
            context_vector = tf.squeeze(prev_context_vector, axis=1)  # Shape: [batch_size, context_dim]
            context_vector = tf.expand_dims(context_vector, 1)  # Shape: [batch_size, 1, context_dim]
            context_vector = tf.repeat(context_vector, repeats=tf.shape(x)[1], axis=1)  # Shape: [batch_size, timesteps_num, context_dim]
            # Concatenate along the feature axis
            x = tf.concat([x, context_vector], axis=-1)  # Shape: [batch_size, timesteps_num, embedding_dim + context_dim]
            x = self.concat_x_projection(x)

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

        return logits, state, context_vector
