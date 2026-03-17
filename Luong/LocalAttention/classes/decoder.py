import tensorflow as tf


# Local-m
class MonotonicAlignment(tf.keras.layers.Layer):
    def __init__(self, units):
        super(MonotonicAlignment, self).__init__()
        self.units = units
        self.window_size = 6
        self.Wa = tf.keras.layers.Dense(units)
        self.Ua = tf.keras.layers.Dense(units)
        self.Va = tf.keras.layers.Dense(1)

    def call(self, query, keys, values, t):
        # Convert timestep 't' to a TensorFlow tensor
        pt = tf.cast(t, tf.float32)

        # Determine the local window around p_t
        alignment_position = tf.cast(pt, tf.int32)
        seq_len = tf.shape(keys)[1]

        # Calculate the start and the end
        start = tf.maximum(0, alignment_position - self.window_size // 2)
        end = tf.minimum(seq_len, alignment_position + self.window_size // 2 + 1)

        # Create masks to handle the window slicing
        range_seq = tf.range(seq_len)
        mask = tf.logical_and(tf.greater_equal(range_seq, start), tf.less(range_seq, end))
        mask = tf.cast(mask, dtype=keys.dtype)

        # Apply mask to keys and values to only keep the relevant window
        keys_window = keys * tf.expand_dims(mask, 0)[:, :, tf.newaxis]
        values_window = values * tf.expand_dims(mask, 0)[:, :, tf.newaxis]

        # Calculate the attention scores
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.Va(tf.nn.tanh(self.Wa(query_with_time_axis) + self.Ua(keys_window)))
        attention_weights = tf.nn.softmax(score, axis=1)

        # Compute the context vector
        context_vector = attention_weights * values_window
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


# Local-p
class PredictiveAlignment(tf.keras.layers.Layer):
    def __init__(self, units):
        super(PredictiveAlignment, self).__init__()
        self.units = units
        self.window_size = 6
        self.sigma = 1.0
        self.Wp = tf.keras.layers.Dense(1)  # For calculating the predicted position p_t
        self.Wa = tf.keras.layers.Dense(units)
        self.Ua = tf.keras.layers.Dense(units)
        self.Va = tf.keras.layers.Dense(1)

    def call(self, query, keys, values):
        # Calculate the predicted position p_t
        batch_size = tf.shape(keys)[0]
        seq_len = tf.shape(keys)[1]
        S = tf.cast(seq_len, tf.float32)  # Length of the source sequence
        pt = S * tf.nn.sigmoid(self.Wp(query))  # Predicted position
        pt = tf.squeeze(pt, axis=-1)

        # Determine the local window around p_t
        alignment_position = tf.cast(pt, tf.int32)

        # Calculate the start and the end
        start = tf.maximum(0, alignment_position - self.window_size // 2)
        end = tf.minimum(seq_len, alignment_position + self.window_size // 2 + 1)

        # Create masks to handle the window slicing
        range_seq = tf.range(seq_len)
        range_seq = tf.expand_dims(range_seq, 0)
        range_seq = tf.tile(range_seq, [batch_size, 1])

        # Expand start and end
        start = tf.expand_dims(start, 1)
        end = tf.expand_dims(end, 1)

        # Create mask and cast it to the same data type as 'keys'
        mask = tf.logical_and(tf.greater_equal(range_seq, start), tf.less(range_seq, end))
        mask = tf.cast(mask, dtype=keys.dtype)

        # Apply mask to keys and values to only keep the relevant window
        keys_window = keys * tf.expand_dims(mask, -1)
        values_window = values * tf.expand_dims(mask, -1)

        # Calculate the attention scores
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.Va(tf.nn.tanh(self.Wa(query_with_time_axis) + self.Ua(keys_window)))

        # Apply Gaussian penalty based on distance from pt
        position_indices = tf.cast(tf.range(tf.shape(keys_window)[1]), tf.float32)
        gaussian_dist = tf.exp(-tf.square(position_indices - pt[:, tf.newaxis]) / (2 * self.sigma ** 2))
        score = score * tf.expand_dims(gaussian_dist, -1)

        # Compute the attention weights and context vector
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values_window
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


### WITH Input Feeding
class Decoder(tf.keras.Model):
    def __init__(self, vocab_size, timesteps_num, embedding_dim, decoder_dim, **kwargs):
        super(Decoder, self).__init__(**kwargs)
        self.decoder_dim = decoder_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=timesteps_num)
        self.rnn = tf.keras.layers.GRU(decoder_dim, return_sequences=True, return_state=True, recurrent_initializer='glorot_uniform')
        self.dense = tf.keras.layers.Dense(vocab_size)
        # Local Attention Types
        self.attention = MonotonicAlignment(units=2*decoder_dim)
        # self.attention = PredictiveAlignment(units=2*decoder_dim)

        # Define the projection layer to handle the bidirectional states
        self.state_projection = tf.keras.layers.Dense(decoder_dim)
        # Define the projection layer to handle the concatenated x dimensionality
        self.concat_x_projection = tf.keras.layers.Dense(embedding_dim)

    def call(self, x, state, enc_output, t=None, prev_context_vector=None):  # For Monotonic Alignment
    # def call(self, x, state, enc_output, prev_context_vector=None):  # For Predictive Alignment
        # Embed the input token
        x = self.embedding(x)

        # Project the state if it has doubled dimension
        if state.shape[-1] == 2 * self.decoder_dim:
            state = self.state_projection(state)

        # Input Feeding - Concatenate previous context vector with the input
        if prev_context_vector is not None:
            context_vector = tf.squeeze(prev_context_vector, axis=1)
            context_vector = tf.expand_dims(context_vector, 1)
            context_vector = tf.repeat(context_vector, repeats=tf.shape(x)[1], axis=1)
            # Concatenate along the feature axis
            x = tf.concat([x, context_vector], axis=-1)
            x = self.concat_x_projection(x)

        # Pass the embedded input through the RNN
        rnn_output, state = self.rnn(x, initial_state=state)

        # Apply attention
        context_vector, attention_weights = self.attention(state, enc_output, enc_output, t=t)  # For Monotonic Alignment
        # context_vector, attention_weights = self.attention(state, enc_output, enc_output)  # For Predictive Alignment

        # Expand context_vector to have the same time dimension as rnn_output
        context_vector = tf.expand_dims(context_vector, 1)

        # Repeat context_vector along the time axis
        context_vector = tf.repeat(context_vector, repeats=rnn_output.shape[1], axis=1)

        # Concatenate context vector and RNN output
        context_and_output = tf.concat([context_vector, rnn_output], axis=-1)

        # Generate logits
        logits = self.dense(context_and_output)

        return logits, state, context_vector
