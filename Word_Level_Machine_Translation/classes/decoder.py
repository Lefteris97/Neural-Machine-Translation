# This Pythob class provides fundamental computational funtionality for the 
# implementation of the encoder. In particular, the decoder has almost the same
# structure as the encoder, except that there exists an additional dense layer
# that converts the vector of size decoder_dim which is the output from the RNN
# layer, into a vector that represents the probability distribution across the
# the target vocabulary. Mind that the decoder returns outputs along all its 
# timesteps since the corresponding return_sequences parameter is set to True.

import tensorflow as tf

class Decoder(tf.keras.Model):

    def __init__(self,vocab_size,timesteps_num,embedding_dim,decoder_dim,**kwards):
        super(Decoder,self).__init__(**kwards)
        self.decoder_dim = decoder_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim,
                                                   input_length=timesteps_num)
        self.rnn = tf.keras.layers.GRU(decoder_dim,return_sequences=True,
                                       return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self,x,state):
        x = self.embedding(x)
        x, state = self.rnn(x,state)
        x = self.dense(x)
        return x,state
    