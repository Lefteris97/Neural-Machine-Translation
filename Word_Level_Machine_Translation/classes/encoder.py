# This Pythob class provides fundamental computational funtionality for the 
# implementation of the encoder. In particular, the encoder will be composed of
# an embedding layer followed by a GRU layer. The input to the encoder is a 
# sequence of integers, which is converted to a sequence of embedding vectors
# of size embedding_dim. This sequence of vectors is subsequently sent to an
# RNN, which converts the input at each of the timesteps_num timesteps to a 
# vector of size encode_dim. Mind that only the output at the last time step
# is actually returned since the return_sequences parameter is set to False.


import tensorflow as tf

class Encoder(tf.keras.Model):

    def __init__(self,vocab_size,timesteps_num,embedding_dim,encoder_dim,**kwards):
        super(Encoder,self).__init__(**kwards)
        self.encoder_dim = encoder_dim
        self.embedding = tf.keras.layers.Embedding(vocab_size,embedding_dim,
                                                   input_length=timesteps_num)
        self.rnn = tf.keras.layers.GRU(encoder_dim,return_sequences=False,
                                       return_state=True)

    def call(self,x,state):
        x = self.embedding(x)
        x, state = self.rnn(x,initial_state=state)
        return x, state

    def init_state(self,batch_size):
        return tf.zeros((batch_size,self.encoder_dim))