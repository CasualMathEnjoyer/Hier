from keras.layers import LSTM, Input, Dense, TimeDistributed, Bidirectional, Flatten, RepeatVector, Permute, Multiply, Lambda
from keras.models import Model, Sequential

import keras.backend as K

from keras.layers import Layer, Activation, dot, concatenate, Attention
import tensorflow as tf

def model_func(sent_len, embed_dim, num_neurons):
    # not bidirectional yet
    encoder_inputs = Input(shape=(None, embed_dim))
    encoder = LSTM(num_neurons, return_state=True, return_sequences=False, activation='tanh')
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, embed_dim))  # sent_len tam mozna byt nemusi?
    decoder = LSTM(num_neurons, return_state=True, return_sequences=True, activation='tanh')
    decoder_outputs, _, _ = decoder(decoder_inputs, initial_state=encoder_states)

    decoder_dense = Dense(embed_dim, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    # at_input = [decoder_hidden_states, network_outputs]
    # attention = Attention()(at_input)
    # output_layer = network_timestep(attention)

    #network_outputs = network_timestep(network_outputs)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    return model