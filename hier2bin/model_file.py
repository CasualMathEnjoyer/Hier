from keras.layers import LSTM, Input, Dense, TimeDistributed, Bidirectional, Flatten, RepeatVector, Permute, Multiply, Lambda
from keras.models import Model, Sequential

import keras.backend as K

from keras.layers import Layer, Activation, dot, concatenate, Attention
import tensorflow as tf

class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W_a = self.add_weight(name='W_a',
                                   shape=tf.TensorShape((input_shape[-1], input_shape[-1])),
                                   initializer='uniform',
                                   trainable=True)
        super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        q = x  # Query
        k = x  # Key
        v = x  # Value

        q_dot_k = dot([q, k], axes=-1)  # Calculate the dot product of Query and Key
        q_dot_k = Activation('softmax')(q_dot_k)  # Apply softmax to get attention scores

        output = dot([q_dot_k, v], axes=(2, 1))  # Weighted sum of Values using attention scores

        self.add_loss(tf.reduce_sum(self.W_a))

        return output


"""attention = Dense(1, activation='tanh')(network_outputs)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(sent_len)(attention)
    attention = Permute([2,1])(attention)

    attention = Lambda(lambda x: K.expand_dims(x, axis=-1))(attention)

    sent_repres = Multiply()([network_outputs, attention])
    sent_repres = Lambda(lambda xin: K.sum(xin, axis=-2), output_shape=(num_neurons+2,))(sent_repres)

    output_layer = TimeDistributed(Dense(1, activation='sigmoid'))(sent_repres)"""

def model_func(sent_len, embed_dim, num_neurons):
    network_inputs = Input(shape=(None, embed_dim))
    network = Bidirectional(LSTM(num_neurons, return_sequences=True, activation='tanh'))
    network_outputs = network(network_inputs)

    network_timestep = TimeDistributed(Dense(1, activation='sigmoid'))
    output_layer = network_timestep(network_outputs)

    model = Model(inputs=network_inputs, outputs=output_layer)
    return model