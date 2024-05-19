import keras
import numpy as np
import tensorflow as tf
import keras_nlp

# todo missing future masking
# TODO masking is not propagating - implement it elseway?
# !!! TODO KerasNLP layers sinecosine positional embeddings !!!
# todo - I would want the potional encoding to be a simple fixed layer that just does addition - find a way
# todo - end dimensions are not matching at the possitional encoding layer : [1,2,128] vs. [22,128]

class CustomSinePositionEncoding(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomSinePositionEncoding, self).__init__(**kwargs)
        self.sine_position_encoding = keras_nlp.layers.SinePositionEncoding()

    def call(self, inputs, mask=None):
        positional = self.sine_position_encoding(inputs)
        return inputs + positional

    def compute_mask(self, inputs, mask=None):
        return mask

class LookAheadMaskLayer(keras.layers.Layer):
    def call(self, inputs):
        batch_size, seq_len, _ = tf.shape(inputs)
        look_ahead_mask = tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)
        # look_ahead_mask = tf.reshape(look_ahead_mask, (1, 1, seq_len, seq_len))
        # return tf.tile(look_ahead_mask, [batch_size, 1, 1, 1])
        return look_ahead_mask
    # def compute_mask(self, inputs, mask=None):
    #     return mask

class MyMaskingLayer(keras.layers.Layer):
    def call(self, x):
        # mask = tf.cast(tf.math.not_equal(x, 0), tf.float32)
        mask = x
        mask = mask[:, tf.newaxis, :]  # this works!!
        return mask

def model_func(encoder_vocab_len, decoder_vocab_len, encoder_maxlen, decoder_maxlen, params):
    num_heads, key_dim, value_dim, d_ff, d_model, n = params

    encoder_input = keras.Input(shape=(encoder_maxlen,))  # fixed len input to apply positional encoding
    decoder_input = keras.Input(shape=(decoder_maxlen,))

    # Encoder part
    embedded = keras.layers.Embedding(input_dim=encoder_vocab_len, output_dim=d_model, mask_zero=True)(encoder_input)
    encoder_mask = embedded._keras_mask
    print(encoder_mask)
    encoder_mask = MyMaskingLayer()(embedded._keras_mask)
    print(encoder_mask)

    embedded_position = CustomSinePositionEncoding()(embedded)
    embedded_position = keras.layers.Dropout(0.1)(embedded_position)

    encoded = embedded_position
    for i in range(n):
        attended_encoded = keras.layers.MultiHeadAttention(
            num_heads,
            key_dim,
            value_dim=value_dim,
            dropout=0.1,
            use_bias=True)(encoded, encoded, encoded,
                           attention_mask=encoder_mask
                           )
        attended_encoded_d = keras.layers.Dropout(0.1)(attended_encoded)

        add = encoded + attended_encoded_d
        normalized = keras.layers.LayerNormalization()(add)

        fed_f = keras.layers.Dense(d_ff)(normalized)  # feed forward 1 part
        fed_ff = keras.layers.Dense(d_model)(keras.activations.relu(fed_f))  # feed forward 2 part
        fed_ff_d = keras.layers.Dropout(0.1)(fed_ff)

        add2 = normalized + fed_ff_d
        normalized2 = keras.layers.LayerNormalization()(add2)

        encoded = normalized2  # and the loop is repeated

    encoder_output = encoded  # output from encoder

    # Decoder part
    de_embed = keras.layers.Embedding(input_dim=decoder_vocab_len, output_dim=d_model, mask_zero=True)(decoder_input)

    de_embed_pos = CustomSinePositionEncoding()(de_embed)
    de_embed_pos = keras.layers.Dropout(0.1)(de_embed_pos)

    decoder_mask = MyMaskingLayer()(de_embed._keras_mask)

    # combined_mask = CombinedMask()(de_mask)
    # decoder_mask = tf.linalg.band_part(tf.ones((decoder_maxlen, decoder_maxlen)), -1, 0)

    decoded = de_embed_pos
    for i in range(n):
        self_attention = keras.layers.MultiHeadAttention(
            num_heads,
            key_dim,
            value_dim=value_dim,
            dropout=0.1,
            use_bias=True)(decoded, decoded, decoded
                           , attention_mask=decoder_mask
                           )
        self_attention_d = keras.layers.Dropout(0.1)(self_attention)

        add = decoded + self_attention_d
        normalized1 = keras.layers.LayerNormalization()(add)

        cross_attention = keras.layers.MultiHeadAttention(
            num_heads,
            key_dim,
            value_dim=value_dim,
            dropout=0.1,
            use_bias=True)(normalized1, encoder_output, encoder_output
                           , attention_mask=encoder_mask
                           )
        cross_attention_d = keras.layers.Dropout(0.1)(cross_attention)

        add2 = normalized1 + cross_attention_d
        normalized2 = keras.layers.LayerNormalization()(add2)

        fed_f = keras.layers.Dense(d_ff)(normalized2)  # feed forward 1 part
        fed_ff = keras.layers.Dense(d_model)(keras.activations.relu(fed_f))  # feed forward 2 part
        fed_ff_d = keras.layers.Dropout(0.1)(fed_ff)

        add3 = normalized2 + fed_ff_d
        normalized3 = keras.layers.LayerNormalization()(add3)

        decoded = normalized3

    decoder_dense_output = keras.layers.Dense(decoder_vocab_len, activation='softmax', name='decoder_output')(decoded)

    return keras.Model(inputs=[encoder_input, decoder_input], outputs=decoder_dense_output)


if __name__ == '__main__':
    params = (8, 64, 64, 256, 512, 6)
    model = model_func(10000, 10000, 100, 100, params)
    model.summary()

    # Generate random input data with appropriate shapes
    encoder_input_data = np.random.randint(0, 10000, (1, 190))  # (batch_size, sequence_length)
    decoder_input_data = np.random.randint(0, 10000, (1, 170))  # (batch_size, sequence_length)

    # Call the model with the random input data
    output = model.call([encoder_input_data, decoder_input_data], training=False)

    # Print the shape of the output
    print(f'Output shape: {output.shape}')