import keras
import numpy as np


def get_positional_encoding(max_seq_len, d_model):
    # Initialize the positional encoding matrix with zeros
    positional_encoding = np.zeros((max_seq_len, d_model))

    # Calculate the positional encoding values using the given formulas
    for pos in range(max_seq_len):
        for i in range(0, d_model, 2):
            positional_encoding[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
            positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((i + 1) / d_model)))

    return positional_encoding

def possitionalEmbedding(input_dim, output_dim):  # TODO

    return keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim)

def model_func(encoder_vocab_len, decoder_vocab_len, encoder_maxlen, decoder_maxlen, params):
    num_heads, key_dim, d_v, d_ff, d_model, n = params

    possitional_encoding_encoder = get_positional_encoding(encoder_maxlen, d_model)
    possitional_encoding_decoder = get_positional_encoding(decoder_maxlen, d_model)

    encoder_input = keras.Input(shape=(encoder_maxlen,))  # fixed len input to apply possitional encoding
    decoder_input = keras.Input(shape=(decoder_maxlen,))

    # encoder part
    embedded = keras.layers.Embedding(input_dim=encoder_vocab_len, output_dim=d_model)(encoder_input)
    embedded_possition = embedded + possitional_encoding_encoder  # possitional encoding should match the emmbeding shape
    embedded_possition = keras.layers.Dropout(0.1)(embedded_possition)

    encoded = embedded_possition
    for i in range(n):
        attended_encoded = keras.layers.MultiHeadAttention(num_heads,
                                        key_dim,
                                        dropout=0.1,
                                        use_bias=True,
                                        output_shape=(d_model,))(encoded, encoded, encoded)  # todo padding_mask
        attended_encoded_d = keras.layers.Dropout(0.1)(attended_encoded)

        add = encoded + attended_encoded_d
        normalised = keras.layers.LayerNormalization()(add)

        fed_f = keras.layers.Dense(d_ff)(normalised)  # feed forward 1 part
        fed_ff = keras.layers.Dense(d_model)(keras.activations.relu(fed_f))  # feed forward 2 part
        fed_ff_d = keras.layers.Dropout(0.1)(fed_ff)

        add2 = normalised + fed_ff_d
        normalised2 = keras.layers.LayerNormalization()(add2)

        encoded = normalised2  # and the loop is repeated

    encoder_output = encoded  # output from encoder

    # decoder part
    de_embed = keras.layers.Embedding(input_dim=decoder_vocab_len, output_dim=d_model)(decoder_input)
    de_embed_pos = de_embed + possitional_encoding_decoder
    de_embed_pos = keras.layers.Dropout(0.1)(de_embed_pos)

    decoded = de_embed_pos
    for i in range(n):
        self_attention = (keras.layers.MultiHeadAttention(num_heads,
                                        key_dim,
                                        dropout=0.1,
                                        use_bias=True,
                                        output_shape=(d_model,))
                            (decoded, decoded, decoded))
        self_attention_d = keras.layers.Dropout(0.1)(self_attention)

        add = decoded + self_attention_d
        normalised1 = keras.layers.LayerNormalization()(add)

        cross_attention = (keras.layers.MultiHeadAttention(num_heads,
                                        key_dim,
                                        dropout=0.1,
                                        use_bias=True,
                                        output_shape=(d_model,))
                           (normalised1, encoder_output,encoder_output))
        cross_attention_d = keras.layers.Dropout(0.1)(cross_attention)

        add2 = normalised1 + cross_attention_d
        normalised2 = keras.layers.LayerNormalization()(add2)

        fed_f = keras.layers.Dense(d_ff)(normalised2)  # feed forward 1 part
        fed_ff = keras.layers.Dense(d_model)(keras.activations.relu(fed_f))  # feed forward 2 part
        fed_ff_d = keras.layers.Dropout(0.1)(fed_ff)

        add3 = normalised2 + fed_ff_d
        normalised3 = keras.layers.LayerNormalization()(add3)

        decoded = normalised3

    decoder_dense_output = keras.layers.Dense(decoder_vocab_len, activation='softmax', name='decoder_output')(decoded)

    return keras.Model(inputs=[encoder_input, decoder_input], outputs=decoder_dense_output)


if __name__ == '__main__':
    params = (8, 64, 64, 256, 512, 6)
    model = model_func(10000, 10000, 100, 100, params)
    model.summary()

    # Generate random input data with appropriate shapes
    encoder_input_data = np.random.randint(0, 10000, (1, 100))  # (batch_size, sequence_length)
    decoder_input_data = np.random.randint(0, 10000, (1, 100))  # (batch_size, sequence_length)

    # Call the model with the random input data
    output = model.call([encoder_input_data, decoder_input_data], training=False)

    # Print the shape of the output
    print(f'Output shape: {output.shape}')