import keras

def possitionalEmbedding(input_dim, output_dim):  # TODO
    return keras.layers.Embedding(input_dim=input_dim, output_dim=output_dim)

def model_func(encoder_vocab_len, decoder_vocab_len, encoder_maxlen, decoder_maxlen, params):
    num_heads, key_dim, d_v, d_ff, d_model, n = params

    encoder_input = keras.Input(shape=(None,))
    decoder_input = keras.Input(shape=(None,))

    # encoder part
    embedded = possitionalEmbedding(encoder_vocab_len, d_model)(encoder_input)
    embedded = keras.layers.Dropout(0.1)(embedded)

    encoded = embedded
    for i in range(n):
        attended_encoded = keras.layers.MultiHeadAttention(num_heads,
                                        key_dim,
                                        dropout=0.0,
                                        use_bias=True,
                                        output_shape=None)(encoded, encoded, encoded)  # todo padding_mask
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
    de_embed = possitionalEmbedding(decoder_vocab_len, d_model)(decoder_input)
    de_embed = keras.layers.Dropout(0.1)(de_embed)

    for i in range(n):
        self_attention = (keras.layers.MultiHeadAttention(num_heads, key_dim, dropout=0.0)
                            (de_embed, de_embed, de_embed))
        self_attention_d = keras.layers.Dropout(0.1)(self_attention)
        add = de_embed + self_attention_d
        normalised1 = keras.layers.LayerNormalization()(add)
        cross_attention = (keras.layers.MultiHeadAttention(num_heads, key_dim, dropout=0.0)
                           (normalised1, encoder_output,encoder_output))
        cross_attention_d = keras.layers.Dropout(0.1)(cross_attention)

        add2 = normalised1 + cross_attention_d
        normalised2 = keras.layers.LayerNormalization()(add2)

        fed_f = keras.layers.Dense(d_ff)(normalised2)  # feed forward 1 part
        fed_ff = keras.layers.Dense(d_model)(keras.activations.relu(fed_f))  # feed forward 2 part
        fed_ff_d = keras.layers.Dropout(0.1)(fed_ff)

        add3 = normalised2 + fed_ff_d
        normalised3 = keras.layers.LayerNormalization()(add3)

        de_embed = normalised3

    decoder_dense_output = keras.layers.Dense(decoder_vocab_len)(keras.activations.softmax(de_embed))  # todo not softmax??

    return keras.Model(inputs=[encoder_input, decoder_input], outputs=decoder_dense_output)


if __name__ == '__main__':
    params = (4, 23, 23, 44, 128, 1)
    model = model_func(20, 20, 3, 7, params)
    model.summary()