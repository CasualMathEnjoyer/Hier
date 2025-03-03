import keras
import numpy as np
import tensorflow as tf
import keras_nlp

class CustomSinePositionEncoding(keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CustomSinePositionEncoding, self).__init__(**kwargs)
        self.sine_position_encoding = keras_nlp.layers.SinePositionEncoding()

    def call(self, inputs, mask=None):
        positional = self.sine_position_encoding(inputs)
        return inputs + positional

    def compute_mask(self, inputs, mask=None):
        return mask

class MyMaskingLayer(keras.layers.Layer):
    def call(self, x):
        mask = tf.cast(tf.not_equal(x, 0), dtype=tf.float32)
        mask = mask[:, tf.newaxis, :]
        # mask = x[:, tf.newaxis, :]  # this works when input is a mask
        return mask

import numpy as np
import keras
from keras.models import Model
from keras.layers import Embedding

def extend_model_embeddings(model, new_encoder_vocab_size, new_decoder_vocab_size):
    """
    Extends the embedding layers of a Transformer model with additional tokens.

    Parameters:
        model (keras.Model): The loaded Transformer model.
        new_encoder_vocab_size (int): The new vocabulary size for the encoder.
        new_decoder_vocab_size (int): The new vocabulary size for the decoder.

    Returns:
        keras.Model: A new model with extended embeddings.
    """
    # Get encoder and decoder embedding layers
    encoder_embedding_layer = model.get_layer(name='embedding')
    decoder_embedding_layer = model.get_layer(name='embedding_1')

    # Extract existing weights
    encoder_weights = encoder_embedding_layer.get_weights()[0]
    decoder_weights = decoder_embedding_layer.get_weights()[0]

    encoder_vocab_size, embedding_dim = encoder_weights.shape
    decoder_vocab_size, _ = decoder_weights.shape

    # Debugging dimensions
    print(f"Original encoder_vocab_size: {encoder_vocab_size}")
    print(f"Original decoder_vocab_size: {decoder_vocab_size}")
    print(f"Embedding dimension: {embedding_dim}")

    # Extend weights
    new_encoder_weights = np.random.normal(size=(new_encoder_vocab_size, embedding_dim))
    new_encoder_weights[:encoder_vocab_size] = encoder_weights

    new_decoder_weights = np.random.normal(size=(new_decoder_vocab_size, embedding_dim))
    new_decoder_weights[:decoder_vocab_size] = decoder_weights

    # Create new embeddings
    new_encoder_embedding = Embedding(
        input_dim=new_encoder_vocab_size,
        output_dim=embedding_dim,
        weights=[new_encoder_weights],
        trainable=True,
        name="new_encoder_embedding"
    )
    new_decoder_embedding = Embedding(
        input_dim=new_decoder_vocab_size,
        output_dim=embedding_dim,
        weights=[new_decoder_weights],
        trainable=True,
        name="new_decoder_embedding"
    )

    # Rebuild the model graph
    encoder_input = model.input[0]
    decoder_input = model.input[1]

    # Replace the encoder embedding
    encoder_embedded = new_encoder_embedding(encoder_input)
    encoder_position_encoded = model.get_layer(name='custom_sine_position_encoding')(encoder_embedded)
    encoder_position_encoded = model.get_layer(name='dropout')(encoder_position_encoded)

    # Reconstruct the encoder layers
    x = encoder_position_encoded
    for layer in model.layers[4:28]:  # Replace these indices based on your architecture
        if isinstance(layer, keras.layers.MultiHeadAttention):
            x = layer(x, x, x)
        else:
            x = layer(x)
    encoder_output = x

    # Replace the decoder embedding
    decoder_embedded = new_decoder_embedding(decoder_input)
    decoder_position_encoded = model.get_layer(name='custom_sine_position_encoding_1')(decoder_embedded)
    decoder_position_encoded = model.get_layer(name='dropout_13')(decoder_position_encoded)

    # Reconstruct the decoder layers
    y = decoder_position_encoded
    for layer in model.layers[34:]:  # Replace these indices based on your architecture
        if isinstance(layer, keras.layers.MultiHeadAttention):
            if "cross_att" in layer.name:  # Cross-attention layer
                y = layer(y, encoder_output, encoder_output)
            else:  # Self-attention layer
                y = layer(y, y, y)
        else:
            y = layer(y)

    # Output layer
    decoder_output = y

    # Create and compile the new model
    new_model = Model(inputs=[encoder_input, decoder_input], outputs=decoder_output)
    new_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return new_model

def adjust_output_layer(model, new_vocab_size):
    """
    Adjusts the output layer of the model to match the target vocabulary size.

    Parameters:
        model (keras.Model): The Transformer model.
        new_vocab_size (int): The new size of the target vocabulary.

    Returns:
        keras.Model: A new model with the adjusted output layer.
    """
    # Get the input and intermediate outputs
    inputs = model.input
    intermediate_output = model.layers[-2].output  # Second-to-last layer output

    # Replace the output layer
    new_output = keras.layers.Dense(new_vocab_size, activation='softmax', name="new_decoder_output")(intermediate_output)

    # Create a new model
    new_model = Model(inputs=inputs, outputs=new_output)

    # Compile the new model
    new_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return new_model

def model_func(encoder_vocab_len, decoder_vocab_len, encoder_maxlen, decoder_maxlen, model_settings_params):
    num_heads = model_settings_params["h"]
    key_dim = model_settings_params["d_k"]
    value_dim = model_settings_params["d_v"]
    d_ff = model_settings_params["d_ff"]
    d_model = model_settings_params["d_model"]
    n = model_settings_params["n"]

    encoder_input = keras.Input(shape=(encoder_maxlen,))  # fixed len input to apply positional encoding
    decoder_input = keras.Input(shape=(decoder_maxlen,))

    # Encoder part
    encoder_mask = MyMaskingLayer()(encoder_input)
    embedded = keras.layers.Embedding(input_dim=encoder_vocab_len, output_dim=d_model, mask_zero=False)(encoder_input)
    # not usig default masking in embedding because it doesn't propagate anyway

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
                           attention_mask=encoder_mask  # [batch, sequences, model_dim(embedding)]
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
    decoder_mask = MyMaskingLayer()(decoder_input)
    de_embed = keras.layers.Embedding(input_dim=decoder_vocab_len, output_dim=d_model, mask_zero=False)(decoder_input)

    de_embed_pos = CustomSinePositionEncoding()(de_embed)
    de_embed_pos = keras.layers.Dropout(0.1)(de_embed_pos)

    decoded = de_embed_pos
    cross_attention_vecs = []
    for i in range(n):
        self_attention = keras.layers.MultiHeadAttention(
            num_heads,
            key_dim,
            value_dim=value_dim,
            dropout=0.1,
            use_bias=True)(decoded, decoded, decoded
                           , attention_mask=decoder_mask
                           , use_causal_mask=True
                           )
        self_attention_d = keras.layers.Dropout(0.1)(self_attention)

        add = decoded + self_attention_d
        normalized1 = keras.layers.LayerNormalization()(add)

        cross_attention, cross_attention_scores = keras.layers.MultiHeadAttention(
            num_heads,
            key_dim,
            value_dim=value_dim,
            dropout=0.1,
            use_bias=True,
            name=f'cross_att{i}')(normalized1, encoder_output, encoder_output
                                   , attention_mask=encoder_mask
                                   , return_attention_scores=True  # to calculate attention matrix
                                   )
        # cross_attention_vecs.append(cross_attention_scores)

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

    return keras.Model(inputs=[encoder_input, decoder_input], outputs=[decoder_dense_output])

if __name__ == '__main__':
    params = (4, 64, 64, 256, 512, 2)
    model = model_func(10, 10, 50, 50, params)
    # model.summary()

    for layer in model.layers:
        print(layer.name)

    # Generate random input data with appropriate shapes
    encoder_input_data = np.random.randint(0, 10, (1, 50))  # (batch_size, sequence_length)
    decoder_input_data = np.random.randint(0, 10, (1, 50))  # (batch_size, sequence_length)

    # visualise_attention(model, encoder_input_data, decoder_input_data, 2, 4)
