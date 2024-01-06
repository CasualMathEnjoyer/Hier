from keras.layers import LSTM, Input, Dense, TimeDistributed, Bidirectional, Flatten, RepeatVector, Permute, Multiply, Lambda
from keras.models import Model, Sequential, load_model

import keras.backend as K

from keras.layers import Masking, Embedding

# 0.3734

def model_func(in_vocab_size, out_vocab_size, in_seq_len, out_seq_len):
    embed_dim = 32
    latent_dim = 32

    # not bidirectional yet
    encoder_inputs = Input(shape=(None, ), dtype="int64")
    masked_encoder = Masking(mask_value=0)(encoder_inputs)
    embed_masked_encoder = Embedding(in_vocab_size, embed_dim, input_length=in_seq_len)(masked_encoder)

    encoder = LSTM(latent_dim, return_state=True, return_sequences=False, activation='sigmoid')
    encoder_outputs, state_h, state_c = encoder(embed_masked_encoder)
    encoder_states = [state_h, state_c]

    decoder_inputs = Input(shape=(None, ), dtype="int64")  # sent_len tam mozna byt nemusi?
    masked_decoder = Masking(mask_value=0)(decoder_inputs)
    embed_masked_decoder = Embedding(out_vocab_size, embed_dim, input_length=out_seq_len)(masked_decoder)
    decoder = LSTM(latent_dim, return_state=True, return_sequences=True, activation='sigmoid')
    decoder_outputs, _, _ = decoder(embed_masked_decoder, initial_state=encoder_states)

    # attention = Attention()([decoder_outputs, encoder_outputs])
    #context_vector = Concatenate(axis=-1)([decoder_outputs, attention])

    decoder_dense = Dense(out_vocab_size, activation="softmax")
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    return model

def load_and_split_model(model_file_path):
    # Load the entire model
    full_model = load_model(model_file_path)

    # Extract the encoder layers from the full model
    encoder_inputs = full_model.input[0]
    encoder_outputs, state_h, state_c = full_model.layers[4].output  # Assuming the LSTM layer is at index 4
    encoder_states = [state_h, state_c]
    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

    # Extract the decoder layers from the full model
    decoder_inputs = full_model.input[1]
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    decoder_embedding_layer = full_model.layers[5]  # Assuming the Embedding layer is at index 5
    decoder_lstm = full_model.layers[6]  # Assuming the LSTM layer is at index 6
    decoder_dense = full_model.layers[7]  # Assuming the Dense layer is at index 7

    embed_masked_decoder = decoder_embedding_layer(decoder_inputs)
    decoder_outputs, state_h, state_c = decoder_lstm(embed_masked_decoder, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)

    return encoder_model, decoder_model

# Example usage:
model_file_path = 'your_model.h5'
encoder_model, decoder_model = load_and_split_model(model_file_path)
