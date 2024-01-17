from keras.layers import LSTM, Input, Dense, TimeDistributed, Bidirectional, Flatten, RepeatVector, Permute, Multiply, Lambda
from keras.models import Model, Sequential, load_model
import keras.backend as K

from keras.layers import Masking, Embedding

# 0.3734

def Encoder(input_vocab_size, input_seq_len, embed_dim, latent_dim):
    encoder_inputs = Input(shape=(None,), dtype="int64", name="encoder_input")
    masked_encoder = Masking(mask_value=0, name="encoder_mask")(encoder_inputs)
    embed_masked_encoder = Embedding(input_vocab_size, embed_dim, input_length=input_seq_len, name="encoder_embed")(
        masked_encoder)
    encoder_lstm = LSTM(latent_dim, return_state=True, return_sequences=False, activation='sigmoid', name="encoder_LSTM")
    encoder_outputs, state_h, state_c = encoder_lstm(embed_masked_encoder)
    encoder_states = [state_h, state_c]
    return encoder_inputs, encoder_states

def Decoder(output_vocab_size, output_seq_len, embed_dim, latent_dim, initial_state):
    decoder_inputs = Input(shape=(None,), dtype="int64", name="decoder_input")  # sent_len tam mozna byt nemusi?
    masked_decoder = Masking(mask_value=0, name="decoder_mask")(decoder_inputs)
    embed_masked_decoder = Embedding(output_vocab_size, embed_dim, input_length=output_seq_len, name="decoder_embed")(
        masked_decoder)
    decoder = LSTM(latent_dim, return_state=True, return_sequences=True, activation='sigmoid', name="decoder_LSTM")
    decoder_outputs, state_h, state_c = decoder(embed_masked_decoder, initial_state=initial_state)
    decoder_states = [state_h, state_c]
    # attention = Attention()([decoder_outputs, encoder_outputs])
    # context_vector = Concatenate(axis=-1)([decoder_outputs, attention])
    decoder_dense = Dense(output_vocab_size, activation="softmax", name="decoder_dense")
    decoder_outputs = decoder_dense(decoder_outputs)
    return decoder_inputs, decoder_outputs, decoder_states

def model_func(in_vocab_size, out_vocab_size, in_seq_len, out_seq_len):
    embed_dim = 32
    latent_dim = 32

    # not bidirectional yet
    encoder_inputs, encoder_states = Encoder(in_vocab_size, in_seq_len, embed_dim, latent_dim)
    decoder_inputs, decoder_outputs, _ = Decoder(out_vocab_size, out_seq_len, embed_dim, latent_dim, encoder_states)

    model = Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    return model

def load_model_mine(model_name):
    from model_file import PositionalEmbedding, TransformerEncoder, TransformerDecoder
    return load_model(model_name, custom_objects={'PositionalEmbedding': PositionalEmbedding,
                                                               'TransformerEncoder': TransformerEncoder,
                                                               'TransformerDecoder': TransformerDecoder
    })
def load_and_split_model(model_folder_path, in_vocab_size, out_vocab_size, in_seq_len, out_seq_len):
    latent_dim = 32
    embed_dim = 32

    # # CREATE THE MODEL
    # encoder_inputs, encoder_states = Encoder(in_vocab_size, in_seq_len, embed_dim, latent_dim)
    # encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)
    # encoder_model.summary()
    #
    # decoder_state_input_h = Input(shape=(latent_dim,))
    # decoder_state_input_c = Input(shape=(latent_dim,))
    # decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    # decoder_inputs, decoder_outputs, decoder_states = Decoder(out_vocab_size, out_seq_len, embed_dim, latent_dim,
    #                                                           decoder_states_inputs)  # protoze ted hned neznam encoder states
    # decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)
    # decoder_model.summary()
    # # TODO - its a new model - I need to set the weights

    # Load the entire model
    full_model = load_model_mine(model_folder_path)
    # print(len(full_model.layers))

    # Extract the encoder layers from the full model
    # encoder_inputs = full_model.input[0]
    # encoder_mask = full_model.layers[2]
    # encoder_embedding_layer = full_model.layers[4]
    # encoder_LSTM = full_model.layers[6]

    encoder_inputs = Input(shape=(None, ), dtype="int64", name="encoder_input_sentence")
    encoder_mask = full_model.get_layer("encoder_mask")
    encoder_embedding_layer = full_model.get_layer("encoder_embed")
    encoder_lstm = full_model.get_layer("encoder_LSTM")

    encoder_mask = encoder_mask(encoder_inputs)
    encoder_embedding_layer = encoder_embedding_layer(encoder_mask)
    encoder_outputs, state_h, state_c = encoder_lstm(encoder_embedding_layer)
    encoder_states = [state_h, state_c]
    encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states, name="Encoder")

    # Extract the decoder layers from the full model
    decoder_inputs = Input(shape=(None, ), dtype="int64", name="decoder_input_letter")
    decoder_state_input_h = Input(shape=(latent_dim,), name="decoder_input_h")
    decoder_state_input_c = Input(shape=(latent_dim,), name="decoder_input_c")
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]

    # decoder_mask = full_model.layers[3]
    # decoder_embedding_layer = full_model.layers[5]
    # decoder_lstm = full_model.layers[7]
    # decoder_dense = full_model.layers[8]

    decoder_mask = full_model.get_layer("decoder_mask")
    decoder_embedding_layer = full_model.get_layer("decoder_embed")
    decoder_lstm = full_model.get_layer("decoder_LSTM")
    decoder_dense = full_model.get_layer("decoder_dense")


    masked_input = decoder_mask(decoder_inputs)
    embed_masked_decoder = decoder_embedding_layer(masked_input)
    decoder_outputs, state_h, state_c = decoder_lstm(embed_masked_decoder, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states,
                          name="Decoder")

    return encoder_model, decoder_model

if __name__ == "__main__":
    model_folder_path = 'transform2seq_fr-eng_4LSTM'
    encoder_model, decoder_model = load_and_split_model(model_folder_path, 0, 0, 0, 0)
    encoder_model.summary()
    decoder_model.summary()