import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model, Sequential, load_model
from keras.layers import Input

# https://github.com/evidentlyai/evidently

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(dense_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask=None):
        attention_output = self.attention(query=inputs, value=inputs, key=inputs)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "dense_dim": self.dense_dim,
                "num_heads": self.num_heads,
            }
        )
        return config
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, vocab_size, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(
            input_dim=vocab_size, output_dim=embed_dim
        )
        self.position_embeddings = layers.Embedding(
            input_dim=sequence_length, output_dim=embed_dim
        )
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "sequence_length": self.sequence_length,
                "vocab_size": self.vocab_size,
                "embed_dim": self.embed_dim,
            }
        )
        return config
class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()
        self.add = layers.Add()  # instead of `+` to preserve mask
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None):
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, use_causal_mask=True
        )
        out_1 = self.layernorm_1(self.add([inputs, attention_output_1]))

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
        )
        out_2 = self.layernorm_2(self.add([out_1, attention_output_2]))

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(self.add([out_2, proj_output]))

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "latent_dim": self.latent_dim,
                "num_heads": self.num_heads,
            }
        )
        return config

def Encoder(in_vocab_size, in_seq_len, embed_dim, latent_dim, num_heads):
    encoder_inputs = keras.Input(shape=(None,), dtype="int64", name="encoder_inputs")
    x = PositionalEmbedding(in_seq_len, in_vocab_size, embed_dim, name="enc_embed")(encoder_inputs)
    encoder_outputs = TransformerEncoder(embed_dim, latent_dim, num_heads, name="encoder_trans")(x)
    encoder = keras.Model(encoder_inputs, encoder_outputs, name="Encoder")
    encoder_outputs = encoder(encoder_inputs)
    return encoder_inputs, encoder_outputs

def Decoder(out_vocab_size, out_seq_len, embed_dim, latent_dim, initial_state, num_heads):
    decoder_inputs = keras.Input(shape=(None,), dtype="int64", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
    x = PositionalEmbedding(out_seq_len, out_vocab_size, embed_dim, name="dec_em")(decoder_inputs)
    x = TransformerDecoder(embed_dim, latent_dim, num_heads, name="decoder_transformer")(x, encoded_seq_inputs)
    # x = layers.Dropout(0.5, name="Dropout")(x)
    decoder_outputs = layers.Dense(out_vocab_size, activation="softmax", name="decoder_dense")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs], decoder_outputs, name="Decoder")
    decoder_outputs = decoder([decoder_inputs, initial_state])  # todo is this needed?
    return decoder_inputs, decoder_outputs, x

def model_func(in_vocab_size, out_vocab_size, in_seq_len, out_seq_len):
    embed_dim = 32
    latent_dim = 32
    num_heads = 2
    # in_seq_len = 40
    # out_seq_len = 40  # can they be different ?

    encoder_inputs, encoder_states = Encoder(in_vocab_size, in_seq_len, embed_dim, latent_dim, num_heads)

    decoder_inputs, decoder_outputs, _ = Decoder(out_vocab_size, out_seq_len, embed_dim,
                                                 latent_dim, encoder_states, num_heads)

    transformer = keras.Model(
        [encoder_inputs, decoder_inputs], decoder_outputs, name="transformer"
    )
    transformer.summary()
    return transformer

def load_model_mine(model_name):
    from model_file import PositionalEmbedding, TransformerEncoder, TransformerDecoder
    return keras.models.load_model(model_name, custom_objects={'PositionalEmbedding': PositionalEmbedding,
                                                               'TransformerEncoder': TransformerEncoder,
                                                               'TransformerDecoder': TransformerDecoder
    })

def encoder_state_transform(encoder_output):
    return encoder_output
def load_and_split_model(model_folder_path, in_vocab_size, out_vocab_size, in_seq_len, out_seq_len):
    latent_dim = 32
    embed_dim = 32

    # Load the entire model
    full_model = load_model_mine(model_folder_path)
    # print(len(full_model.layers))

    encoder_inputs = Input(shape=(None, ), dtype="int64", name="encoder_input_sentence")
    # encoder_embedding_layer = full_model.get_layer("enc_embed")
    # encoder_transformer = full_model.get_layer("encoder_trans")
    #
    # encoder_embedding_layer = encoder_embedding_layer(encoder_inputs)
    # encoder_outputs = encoder_transformer(encoder_embedding_layer)
    #
    # encoder_model = Model(inputs=encoder_inputs, outputs=encoder_outputs, name="Encoder")
    #
    # # Extract the decoder layers from the full model
    decoder_inputs = Input(shape=(None, ), dtype="int64", name="decoder_inputs")
    decoder_state_input = Input(shape=(latent_dim,), name="decoder_state_inputs")
    #
    # decoder_embedding_layer = full_model.get_layer("dec_em")
    # decoder_transformer = full_model.get_layer("decoder_transformer")
    # decoder_dense = full_model.get_layer("decoder_dense")
    encoder = full_model.get_layer("Encoder")
    decoder = full_model.get_layer("Decoder")

    # embed_masked_decoder = decoder_embedding_layer(decoder_inputs)
    # decoder_outputs, state_h, state_c = decoder_lstm(embed_masked_decoder, initial_state=decoder_states_inputs)
    # decoder_states = [state_h, state_c]
    # decoder_outputs = decoder_dense(decoder_outputs)
    # decoder_model = Model(inputs=[decoder_inputs] + decoder_state_input, outputs=[decoder_outputs] + decoder_states,
    #                       name="Decoder")

    return encoder, decoder

if __name__ == "__main__":
    model = model_func(2, 3, 5, 7)
    load_and_split_model("transform2seq_fr-eng_trans1", in_vocab_size, out_vocab_size, in_seq_len, out_seq_len)
    model.summary()
