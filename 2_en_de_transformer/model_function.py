import tensorflow.keras as keras
from model_file_2 import *  # objects for loading
from model_file_mine import *
import os
import pickle

def load_model_mine(model_name):
    print("[MODEL] - loading existing model")
    try:
        custom_objects = {
            'EncoderLayer': EncoderLayer,
            'Encoder': Encoder,
            'DecoderLayer': DecoderLayer,
            'Decoder': Decoder,
            'TransformerModel': TransformerModel,
            'MultiHeadAttention': MultiHeadAttention,
            'PositionEmbeddingFixedWeights': PositionEmbeddingFixedWeights,
            'AddNormalization': AddNormalization,
            'FeedForward': FeedForward
        }
        model = keras.models.load_model(model_name, custom_objects=custom_objects)  # KERAS 2
        print("[MODEL] - LOADED - Keras 2")
        return model
    except Exception as e:
        custom_objects = {
            "MyMaskingLayer" : MyMaskingLayer,
            "CustomSinePositionEncoding" : CustomSinePositionEncoding
        }
        model = keras.models.load_model(model_name + ".keras", custom_objects=custom_objects)
        print("[MODEL] - LOADED - Keras 3")
        return model

def save_model(model, model_file_name):
    try:
        model.save(model_file_name)
        print("Model saved successfully the unsuccessful_attempts way")
    except Exception as e:
        model.save(model_file_name + ".keras")
        print("Model saved using KERAS 3")

def plot_attention_weights(attention_list, input_sentence, output_sentence, n, h, line_num, model_full_path):
    fig = plt.figure(figsize=(16, 8))
    # print("len(attention_list):", len(attention_list))
    for i, attention in enumerate(attention_list):
        # print("i, attention:", i, len(attention))
        # print("i, attention[-1]:", i, len(attention[-1]))
        attention = attention[-1][0].numpy()  # because it's surrounded by brackets
        # print("attention[-1][0].shape :", attention.shape)
        # if i == 1:
        #     continue
        for j, attention_head in enumerate(attention):
            ax = fig.add_subplot(n, h, i*h + j + 1)

            # Plot the attention weights
            ax.matshow(attention_head[:, :len(input_sentence)],
                       cmap='viridis')
            # ax.matshow(attention_head[:-1, 1:len(input_sentence)+1],
            #            cmap='viridis')
            # ax.matshow(attention_head,
            #            cmap='viridis')

            fontdict = {'fontsize': 10}

            ax.set_xticks(range(len(input_sentence)))
            ax.set_yticks(range(len(output_sentence)))

            ax.set_xticklabels(input_sentence, fontdict=fontdict, rotation=90)
            ax.set_yticklabels(output_sentence, fontdict=fontdict)

            ax.set_xlabel(f'Head {j + 1}')
    plt.tight_layout()
    folder_name = "plots/attention/"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    fig_name = model_full_path.split("/")[-1] + f"_{line_num}"
    plt.savefig(folder_name + fig_name, bbox_inches='tight')
    plt.show()

def visualise_attention(model, encoder_input_data, decoder_input_data, n, h, line_num, test_source, test_target, model_full_path):
    n_attention_scores = []
    for i in range(n):
        model = keras.Model(inputs=model.input,
                            outputs=[model.output, model.get_layer(f'cross_att{i}').output])
        _, attention_scores = model.call((encoder_input_data, decoder_input_data), training=False)
        n_attention_scores.append(attention_scores)

    input_sentence = []
    for token in encoder_input_data[0]:
        if token == 0:
            break
        if token == 3:  # "_"
            input_sentence.append(" ")
        # elif token == 1:  # remove the <bos> token
        #     pass
        # elif token == 2:  # remove the <eos> token
        #     pass
        else:
            input_sentence.append(test_source.reverse_dict[token])

    output_sentence = []
    for token in decoder_input_data[0]:
        if token == 3:  # "_"
            output_sentence.append(" ")
        # elif token == 1:  # remove the <bos> token
        #     pass
        else:
            output_sentence.append(test_target.reverse_dict[token])

    # placeholder code before i fill in the text
    # input_sentence = [str(i) for i in range(encoder_input_data.shape[1])]
    # output_sentence = [str(i) for i in range(decoder_input_data.shape[1])]

    plot_attention_weights(n_attention_scores, input_sentence, output_sentence, n, h, line_num, model_full_path)

def get_epochs_train_accuracy(history_dict):
    with open(history_dict, 'rb') as file_pi:
        history = pickle.load(file_pi)
        epochs = len(history['accuracy'])
        results = {
            "train_accuracy": history['accuracy'][-1],
            "val_accuracy": history['val_accuracy'][-1],
            "train_loss": history['loss'][-1],
            "val_loss": history['val_loss'][-1]
        }
    return epochs, results

def translate(model, encoder_input, output_maxlen, line_num):
    output_line = [1]
    # i = 1
    i = 0
    while i < output_maxlen:
        prediction = model.call((encoder_input, np.array([output_line])), training=False)  # enc shape: (1, maxlen), out shape: (1, j)
        # next_token_probs = prediction[0, -1, :]  # Prediction is shape (1, i, 63)
        next_token_probs = prediction[0, i, :]  # prediction has the whole sentence every time
        # next_token = np.random.choice(len(next_token_probs), p=next_token_probs)
        next_token = np.argmax(next_token_probs)
        if next_token == 0:
            break
        # Update the output sequence with the sampled token
        output_line.append(next_token)
        i += 1
    # try:
    #     visualise_attention(model, encoder_input, np.array([output_line]), n, h, line_num, test_source, test_target, model_full_path)
    # except Exception as e:
    #     print(f"Attention failed due to: {e}")
    return output_line

def test_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)