import numpy as np
import os
import pickle
from keras.utils import to_categorical
import time

from Data import Data

# TODO - check for lines of all zeros in tokens

# model_file_name = "transform2seq_LSTM_delete"
# train_in_file_name = "../data/smallvoc_fr_.txt"
# train_out_file_name = "../data/smallvoc_en_.txt"
# val_in_file_name = "../data/smallvoc_fr_.txt"
# val_out_file_name = "../data/smallvoc_en_.txt"
# test_in_file_name = "../data/smallervoc_fr_.txt"
# test_out_file_name = "../data/smallervoc_en_.txt"

# train_in_file_name = "../data/fr_train.txt"
# train_out_file_name = "../data/en_train.txt"
# val_in_file_name = "../data/fr_val.txt"
# val_out_file_name = "../data/en_val.txt"
# test_in_file_name = "../data/fr_test.txt"
# test_out_file_name = "../data/en_test.txt"


# train_in_file_name = "../data/src-sep-train.txt"
# train_out_file_name = "../data/tgt-train.txt"
# val_in_file_name = "../data/src-sep-val.txt"
# val_out_file_name = "../data/tgt-val.txt"
# test_in_file_name = "../data/src-sep-test.txt"
# test_out_file_name = "../data/tgt-test.txt"
# train_in_file_name = "../data/src-sep-train-short.txt"
# train_out_file_name = "../data/tgt-train-short.txt"
# val_in_file_name = "../data/src-sep-train-short.txt"
# val_out_file_name = "../data/tgt-train-short.txt"
# test_in_file_name = "../data/src-sep-train-short.txt"
# test_out_file_name = "../data/tgt-train-short.txt"
# train_in_file_name = "data/src-sep-train-short.txt"
# train_out_file_name = "data/tgt-train-short.txt"
# val_in_file_name = "data/src-sep-train-short.txt"
# val_out_file_name = "data/tgt-train-short.txt"
# test_in_file_name = "data/src-sep-train-short.txt"
# test_out_file_name = "data/tgt-train-short.txt"


# sep = ' '
# mezera = '_'
# end_line = '\n'


print()


def prepare_data(run_settings, skip_valid=False, files=None, files_val=None, files_additional_train=None):
    print("data preparation...")

    train_in_file_name, train_out_file_name = files
    if not skip_valid: val_in_file_name, val_out_file_name = files_val

    print("training files:", train_in_file_name, train_out_file_name)

    # class object inicialised, data added
    source = Data(run_settings["sep"], run_settings["mezera"], run_settings["end_line"])
    target = Data(run_settings["sep"], run_settings["mezera"], run_settings["end_line"])
    with open(train_in_file_name, "r", encoding="utf-8") as f:  # with spaces
        source.file = f.read()
        f.close()
    with open(train_out_file_name, "r", encoding="utf-8") as ff:
        target.file = ff.read()
        ff.close()

    if files_additional_train is None:
        print("first file:")
        x_train = source.split_n_count(True)
        source.padded = source.padding(x_train, source.maxlen)
        print("second file:")
        y_train = target.split_n_count(True)
        target.padded = target.padding(y_train, target.maxlen)
        # y_train_pad_one = to_categorical(y_train_pad)
        target.padded_shift = target.padding_shift(y_train, target.maxlen)
        target.padded_shift_one = to_categorical(target.padded_shift)

    elif files_additional_train:
        print("finetuning training files:", files_additional_train)
        file_in, file_out = files_additional_train
        with open(file_in, "r", encoding="utf-8") as f:
            additional_source_file = f.read()
            f.close()
        with open(file_out, "r", encoding="utf-8") as ff:
            additional_target_file = ff.read()
            ff.close()
        source.file += additional_source_file  # adding data to extend the vocabulary of the model
        target.file += additional_target_file
        _ = source.split_n_count(True)
        _ = target.split_n_count(True)

        source_maxlen = source.maxlen
        target_maxlen = target.maxlen

        source.file = additional_source_file  # removing the original data from the training
        target.file = additional_target_file

        # main pipeline
        print("Additional files being processed...")
        x_train = source.split_n_count(False)  # dict alreasy created from the bigger file
        source.maxlen = source_maxlen
        source.padded = source.padding(x_train, source.maxlen)
        y_train = target.split_n_count(False)
        target.maxlen = target_maxlen
        target.padded = target.padding(y_train, target.maxlen)
        target.padded_shift = target.padding_shift(y_train, target.maxlen)
        target.padded_shift_one = to_categorical(target.padded_shift, num_classes=len(target.dict_chars))

    assert len(source.padded) == len(target.padded_shift)
    assert len(source.padded) == len(target.padded_shift_one)

    print(np.array(source.padded).shape)            # (1841, 98)
    print(np.array(target.padded).shape)            # (1841, 109)
    print(np.array(target.padded_shift).shape)      # (1841, 109)
    print(np.array(target.padded_shift_one).shape)  # (1841, 109, 55)

    del source.file
    del target.file
    val_source, val_target = None, None

    assert len(source.padded) == len(target.padded_shift)
    assert len(source.padded) == len(target.padded_shift_one)

    if not skip_valid:
        # VALIDATION:
        print("validation files:")
        val_source = Data(run_settings["sep"], run_settings["mezera"], run_settings["end_line"])
        val_target = Data(run_settings["sep"], run_settings["mezera"], run_settings["end_line"])
        with open(val_in_file_name, "r", encoding="utf-8") as f:
            val_source.file = f.read()
            f.close()
        with open(val_out_file_name, "r", encoding="utf-8") as ff:
            val_target.file = ff.read()
            ff.close()

        val_source.dict_chars = source.dict_chars
        x_val = val_source.split_n_count(False)
        val_source.padded = val_source.padding(x_val, source.maxlen)

        val_target.dict_chars = target.dict_chars
        y_val = val_target.split_n_count(False)
        val_target.padded = val_target.padding(y_val, target.maxlen)
        val_target.padded_shift = val_target.padding_shift(y_val, target.maxlen)
        val_target.padded_shift_one = to_categorical(val_target.padded_shift, num_classes=len(target.dict_chars))

        # print("source.maxlen:", source.maxlen)
        # print("target.maxlen:", target.maxlen)
        # print("source_val.maxlen:", val_source.maxlen)
        # print("target_val.maxlen:", val_target.maxlen)
        # print("source.dict:", len(source.dict_chars))
        # print("target.dict:", len(target.dict_chars))
        # print("source_val.dict:", len(val_source.dict_chars))
        # print("target_val.dict:", len(val_target.dict_chars))

        assert len(x_val) == len(val_source.padded)
        assert len(x_val) == len(y_val)
        assert len(val_source.padded) == len(val_target.padded_shift)
        assert len(val_source.padded) == len(val_target.padded_shift_one)

        print(np.array(val_source.padded).shape)            # (1841, 98)
        print(np.array(val_target.padded).shape)            # (1841, 109)
        print(np.array(val_target.padded_shift).shape)      # (1841, 109)
        print(np.array(val_target.padded_shift_one).shape)  # (1841, 109, 55)

        del val_source.file
        del val_target.file

    # ValueError: Shapes (None, 109, 55) and (None, 109, 63) are incompatible

    # print(y_train_pad_one)
    # print()
    # print(x_train_pad.shape)
    # print(y_train_pad.shape)
    # print(y_train_pad_shift.shape)
    # print(y_train_pad_one.shape)

    print("Data prepared")
    return source, target, val_source, val_target


def get_history_dict(dict_name, new):
    dict_exist = os.path.isfile(dict_name)
    if dict_exist:
        if new:
            # q = input(f"Dict with the name {dict_name} exist but we create a new one, ok?")
            q = "ok"
            print("Rewritting the dict")
            if q == "ok":
                return {}
            else:
                raise Exception("Dont do this")
        else:
            with open(dict_name, "rb") as file_pi:
                old_dict = pickle.load(file_pi)
                return old_dict
    return {}

def join_dicts(dict1, dict2):
    dict = {}
    if dict1 == {}:
        dict = dict2

    if dict1.keys() == dict2.keys():
        pass
    else:
        print(dict1.keys(), " != ", dict2.keys())

    for i in range(len(dict1.keys())):
        history_list = list(dict1.keys())
        # print(history_list[i])
        ar = []
        for item in dict1[history_list[i]]:
            ar.append(item)
        for item in dict2[history_list[i]]:
            ar.append(item)
        dict[history_list[i]] = ar
    # print(dict)
    return dict

def cache_dict(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)
        print("Dict successfully cached")

def load_cached_dict(filename):
    if os.path.exists(filename):
        with open(filename, 'rb') as f:
            loaded_dict = pickle.load(f)
            if loaded_dict:
                print("Loaded dictionary from:", filename)
                return loaded_dict
            else:
                print("Empty dictionary loaded from:", filename)
                return {}
    else:
        print("No cached dictionary found at:", filename)
        return {}

def split_by_underscore(input_list):
    result = [list(group) for group in ''.join(input_list).split('_') if group]
    return result

def create_new_class_dict(run_settings):
    train_in_file_name = run_settings["train_in_file_name"]
    train_out_file_name = run_settings["train_out_file_name"]
    val_in_file_name = run_settings["val_in_file_name"]
    val_out_file_name = run_settings["val_out_file_name"]

    class_data = run_settings["class_data"]

    finetune_source = run_settings["finetune_source"]
    finetune_tgt = run_settings["finetune_tgt"]

    start = time.time()
    print("[DATA] - preparation started")
    if run_settings['finetune_model']:
        source, target, val_source, val_target = prepare_data(run_settings, skip_valid=False, files=[train_in_file_name, train_out_file_name], files_val=[finetune_source, finetune_tgt],
                                                              files_additional_train=[finetune_source, finetune_tgt])
    else:
        source, target, val_source, val_target = prepare_data(run_settings, skip_valid=False, files=[train_in_file_name, train_out_file_name], files_val=[val_in_file_name, val_out_file_name], )
    to_save_list = [source, target, val_source, val_target]
    end = time.time()
    print("[DATA] - preparation finished")
    print("[DATA] - preparation of data took:", end - start)

    def save_object(obj, filename):
        with open(filename, 'wb') as outp:  # Overwrites any existing file.
            pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

    print("[DATA] - saving started")
    save_start = time.time()
    save_object(to_save_list, class_data)
    save_end = time.time()
    print("[DATA] - saving finished")
    print("[DATA] - saving took: ", save_end - save_start)
    return source, target, val_source, val_target


def load_class_data(run_settings):
    class_data = run_settings["class_data"]
    start = time.time()
    print("[DATA] - loading DATA classs")
    with open(class_data, 'rb') as class_data_dict:
        source, target, val_source, val_target = pickle.load(class_data_dict)
        end = time.time()
    print("[DATA] - loadig finished")
    print("[DATA] - loadig took:", end - start)
    return source, target, val_source, val_target

if __name__ == "__main__":
    source, target, val_source, val_target = prepare_data()