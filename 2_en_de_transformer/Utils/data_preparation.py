import os
import pickle
from keras.utils import to_categorical
import time

from Data import Data

# TODO - check for lines of all zeros in tokens



def prepare_data(run_settings, skip_valid=False, files=None, files_val=None, files_additional_train=None):
    train_in_file_name, train_out_file_name = files
    if not skip_valid: val_in_file_name, val_out_file_name = files_val

    print("[DATA] - processing training files:", train_in_file_name, train_out_file_name)

    source = Data(run_settings["sep"], run_settings["mezera"], run_settings["end_line"])
    target = Data(run_settings["sep"], run_settings["mezera"], run_settings["end_line"])

    with open(train_in_file_name, "r", encoding="utf-8") as f:  # with spaces
        source.file = f.read()
        f.close()
    with open(train_out_file_name, "r", encoding="utf-8") as ff:
        target.file = ff.read()
        ff.close()

    if files_additional_train is None:
        print("[DATA] - first training file...")
        x_train = source.split_n_count(create_dic=True)
        source.padded = source.padding(x_train, source.maxlen)
        print("[DATA] - second training file...")
        y_train = target.split_n_count(create_dic=True)
        target.padded = target.padding(y_train, target.maxlen)
        # y_train_pad_one = to_categorical(y_train_pad)
        target.padded_shift = target.padding_shift(y_train, target.maxlen)
        target.padded_shift_one = to_categorical(target.padded_shift)

    elif files_additional_train:
        print("[DATA] - finetuning training files:", files_additional_train)
        file_in, file_out = files_additional_train
        with open(file_in, "r", encoding="utf-8") as f:
            additional_source_file = f.read()
            f.close()
        with open(file_out, "r", encoding="utf-8") as ff:
            additional_target_file = ff.read()
            ff.close()
        source.file += additional_source_file  # adding data to extend the vocabulary of the model
        target.file += additional_target_file
        _ = source.split_n_count(create_dic=True)
        _ = target.split_n_count(create_dic=True)

        source_maxlen = source.maxlen
        target_maxlen = target.maxlen

        source.file = additional_source_file  # removing the original data from the training
        target.file = additional_target_file

        # main pipeline
        print("[DATA] - Additional files being processed...")
        x_train = source.split_n_count(create_dic=False)  # dict already created from the bigger file
        source.maxlen = source_maxlen
        source.padded = source.padding(x_train, source.maxlen)
        y_train = target.split_n_count(create_dic=False)
        target.maxlen = target_maxlen
        target.padded = target.padding(y_train, target.maxlen)
        target.padded_shift = target.padding_shift(y_train, target.maxlen)
        target.padded_shift_one = to_categorical(target.padded_shift, num_classes=len(target.dict_chars))
        print("[DATA] - Additional files processed.")

    assert len(source.padded) == len(target.padded_shift)
    assert len(source.padded) == len(target.padded_shift_one)

    del source.file
    del target.file

    val_source, val_target = None, None
    if not skip_valid:
        # VALIDATION:
        print("[DATA] - validation files")
        val_source = Data(run_settings["sep"], run_settings["mezera"], run_settings["end_line"])
        val_target = Data(run_settings["sep"], run_settings["mezera"], run_settings["end_line"])

        with open(val_in_file_name, "r", encoding="utf-8") as f:
            val_source.file = f.read()
            f.close()
        with open(val_out_file_name, "r", encoding="utf-8") as ff:
            val_target.file = ff.read()
            ff.close()

        val_source.dict_chars = source.dict_chars
        x_val = val_source.split_n_count(create_dic=False)
        val_source.padded = val_source.padding(x_val, source.maxlen)

        val_target.dict_chars = target.dict_chars
        y_val = val_target.split_n_count(create_dic=False)
        val_target.padded = val_target.padding(y_val, target.maxlen)
        val_target.padded_shift = val_target.padding_shift(y_val, target.maxlen)
        val_target.padded_shift_one = to_categorical(val_target.padded_shift, num_classes=len(target.dict_chars))

        print("[DATA] - validation files processed")

        assert len(x_val) == len(val_source.padded)
        assert len(x_val) == len(y_val)
        assert len(val_source.padded) == len(val_target.padded_shift)
        assert len(val_source.padded) == len(val_target.padded_shift_one)

        del val_source.file
        del val_target.file

    print("[DATA] - prepared")
    return source, target, val_source, val_target


def get_history_dict(dict_name, new):
    dict_exist = os.path.isfile(dict_name)
    if dict_exist:
        if new:
            print("Rewriting History Dict")
            return {}
        else:
            print("Loading History Dict")
            with open(dict_name, "rb") as file_pi:
                old_dict = pickle.load(file_pi)
                return old_dict
    return {}
def join_dicts(dict1, dict2):
    dict = {}
    if dict1 == {}:
        dict = dict2

    if dict1.keys() == dict2.keys(): pass
    else: print(f'{dict1.keys()=} != {dict2.keys()=}')

    history_list = list(dict1.keys())
    for i in range(len(dict1.keys())):
        ar = []
        for item in dict1[history_list[i]]: ar.append(item)
        for item in dict2[history_list[i]]: ar.append(item)
        dict[history_list[i]] = ar
    return dict

def cache_dict(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dictionary, f)
        print("Dict successfully saved: {}".format(filename))
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

def create_new_class_dict_testing(run_settings, source, target):
    test_in_file_name = run_settings["test_in_file_name"]
    test_out_file_name = run_settings["test_out_file_name"]
    testing_samples = run_settings["testing_samples"]

    print("[TESTING] - data preparation")
    test_source = Data(run_settings["sep"], run_settings["mezera"], run_settings["end_line"])
    test_target = Data(run_settings["sep"], run_settings["mezera"], run_settings["end_line"])

    if run_settings['use_custom_testing']:
        test_in_file_name = run_settings['custom_test_src']
        test_out_file_name = run_settings['custom_test_tgt']

    with open(test_in_file_name, "r", encoding="utf-8") as f:  # with spaces
        test_source.file = f.read()
        f.close()
    with open(test_out_file_name, "r", encoding="utf-8") as f:  # with spaces
        test_target.file = f.read()
        f.close()

    test_source.dict_chars = source.dict_chars
    if testing_samples == -1:
        x_test = test_source.split_n_count(False)
    else:
        x_test = test_source.split_n_count(False)[:testing_samples]
    test_source.padded = test_source.padding(x_test, source.maxlen)

    test_target.dict_chars = target.dict_chars
    if testing_samples == -1:
        y_test = test_target.split_n_count(False)
    else:
        y_test = test_target.split_n_count(False)[:testing_samples]

    test_target.padded = test_target.padding(y_test, target.maxlen)
    test_target.padded_shift = test_target.padding_shift(y_test, target.maxlen)

    assert len(x_test) == len(y_test)
    del x_test, y_test

    test_source.create_reverse_dict(test_source.dict_chars)
    return test_source, test_target


if __name__ == "__main__":
    source, target, val_source, val_target = prepare_data()