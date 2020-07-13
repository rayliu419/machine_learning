import pandas as pd
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import chinese_preprocess
import type_cast_helper


def prepare_raw_data_for_sentence_sim(line_num=100):
    # 1. 读取数据
    print("#1 load data")
    data_file = os.path.dirname(__file__) + "/input_data/atec_nlp_sim_train_all.csv"
    sentence_sim_data = pd.read_csv(data_file, error_bad_lines=False, sep="\t", header=None, dtype=str)
    # movie_reviews.info()
    sentence_sim_data = sentence_sim_data[0:line_num]
    return sentence_sim_data


def prepare_word_encoding_int_for_sentence_sim(line_num=100, max_len=20):
    # 1. 读数据
    print("#1 load data")
    data_file = os.path.dirname(__file__) + "/input_data/atec_nlp_sim_train_all.csv"
    sentence_sim_data = pd.read_csv(data_file, error_bad_lines=False, sep="\t", header=None, dtype=str)
    if line_num != -1:
        sentence_sim_data = sentence_sim_data[0:line_num]
    print("total line number - %d" % (len(sentence_sim_data)))
    # 2. 预处理
    print("#2 build vocab dict")
    # 使用所有的数据集build 词典，先分词
    sen1 = sentence_sim_data[1].tolist()
    sen2 = sentence_sim_data[2].tolist()
    all_sen = sen1
    all_sen.extend(sen2)
    all_sen_seg_word_arr = chinese_preprocess.seg_chinese_word_array(all_sen)
    tok, vocab_size = chinese_preprocess.get_word_dict(all_sen_seg_word_arr)

    print("#3 train/test split")
    sim_pair = sentence_sim_data.iloc[:, 1:3]
    is_sim = sentence_sim_data.iloc[:, 3]
    X_train, X_test, Y_train, Y_test = train_test_split(sim_pair, is_sim, test_size=0.20, random_state=42)

    print("#4 encoding to int")
    X_train_sen1 = type_cast_helper.series_to_list(X_train[1])
    X_train_sen2 = type_cast_helper.series_to_list(X_train[2])
    X_train_sen1_seg_word = chinese_preprocess.seg_chinese_word_array(X_train_sen1)
    X_train_sen2_seg_word = chinese_preprocess.seg_chinese_word_array(X_train_sen2)
    X_train_sen1_encode_int = chinese_preprocess.encode_chinese_word_to_int_with_tok(tok, X_train_sen1_seg_word)
    X_train_sen2_encode_int = chinese_preprocess.encode_chinese_word_to_int_with_tok(tok, X_train_sen2_seg_word)

    X_test_sen1 = type_cast_helper.series_to_list(X_test[1])
    X_test_sen2 = type_cast_helper.series_to_list(X_test[2])
    X_test_sen1_seg_word = chinese_preprocess.seg_chinese_word_array(X_test_sen1)
    X_test_sen2_seg_word = chinese_preprocess.seg_chinese_word_array(X_test_sen2)
    X_test_sen1_encode_int = chinese_preprocess.encode_chinese_word_to_int_with_tok(tok, X_test_sen1_seg_word)
    X_test_sen2_encode_int = chinese_preprocess.encode_chinese_word_to_int_with_tok(tok, X_test_sen2_seg_word)

    Y_train = type_cast_helper.series_to_list(Y_train)
    Y_test = type_cast_helper.series_to_list(Y_test)
    Y_train = list(map(int, Y_train))
    Y_test = list(map(int, Y_test))
    print(X_train_sen1_seg_word[1])
    print(X_train_sen1_encode_int[1])
    print(X_train_sen2_seg_word[1])
    print(X_train_sen2_encode_int[1])
    print(Y_train[1])
    print("orig_size - %d, encode_size - %d" % (len(X_train_sen1_seg_word[1]), len(X_train_sen1_encode_int[1])))
    print("orig_size - %d, encode_size - %d" % (len(X_train_sen2_seg_word[1]), len(X_train_sen2_encode_int[1])))

    print("#5 padding")
    if max_len != -1:
        X_train_sen1_encode_int = pad_sequences(X_train_sen1_encode_int, padding='post', maxlen=max_len)
        X_train_sen2_encode_int = pad_sequences(X_train_sen2_encode_int, padding='post', maxlen=max_len)
        X_test_sen1_encode_int = pad_sequences(X_test_sen1_encode_int, padding='post', maxlen=max_len)
        X_test_sen2_encode_int = pad_sequences(X_test_sen2_encode_int, padding='post', maxlen=max_len)
        print(X_train_sen1_encode_int[1])
        print(X_train_sen2_encode_int[1])

    # return final result.
    return tok, vocab_size, X_train_sen1_encode_int, X_train_sen2_encode_int, X_test_sen1_encode_int, X_test_sen2_encode_int, \
           Y_train, Y_test


if __name__ == "__main__":
    data = prepare_raw_data_for_sentence_sim(100)
    print("data statistics")
    print(data.describe())
    prepare_word_encoding_int_for_sentence_sim(100)

