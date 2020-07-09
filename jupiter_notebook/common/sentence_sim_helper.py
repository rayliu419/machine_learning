import pandas as pd
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import chinese_preprocess


def prepare_raw_data_for_sentence_sim(line_num=100):
    # 1. 读取数据
    print("#1 load data")
    data_file = os.path.dirname(__file__) + "/input_data/atec_nlp_sim_train_all.csv"
    sentence_sim_data = pd.read_csv(data_file, error_bad_lines=False, sep="\t", header=None, dtype=str)
    # movie_reviews.info()
    sentence_sim_data = sentence_sim_data[0:line_num]
    return sentence_sim_data


def prepare_word_encoding_int_for_sentence_sim(line_num=100, max_len=100):
    # 1. 读数据
    print("#1 load data")
    data_file = os.path.dirname(__file__) + "/input_data/atec_nlp_sim_train_all.csv"
    sentence_sim_data = pd.read_csv(data_file, error_bad_lines=False, sep="\t", header=None, dtype=str)
    # movie_reviews.info()
    sentence_sim_data = sentence_sim_data[0:line_num]
    # 2. 预处理
    print("#2 preprocess")
    sen1 = sentence_sim_data[1].tolist()
    sen2 = sentence_sim_data[2].tolist()
    all_sen = sen1
    all_sen.extend(sen2)
    is_sim = sentence_sim_data[3].tolist()
    print("print sample data")
    print(sen1[1])
    print(sen2[1])
    print(is_sim[1])
    all_sen_seg_word_arr = chinese_preprocess.seg_chinese_word_array(all_sen)
    tok, vocab_size = chinese_preprocess.get_word_dict(all_sen_seg_word_arr)
    # sen1_seg_word_arr = chinese_preprocess.seg_chinese_word_array(sen1)
    # sen2_seg_word_arr = chinese_preprocess.seg_chinese_word_array(sen2)
    # tok, sen1_encode_int, vocab_size = chinese_preprocess.encode_chinese_word_to_int(sen1_seg_word_arr)
    # tok, sen2_encode_int, vocab_size = chinese_preprocess.encode_chinese_word_to_int(sen2_seg_word_arr)
    all_sen_encode_int = chinese_preprocess.encode_chinese_word_to_int_with_tok(tok, all_sen_seg_word_arr)
    print(all_sen_seg_word_arr[1])
    print(all_sen_encode_int[1])
    print("orig_size - %d, encode_size - %d" % (len(all_sen_seg_word_arr[1]), len(all_sen_encode_int[1])))
    print("#3 padding")
    print("#4 train/test split")
    return sen1, sen2, is_sim


if __name__ == "__main__":
    data = prepare_raw_data_for_sentence_sim(10000)
    print("data statistics")
    print(data.describe())
    sen1_arr, sen2_arr, is_sim_arr = prepare_word_encoding_int_for_sentence_sim(100)

