# 从SQuAD 2.0出去文本数据，加入随机噪音生成训练集。
# 比较不同的model的纠错效果。
# 英文数据
# encoder-decoder 架构

import tensorflow as tf
import keras
import math
import random
import numpy as np
import pandas as pd
from numpy.random import choice as random_choice, randint as random_randint
import sys
import json
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Masking, Input
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

english_text_file = "./input_data/abstract_english_text_file"
clean_english_text_file = "./input_data/clean_english_text_file"
before_add_error_file = "./input_data/before_add_error"
after_add_error_file = "./input_data/after_add_error"
change_index_file = "./input_data/change_index_file"
MAX_LINE_NUMBER = 500

err_prob = {
    "replace_one_char": 0.4,
    "add_one_char": 0.2,
    "delete_one_char": 0.2,
    "change_neighbor_order": 0.2
}

# max_error_rate = 0.2
char_list = list("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .")
max_error_line_length_rate = 0.05


# 函数用来parse SQuAD 2.0的数据
# 我们只需要抽取文本信息, SQuAD2.0的数据格式比较奇怪
def read_squad_examples(input_file, is_training):
    with tf.io.gfile.GFile(input_file, "r") as reader:
        input_data = json.load(reader)["data"]
    return input_data


def isEnglish(s):
    try:
        s.encode(encoding='utf-8').decode('ascii')
    except UnicodeDecodeError:
        return False
    else:
        return True


# 抽取英文文本，中文和其他类别的去掉
def abstarct_sentence(json_data):
    global english_text_file
    fp = open(english_text_file, "w")
    for article in json_data:
        paragraphs = article["paragraphs"]
        for paragraph in paragraphs:
            qas = paragraph["qas"]
            for qa in qas:
                question = qa["question"]
                if isEnglish(question):
                    fp.write(question + "\n")
                answer_struct = qa["answers"]
                if (len(answer_struct) > 0):
                    answer = answer_struct[0]["text"]
                    if isEnglish(answer):
                        fp.write(answer + "\n")
    fp.close()
    print("abstract finish")


def clean_file(file=english_text_file):
    global clean_english_text_file
    fp = open(file, "r")
    fp2 = open(clean_english_text_file, "w")
    line_num = 0
    for line in fp:
        line = line.strip()
        if line != "" and line != "null" and len(line) > 10:
            fp2.write(line + "\n")
        line_num += 1
        if (line_num == MAX_LINE_NUMBER):
            break
    fp.close()
    fp2.close()


def map_prob_to_range_with_keys(key_prob):
    prob_list = list(key_prob.values())
    prob_sum = sum(map(float,prob_list))
    key_range = dict()
    if not math.isclose(prob_sum, 1):
        print("prob sum is not 1")
        sys.exit(-1)
    else:
        threshold = 0
        for key, prob in key_prob.items():
            key_range[key] = [threshold, threshold + prob]
            threshold += prob
    return key_range


def choose_item_based_on_prob(key_range):
    value = random.uniform(0, 1)
    last_key = None
    for key, prob_range in key_range.items():
        last_key = key
        if value >= prob_range[0] and value < prob_range[1]:
            return key
    return last_key


# 随机修改正确的句子到错误的句子
# 包括添加字符，删除字符，交换临近字符，替换字符
def add_error_to_line(line):
    max_error_line_length_rate
    max_error_num = (int)(max_error_line_length_rate * len(line))
    set_error_num = (int)(random.uniform(0, 1) * max_error_num)
    cur_error_num = 0
    before = line
    after = line
    key_range = map_prob_to_range_with_keys(err_prob)
    while cur_error_num < set_error_num:
        err_type = choose_item_based_on_prob(key_range)
        cur_error_num += 1
        if err_type == "replace_one_char":
            random_char_position = random_randint(len(after))
            after = after[:random_char_position] + random_choice(char_list[:-1]) \
                    + after[random_char_position + 1:]
        elif err_type == "add_one_char":
            random_char_position = random_randint(len(after))
            after = after[:random_char_position] + random_choice(char_list[:-1]) \
                    + after[random_char_position:]
        elif err_type == "delete_one_char":
            random_char_position = random_randint(len(after))
            after = after[:random_char_position] + after[random_char_position + 1:]
        elif err_type == "change_neighbor_order":
            random_char_position = random_randint(len(after) - 1)
            after = (after[:random_char_position] + after[random_char_position + 1] \
                     + after[random_char_position] + after[random_char_position + 2:])
    return before, after


def gen_X_y(tk):
    global clean_english_text_file, before_add_error_file, after_add_error_file
    X = []
    Y = []
    X_encoding = []
    Y_encoding = []
    before_file = open(before_add_error_file, "w")
    after_file = open(after_add_error_file, "w")
    change_index_fp = open(change_index_file, "w")
    ori_file = open(clean_english_text_file, "r")
    line_num = 1
    print_test = True
    for line in ori_file:
        line = line.strip("\n")
        correct, mistaken = add_error_to_line(line)
        X.append([mistaken])
        Y.append([correct])
        before_file.write(correct + "\n")
        after_file.write(mistaken + "\n")
        if correct != mistaken:
            change_index_fp.write(str(line_num) + "\n")
            if print_test:
                print(correct)
                print(mistaken)
            print_test = False
        line_num += 1
        X_encoding.append(tk.texts_to_sequences([mistaken])[0])
        Y_encoding.append(tk.texts_to_sequences([correct])[0])
    np_array_X = np.array([np.array(xi) for xi in X_encoding])
    np_array_Y = np.array([np.array(yi) for yi in Y_encoding])
    # np_array_X 是错的句子
    # np_array_Y 是对的句子
    return np_array_X, np_array_Y


# DL要求将字符转成int作为输入
def char2int(file=clean_english_text_file):
    fp = open(file, "r")
    all_text = fp.read()
    fp.close()
    tk = Tokenizer(num_words=None, char_level=True, oov_token="UNK", lower=False)
    tk.fit_on_texts(all_text)
    return tk


# pad 2d 数组
def pad_sequence(array_2d, arg_max_len=None):
    max_len = max(len(a) for a in array_2d)
    if arg_max_len:
        max_len = arg_max_len
    return pad_sequences(array_2d, padding="post", maxlen=max_len)


def prepare_test_sentences(tk, sentences, max_length, char_table_size):
    sentences_int = tk.texts_to_sequences(sentences)
    sample_num = len(sentences_int)
    np_array = np.array([np.array(xi) for xi in sentences_int])
    sentences_int_pad_one_hot = one_hot_encode_2d_array(np_array, sample_num, True, max_length, char_table_size)
    return sentences_int_pad_one_hot


def get_dict_index_back(ndarray):
    result = []
    for row in range(0, ndarray.shape[1]):
        mapping_int = np.argmax(ndarray[0][row])
        result.append(mapping_int)
    return result


def reverse_dict(tk, int_array):
    char_result = []
    inverse_dict = {v: k for k, v in tk.word_index.items()}
    for dict_index in int_array:
        if dict_index in inverse_dict:
            char_result.append(inverse_dict[dict_index])
        else:
            char_result.append(" ")
    return char_result


def reset_zero(one_hot):
    find_EOS = False
    for i in range(one_hot.shape[0]):
        for j in range(one_hot.shape[1]):
            if j == 0 and one_hot[i][j] == 1:
                if find_EOS:
                    one_hot[i][j] = 0
                else:
                    find_EOS = True
                break


# 将2d转成one_hot的3d
def one_hot_encode_2d_array(array_2d, sample_num, need_pad, pad_length, code_table_size, pad_clear=True):
    np_pad = array_2d
    if need_pad:
        np_pad = pad_sequence(array_2d, pad_length)
    np_pad_one_hot = np.empty(shape=(sample_num, pad_length, code_table_size))
    for index in range(0, sample_num):
        cur_one_hot = to_categorical(np_pad[index], num_classes=code_table_size)
        if pad_clear:
            reset_zero(cur_one_hot)
        np_pad_one_hot[index] = cur_one_hot
    print(np_pad_one_hot.shape)
    return np_pad_one_hot


def save_model_history_to_file(history, filename):
    hist_df = pd.DataFrame.from_dict(history.history)
    # save to json:
    hist_json_file = filename
    with open(hist_json_file, mode='w') as f:
        hist_df.to_csv(f)


def train_wrong_one_hot_simple_char_model(np_pad_X, np_pad_Y, pad_max_length, char_table_size):
    """ 使用自编码的one_hot来作为输入和输出
    这种模型不是真的有效果，主要是填充的值用什么表示的问题，我尝试过几种方法:
    1. 默认用0的to_categorical，会在0位置1。 - 由于大量的这种数据存在，直观上来说会导致预测都向空字符。
    2. 0->0 向量，也不行。
    3. 第一个结束用[1,0,0...]，后续用[0,0,0]表示。- 有些例子中这样做，但是我用了还是不行。
    然而都不行。
    所以在输入层还是必须要显式的告诉哪些是有用的。
    """
    sample_num = np_pad_Y.shape[0]
    np_pad_one_hot_Y = one_hot_encode_2d_array(np_pad_Y, sample_num, False, pad_max_length, char_table_size, True)
    np_pad_one_hot_X = one_hot_encode_2d_array(np_pad_X, sample_num, False, pad_max_length, char_table_size, True)
    print('Build model...')
    model = Sequential()
    # LSTM作为首层才需要设置input_shape参数。
    model.add(LSTM(20, input_shape=(pad_max_length, char_table_size), return_sequences=True))
    model.add(LSTM(20, return_sequences=True))
    model.add(Dense(char_table_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    plot_model(model, to_file='model.png', show_shapes=True)
    model.fit(np_pad_one_hot_X, np_pad_one_hot_Y, epochs=10, verbose=1, batch_size=64)
    return model


def train_simple_embedding_model(max_len, feature_num, output_unit, np_pad_X, np_pad_Y):
    """
    这并不是传统的seq2seq，因为decoder的输入Y_predict(i+1)不是根据Y_predict(i)和decoder的state计算的。
    加入了masking，并且使用Embedding层来编码，效果训练1小时能达到60%
    这是一种自己创建的方法，看起来也是有效果的。
    """
    char_table_size = feature_num
    sample_num = np_pad_Y.shape[0]
    np_pad_one_hot_Y = one_hot_encode_2d_array(np_pad_Y, sample_num, False, max_len, char_table_size, False)
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim=feature_num, input_length=max_len, output_dim=10, mask_zero=True))
    model.add(LSTM(20, return_sequences=True))
    model.add(LSTM(20, return_sequences=True))
    model.add(Dense(output_unit, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(np_pad_X, np_pad_one_hot_Y, epochs=10, verbose=1, batch_size=64)
    save_model_history_to_file(history, "simple_mapping.csv")
    return model


def train_non_teacher_forcing_model(max_len, feature_num, output_unit, np_pad_X, np_pad_Y):
    """
    decoder的输出是一次一个timestep，因为LSTM的新的预测是基于上一个output来做的。
    完全train不动，感觉是哪里写的有问题。但是框架大概是这样的
    """
    # 为实参输入做准备
    max_len_adjust = max_len + 1
    np_encoder_input_pad = pad_sequences(np_pad_X, padding="post", maxlen=max_len_adjust)
    np_dense_pad = pad_sequences(np_pad_Y, padding="post", maxlen=max_len_adjust)
    sample_num = np_dense_pad.shape[0]
    np_dense_pad_one_hot_Y = one_hot_encode_2d_array(np_dense_pad, sample_num, False, max_len_adjust, char_table_size)
    decoder_input_data = np.zeros((sample_num, 1, feature_num))

    embedding_input = Input(shape=(max_len_adjust, ))
    embedding_layer = Embedding(input_dim=feature_num, input_length=max_len_adjust, output_dim=10)
    embedding_output = embedding_layer(embedding_input)

    encoder_layer = LSTM(20, return_state=True)
    encoder_output, encoder_last_output, encoder_last_state = encoder_layer(embedding_output)
    states = [encoder_last_output, encoder_last_state]

    decoder_input = Input(shape=(1, feature_num))
    decoder_layer = LSTM(20, return_sequences=True, return_state=True)
    dense_layer = Dense(output_unit,  activation="softmax")

    all_output = []
    cur_input = decoder_input
    for _ in range(max_len_adjust):
        decoder_output, decoder_last_output, decoder_last_state = decoder_layer(cur_input, initial_state=states)
        dense_output = dense_layer(decoder_output)
        all_output.append(dense_output)
        cur_input = dense_output
        states = [decoder_last_output, decoder_last_state]
    predict_outputs = keras.backend.concatenate(all_output, axis=1)
    model = Model([embedding_input, decoder_input], predict_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # plot_model(model, to_file='model.png', show_shapes=True)

    # model = Model([embedding_input, decoder_input], predict_outputs)
    # model.fit([np_encoder_input_pad, decoder_input_data], np_dense_pad_one_hot_Y) 比较
    model.fit([np_encoder_input_pad, decoder_input_data], np_dense_pad_one_hot_Y, epochs=10, verbose=1, batch_size=64)
    return model


def train_teacher_forcing_model(max_len, feature_num, output_unit, np_pad_X, np_pad_Y):
    """
    使用Function APi来定义teacher forcing model.
    1. 先embedding.
    2. 将embedding的output传给encoder LSTM, 在teacher forcing model下，只需要last state
    3. decoder的input跟simple embedding的输入不一样。
    simple embedding的序列Y_predict(i+1)是根据Y_predict(i)和decoder的state计算的。
    teacher forcing modelY_predict(i+1)是根据Y_actual(i)和decoder的state计算的。
    从效果上来说，还不如simple_mapping...
    TODO: 可以不适用Embedding的来train，而是使用gensim的pre-train的向量看看
    """
    # 为teacher_forcing fit model准备实参
    max_len_adjust = max_len + 1
    np_encoder_input_pad = pad_sequences(np_pad_X, padding="post", maxlen=max_len_adjust)
    np_decoder_input_pad = pad_sequences(np_pad_Y, padding="pre", maxlen=max_len_adjust)
    np_dense_pad = pad_sequences(np_pad_Y, padding="post", maxlen=max_len_adjust)
    sample_num = np_dense_pad.shape[0]
    np_dense_pad_one_hot_Y = one_hot_encode_2d_array(np_dense_pad, sample_num, False, max_len_adjust, char_table_size)

    # 定义encoder网络
    # 接受一个max_len长度的sequence的int
    encoder_input_ph = Input(shape=(max_len_adjust, ))
    encoder_embedding_layer = Embedding(input_dim=feature_num, input_length=max_len_adjust, output_dim=10, mask_zero=True)
    embedding_output_ph = encoder_embedding_layer(encoder_input_ph)
    encoder_LSTM_layer = LSTM(20, return_state=True)
    _, encoder_last_output, encoder_last_state = encoder_LSTM_layer(embedding_output_ph)
    decoder_initial_state = [encoder_last_output, encoder_last_state]

    # 定义decoder网络
    decoder_input_ph = Input(shape=(max_len_adjust, ))
    # 设置return_state是为了inference
    # 通过initial_state将encoder和decoder连接在了一起
    decoder_embedding_layer = Embedding(input_dim=feature_num, input_length=max_len_adjust, output_dim=10, mask_zero=True)
    embedding_output_ph2 = decoder_embedding_layer(decoder_input_ph)

    decoder_LSTM_layer = LSTM(20, return_sequences=True, return_state=True)
    decoder_output_ph, _, _ = decoder_LSTM_layer(embedding_output_ph2, initial_state=decoder_initial_state)
    dense_layer = Dense(output_unit, activation="softmax")
    final_dense_output = dense_layer(decoder_output_ph)

    model = Model([encoder_input_ph, decoder_input_ph], final_dense_output)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    plot_model(model, to_file='model.png', show_shapes=True)
    # model = Model([encoder_input_ph, decoder_input_ph], final_dense_output)
    # model.fit([np_encoder_input_pad, np_decoder_input_pad], np_dense_pad_one_hot_Y)
    # 合起来看，第一行其实是用占位符描述了, 但是最后的参数 final_dense_output 和 np_pad_one_hot_Y不一样。
    history = model.fit([np_encoder_input_pad, np_decoder_input_pad], np_dense_pad_one_hot_Y, epochs=10, verbose=1, batch_size=64)
    save_model_history_to_file(history, "teacher_forcing.csv")
    return model


def test_one_hot_simple_model(model, tk, sentences, max_length, char_table_size):
    test_np_pad_one_hot_X = prepare_test_sentences(tk, sentences, max_length, char_table_size)
    # print(test_np_pad_one_hot_X)
    predict_Y = model.predict(test_np_pad_one_hot_X, verbose=1)
    predict_Y_int = get_dict_index_back(predict_Y)
    final_result = reverse_dict(tk, predict_Y_int)
    print("input======================================================================")
    print(sentences)
    print(test_np_pad_one_hot_X)
    print("after======================================================================")
    print(predict_Y_int)
    predict_sentence = "".join(final_result)
    print(predict_sentence)


def test_simple_embedding_model(model, tk, sentences, max_length):
    sentences_int = tk.texts_to_sequences(sentences)
    sentences_int_pad = pad_sequences(sentences_int, padding='post', maxlen=max_length)
    predict_Y = model.predict(sentences_int_pad, verbose=1)
    predict_Y_int = get_dict_index_back(predict_Y)
    final_result = reverse_dict(tk, predict_Y_int)
    print("input======================================================================")
    print(sentences)
    print(sentences_int)
    print("after======================================================================")
    print(predict_Y_int)
    predict_sentence = "".join(final_result)
    print(predict_sentence)


def test_teacher_forcing_model():
    return


def test_attention_model():
    return


def test_beam_search_mode():
    return


def test_bi_LSTM_mode():
    return


sentences = [
    "growing upY"
]


# 1. 抽取SQuAD 2.0的问答数据
json_data = read_squad_examples("./input_data/train-v2.0.json", False)
abstarct_sentence(json_data)
clean_file()

# 2. 生成码表
tk = char2int()
print(tk.word_index)
char_table_size = len(tk.word_index) + 1
print(char_table_size)

# 3. 在抽取的文件中加入错误的噪音
np_array_X, np_array_Y = gen_X_y(tk)
# 确保行数一样
print(np_array_X.shape)
print(np_array_Y.shape)

# 4. padding
max_X_length = max(len(xi) for xi in np_array_X)
max_Y_length = max(len(yi) for yi in np_array_Y)
max_length = max(max_X_length, max_Y_length)
np_pad_X = pad_sequences(np_array_X, padding='post', maxlen=max_length)
np_pad_Y = pad_sequences(np_array_Y, padding='post', maxlen=max_length)

# one hot model
self_one_hot_model = train_wrong_one_hot_simple_char_model(np_pad_X, np_pad_Y, max_length, char_table_size)
test_one_hot_simple_model(self_one_hot_model, tk, sentences, max_length, char_table_size)

# embedding model
simple_embedding_model = train_simple_embedding_model(max_length, char_table_size, char_table_size, np_pad_X, np_pad_Y)
test_simple_embedding_model(simple_embedding_model, tk, sentences, max_length)

# teacher forcing model
teacher_forcing_model = train_teacher_forcing_model(max_length, char_table_size, char_table_size, np_pad_X, np_pad_Y)

# non teacher forcing model - 训练不动，考虑使用pre-train的向量试试
non_teacher_forcing_model = train_non_teacher_forcing_model(max_length, char_table_size, char_table_size, np_pad_X, np_pad_Y)