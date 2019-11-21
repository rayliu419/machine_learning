# 从SQuAD 2.0出去文本数据，加入随机噪音生成训练集。
# 比较不同的model的纠错效果。
# 英文数据

import json
import tensorflow as tf
import math
import random
import numpy as np
from numpy.random import choice as random_choice, randint as random_randint
import sys
from keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Masking
from keras.preprocessing.text import Tokenizer

english_text_file = "./input_data/abstract_english_text_file"
clean_english_text_file = "./input_data/clean_english_text_file"
before_add_error_file = "./input_data/before_add_error"
after_add_error_file = "./input_data/after_add_error"
change_index_file = "./input_data/change_index_file"
MAX_LINE_NUMBER = 20000

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
    #     np_array_X = np.array(X_encoding)
    #     np_array_Y = np.array(Y_encoding)
    np_array_X = np.array([np.array(xi) for xi in X_encoding])
    np_array_Y = np.array([np.array(yi) for yi in Y_encoding])
    return np_array_X, np_array_Y


# DL要求将字符转成int作为输入
def char2int(file=clean_english_text_file):
    fp = open(file, "r")
    all_text = fp.read()
    fp.close()
    tk = Tokenizer(num_words=None, char_level=True, oov_token="UNK", lower=False)
    tk.fit_on_texts(all_text)
    return tk

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


def prepare_test_sentences(tk, sentences, max_length, char_table_size):
    sentences_int = tk.texts_to_sequences(sentences)
    print(sentences_int)
    sentences_int_pad = pad_sequences(sentences_int, padding='post', maxlen=max_length)
    sentence_num = len(sentences)
    sentences_int_pad_one_hot = np.empty(shape=(sentence_num, max_length, char_table_size))
    for i in range(0, sentence_num):
        sentences_int_pad_one_hot[i] = to_categorical(sentences_int_pad[i], num_classes=char_table_size)
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


# 使用自己编码的one_hot来作为输入和输出
def train_one_hot_simple_char_model(np_pad_X, np_pad_Y, pad_max_length, char_table_size):
    # X, Y转化为one-hot
    rows = np_pad_Y.shape[0]
    np_pad_one_hot_Y = np.empty(shape=(rows, pad_max_length, char_table_size))
    np_pad_one_hot_X = np.empty(shape=(rows, pad_max_length, char_table_size))
    for index in range(0, rows):
        cur_one_hot_x = to_categorical(np_pad_X[index], num_classes=char_table_size)
        cur_one_hot_y = to_categorical(np_pad_Y[index], num_classes=char_table_size)
        np_pad_one_hot_X[index] = cur_one_hot_x
        np_pad_one_hot_Y[index] = cur_one_hot_y
    print(type(np_pad_one_hot_X))
    print(np_pad_one_hot_X.shape)
    print('Build model...')
    model = Sequential()
    # LSTM作为首层才需要设置input_shape参数。
    model.add(LSTM(20, input_shape=(pad_max_length, char_table_size), return_sequences=True))
    model.add(LSTM(20, return_sequences=True))
    model.add(Dense(char_table_size, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    print("start training")
    model.fit(np_pad_one_hot_X, np_pad_one_hot_Y, epochs=1, verbose=1)
    return model


# 这个是many to many，应该用encoder-decoder方法
def train_simple_embedding_model(max_len, feature_num, output, np_pad_X, np_pad_Y):
    char_table_size = feature_num
    rows = np_pad_Y.shape[0]
    np_pad_one_hot_Y = np.empty(shape=(rows, max_len, char_table_size))
    for index in range(0, rows):
        cur_one_hot_y = to_categorical(np_pad_Y[index], num_classes=char_table_size)
        np_pad_one_hot_Y[index] = cur_one_hot_y
    print('Build model...')
    model = Sequential()
    model.add(Embedding(input_dim=feature_num, input_length=max_len, output_dim=10, mask_zero=True))
    model.add(LSTM(20, return_sequences=True))
    model.add(LSTM(20, return_sequences=True))
    model.add(Dense(output, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    model.fit(np_pad_X, np_pad_one_hot_Y, epochs=10, verbose=1, batch_size=64)
    return model


sentences = [
    "growing upY"
]


def test_one_hot_simple_mode(model, tk, sentences, max_length, char_table_size):
    test_np_pad_one_hot_X = prepare_test_sentences(tk, sentences, max_length, char_table_size)
    # print(test_np_pad_one_hot_X)
    predict_Y = model.predict(test_np_pad_one_hot_X, verbose=1)
    print(predict_Y.shape)
    predict_Y_int = get_dict_index_back(predict_Y)
    print(predict_Y_int)
    final_result = reverse_dict(tk, predict_Y_int)
    print(final_result)


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


# # one hot model
# self_one_hot_model = train_one_hot_simple_char_model(np_pad_X, np_pad_Y, max_length, char_table_size)
# test_one_hot_simple_mode(self_one_hot_model, tk, sentences, max_length, char_table_size)

# embedding model
simple_embedding_model = train_simple_embedding_model(max_length, char_table_size, char_table_size, np_pad_X, np_pad_Y)
test_simple_embedding_model(simple_embedding_model, tk, sentences, max_length)