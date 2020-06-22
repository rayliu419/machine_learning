import os
import string
import unicodedata
import glob
import torch
import numpy as np
import english_preprocess


def unicode2Ascii(s, all_letters):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def readLines(filename, all_letters):
    """
    read lines and parse unicode to ascii
    :param filename:
    :param all_letters:
    :return:
    """
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode2Ascii(line, all_letters) for line in lines]


# Find letter index from all_letters, e.g. "a" = 0
def letterToIndex(letter, all_letters):
    return all_letters.find(letter)


def lineEncoding(line, n_letters):
    int_array = np.zeros((len(line), n_letters))
    for li, letter in enumerate(line):
        int_array[li][letterToIndex(letter, n_letters)] = 1
    return int_array


def prepare_name2countries_data_for_task():
    """
    :return:
        X - 原始的名字
        Y - 原始的国家
        X_int_encoding - char level编码后的int
        Y_int_encoding - 编码后的国家int
        n_letters - char字符的总数
        country_num - 国家的个数即标签的个数
        char_tokenizer - char的分词符
    """
    all_letters = string.ascii_letters + " .,;'"
    all_countries = []
    # Build the category_lines dictionary, a list of names per language
    char_tokenizer = english_preprocess.char2int(all_letters)
    data_dir = os.path.dirname(__file__) + "/input_data/names/*.txt"
    X = []
    Y = []
    X_int_encoding = []
    Y_int_encoding = []
    cur_country_int = 0
    int2country = dict()
    for filename in glob.glob(data_dir):
        country = os.path.splitext(os.path.basename(filename))[0]
        all_countries.append(country)
        lines = readLines(filename, all_letters)
        X.extend(lines)
        Y.extend([country] * len(lines))
        # 增加了一个维度
        cur_encoding = char_tokenizer.texts_to_sequences(lines)
        X_int_encoding.extend(cur_encoding)
        Y_int_encoding.extend([cur_country_int] * len(lines))
        int2country[cur_country_int] = country
        cur_country_int += 1
    country_num = len(all_countries)
    # n_letters = len(all_letters)
    n_letters = len(char_tokenizer.index_word) + 1
    print("country_num - {}, n_letters - {}".format(country_num, n_letters))
    # 由于名字的长短不一致，好像不能直接转成ndarray
    # X_np = np.array(X_encoding)
    # Y_np = np.array(Y_encoding)
    # return X, Y, X_np, Y_np
    return char_tokenizer, n_letters, country_num, X, Y, X_int_encoding, Y_int_encoding, int2country


if __name__ == "__main__":
    prepare_name2countries_data_for_task()