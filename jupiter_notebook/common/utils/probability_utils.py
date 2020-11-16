"""
Util class to help calculate probability.

"""
import torch
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer


def sum_equal(tensor1, tensor2):
    """
    比较预测的类别正确率
    输入都是pytroch.tensor类型
    :param tensor1: [[1], [0], [2]]...
    :param tensor2: [[2], [1], [3]]...
    :return:
    """
    ndarray1 = tensor1.numpy()
    ndarray2 = tensor2.numpy()
    return (ndarray1 == ndarray2).astype(int).sum()


def top_k_values_and_indexes(prob_array, k=1):
    """
    Use torch.topk to get top values and corresponding indexes.
    work for both list/ndarray.
    :param prob_array:
    :param k:
    :return:
    index是按大到小返回的。
    返回list类型
    """
    print(type(prob_array))
    temp = torch.FloatTensor(prob_array)
    # topk 返回tensor类型
    values, indexes = temp.topk(k)
    return values.numpy().tolist(), indexes.numpy().tolist()


def map_indexes_to_word(index_array, tokenizer, prob_array):
    """
    :param index_array:
    :param tokenizer:
    :param prob_array:
    :return:
    """
    result = []
    for index in index_array:
        word = tokenizer.index_word[index]
        probability = prob_array[index]
        result.append((word, probability))
    return result


if __name__ == "__main__":
    print(top_k_values_and_indexes([0.2, 0.3, 0.5], 2))
    print(top_k_values_and_indexes(np.array([0.2, 0.3, 0.5]), 2))

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts([["I", "am", "god"], ["you", "are", "idiot"]])
    prob_array = [0.1, 0.15, 0.2, 0.25, 0.3, 0]
    top_values, top_indexes = top_k_values_and_indexes(prob_array, 2)
    print(top_indexes)
    print(tokenizer.index_word)
    print(map_indexes_to_word(top_indexes, tokenizer, prob_array))


