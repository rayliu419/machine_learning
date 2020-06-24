import numpy as np
from nltk.util import ngrams


def generate_ngram_multiple(encoded, length=1):
    """
    构造ngram，
    :param encoded:
    :param length:
    :return:
    """
    sequences = list()
    for array in encoded:
        n_grams = ngrams(array, length + 1)
        for ngram in n_grams:
            sequences.append(ngram)
    print('Total Sequences: %d' % len(sequences))
    sequences = np.array(sequences)
    print(type(sequences))
    return sequences


def generate_ngram_single(x_array, length=1):
    """
    构造ngram，
    :param encoded:
    :param length:
    :return:
    """
    sequences = list()
    n_grams = ngrams(x_array, length + 1)
    for ngram in n_grams:
        sequences.append(ngram)
    print('Total Sequences: %d' % len(sequences))
    sequences = np.array(sequences)
    print(type(sequences))
    return sequences


if __name__ == "__main__":
    print(generate_ngram_single(["a", "b", "c"]))
    print(generate_ngram_multiple([["a", "b", "c"], ["x", "y"]]))
