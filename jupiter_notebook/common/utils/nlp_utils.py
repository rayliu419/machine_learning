import numpy as np
from nltk.util import ngrams


class WordDict():
    def __init__(self):
        self.word2int_dict = {}
        self.int2word_dict = {}
        # 0 for padding, 1 for OOV.
        self.total_token = 2
        self.index = 2
        self.total_uniq_word = 0

    def add_word(self, word):
        if word in self.word2int_dict:
            pass
        else:
            self.word2int_dict[word] = self.index
            self.int2word_dict[self.index] = word
            self.index += 1
            self.total_uniq_word += 1
            self.total_token += 1

    def encode_tokenized_sentence_array(self, tokenized_sentence_array):
        if self.total_uniq_word == 0:
            print("haven't init word dict")
            raise RuntimeError
        else:
            encode_table = []
            for sentence in tokenized_sentence_array:
                cur = self.encode_tokenized_sentence(sentence)
                encode_table.append(cur)
        return encode_table

    def encode_tokenized_sentence(self, tokenized_sentence):
        cur = []
        for token in tokenized_sentence:
            index = 1
            if token in self.word2int_dict:
                index = self.word2int_dict[token]
            cur.append(index)
        return cur

    def word_to_index(self, word):
        index = 1
        if word in self.word2int_dict:
            index = self.word2int_dict[word]
        return index

    def index_to_word(self, index):
        if index in self.int2word_dict:
            return self.int2word_dict[index]
        else:
            raise RuntimeError


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


def merge_to_single_line(text):
    single_line = ''
    for line in text:
        line = line.strip("\n")
        single_line += line
    return single_line


def build_word_to_int_dict(sentence_array):
    word_dict = WordDict()
    for sentence in sentence_array:
        for word in sentence:
            word_dict.add_word(word)
    return word_dict


if __name__ == "__main__":
    print(generate_ngram_single(["a", "b", "c"]))
    print(generate_ngram_multiple([["a", "b", "c"], ["x", "y"]]))
