import os
from io import open
import unicodedata
import re
import random

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
good_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re "
)


class Lang:
    """
    自定义了一个word编码成int的类，注意是word级，不是char级。
    """
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.word2count = {}
        self.n_words = 2  # Count SOS and EOS

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

"""
处理字符串之类的
"""
# Turn a Unicode string to plain ASCII
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


"""
过滤有一些pair
"""
def filter_pair(p):
    """
    这里主要是过滤掉训练集的一部分，主要是为了速度快点。
    :param p:
    :return:
    """
    return len(p[0].split(' ')) < MAX_LENGTH and len(p[1].split(' ')) < MAX_LENGTH and p[1].startswith(good_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def read_langs(lang1, lang2, reverse=False):
    print("Reading lines...")
    data_file = os.path.dirname(__file__) + "/input_data/%s-%s.txt"
    # Read the file and split into lines
    lines = open(data_file % (lang1, lang2), encoding='utf-8').read().strip().split('\n')
    # 文件的格式是[Shut up!	Tais-toi !]，英文和法文用"\t"分割。
    pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]
    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)
    return input_lang, output_lang, pairs


def prepare_data(lang1_name, lang2_name, reverse=False):
    input_lang, output_lang, pairs = read_langs(lang1_name, lang2_name, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filter_pairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Indexing words...")
    for pair in pairs:
        input_lang.index_words(pair[0])
        output_lang.index_words(pair[1])
    print(random.choice(pairs))
    print(len(input_lang.index2word))
    print(len(output_lang.index2word))
    return input_lang, output_lang, pairs


def prepare_translation_raw_data_for_task():
    return prepare_data('eng', 'fra', True)


if __name__ == "__main__":
    prepare_translation_raw_data_for_task()
