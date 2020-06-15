from tensorflow.keras.preprocessing.text import Tokenizer
import re
import gensim
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences
from numpy import *
import nltk

"""
需要下载某些需要使用的语料
"""
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

TAG_RE = re.compile(r'<[^>]+>')
default_max_row = 10000


def correct_and_wrong_embedding():
    """
    将word转化成int，生成一个word_to_int的dict
    演示fix_on_texts 和 gensim的使用方法
    """
    # fit_on_texts错误的用法
    tokenizer2 = Tokenizer(num_words=5000)
    tokenizer2.fit_on_texts([["I am god"], ["you are idiot"]])
    # fit_on_texts正确的用法
    tokenizer3 = Tokenizer(num_words=5000)
    tokenizer3.fit_on_texts([["I", "am", "god"], ["you", "are", "idiot"]])
    print(tokenizer2.word_index)
    print(tokenizer3.word_index)
    word_model2 = gensim.models.Word2Vec([["I am god"], ["you are idiot"]], min_count=1, size=5, window=5)
    word_model3 = gensim.models.Word2Vec([["I", "am", "god"], ["you", "are", "idiot"]], min_count=1, size=5, window=5)
    # word_model2.wv.save_word2vec_format("wrong_embedding")
    # word_model3.wv.save_word2vec_format("correct_embedding")

# print("==================test correct_and_wrong_embedding======================")
# correct_and_wrong_embedding()
# print("==================test correct_and_wrong_embedding======================")


def column_text_to_sentence_array(lines, use_stop_word=False):
    """
    英文句子分词作为gensim.models.Word2Vec函数的输入
    :param lines:  格式 ["I am luru", "you are ivy", "I cannot swim"]
    :param use_stop_word: 是否使用停用词
    :return: [['i', 'am', 'luru'], ['you', 'are', 'ivy'], ['i', 'can', 'not', 'swim']]
    """
    stopwords_en = set()
    if use_stop_word:
        stopwords_en = set(stopwords.words('english'))
    sentence_array = []
    word_num = 0
    word_set = set()
    for line in lines:
        temp = []
        for word in word_tokenize(line):
            if word not in stopwords_en:
                word_num += 1
                word_uniform = word.lower()
                word_set.add(word_uniform)
                temp.append(word_uniform)
        sentence_array.append(temp)
    print("total word number : {}".format(word_num))
    print("total uniq word : {}".format(len(word_set)))
    return sentence_array, word_num, word_set

# print("==================test column_text_to_sentence_array======================")
# sentence, num, word_set = column_text_to_sentence_array(["I am luru", "you are ivy", "I cannot swim"], False)
# print("split sentence into segment - ")
# print(sentence)
# print("word number - ")
# print(num)
# print("unique word - ")
# print(word_set)
# print("==================test column_text_to_sentence_array======================")


def remove_tags(text):
    return TAG_RE.sub('', text)


def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence


def generate_word_vector_with_gensim(data, embedding_size, window_size, min_count):
    cbow = gensim.models.Word2Vec(data, min_count=min_count, size=embedding_size, window=window_size)
    word_vector = cbow.wv
    # word to index dict
    return word_vector


print("==================test generate_word_vector_with_gensim======================")
data = [
    ["i", "am", "god"],
    ["you", "are", "idiot"]
]
wv = generate_word_vector_with_gensim(data, 5, 2, 1)
print("word to vector number : {}".format(len(wv.index2word)))
print("word to int - ")
for word in wv.vocab:
    print("{} - {}".format(word, wv[word]))
print("==================test generate_word_vector_with_gensim======================")


"""
这两个函数要配合使用，先根据出现的word生成一个word->int的映射，然后后续的句子按这个词典生成int数组。
常用于NLP任务中。
"""
def build_word_to_int_dict(dict_data):
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data)
    return tokenizer


def encoding_word_to_int(tokenizer, real_data):
    return tokenizer.texts_to_sequences(real_data)


print("==================test encoding_word_to_int======================")
dict_data = [
    ["i", "am", "god"],
    ["you", "are", "idiot"]
]
tokenizer = build_word_to_int_dict(dict_data)
for word, index in tokenizer.word_index.items():
    print("{} - {}".format(word, index))
test_data = [
    ["i", "am", "idiot"]
]
test_encoding = encoding_word_to_int(tokenizer, test_data)
print(test_encoding)
# 用0填充
test_encoding_padding = pad_sequences(test_encoding, padding='post', maxlen=10)
print(test_encoding_padding)
print("==================test encoding_word_to_int======================")


def build_int_to_vector_mapping(tokenizer, word_vector):
    """
    现在有两个映射
    1. gensim 的 word->vector 存在word_vector
    2. tokenizer 的 word->int 存在tokenizer
    矩阵中存放 int->vector的映射
    最终的映射是 ["I" "am" "a" "super" "star"] -> [1, 3, 4, 5, 7] -> [[0.1, 0.4], [0.21, 0.233] ...]
    在Embedding layer embedding matrix被传入
    index=0流出来用来padding
    """
    embedding_matrix = zeros((len(tokenizer.word_index) + 1, word_vector.vector_size))
    for word, index in tokenizer.word_index.items():
        try:
            embedding_vector = word_vector[word]
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
        except KeyError:
            print("word {} is OOV".format(word))
            pass
    return embedding_matrix

print("==================test build_int_to_vector_mapping======================")
for word in wv.vocab:
    print("{} - {}".format(word, wv[word]))
for word, index in tokenizer.word_index.items():
    print("{} - {}".format(word, index))
embedding_matrix = build_int_to_vector_mapping(tokenizer, wv)
for sentence in test_data:
    for word in sentence:
        print("word - {}".format(word))
        print("int - {}, vector from wv - {}".format(tokenizer.word_index[word], wv[word]))
        print("int - {}, vector from wv - {}".format(tokenizer.word_index[word], embedding_matrix[tokenizer.word_index[word]]))
print("==================test build_int_to_vector_mapping======================")


def char2int(all_text):
    tokenizer = Tokenizer(num_words=None, char_level=True, oov_token="UNK", lower=False)
    tokenizer.fit_on_texts(all_text)
    return tokenizer


print("==================test char2int======================")
wrong_char_data_format = [
    ["i", "am", "god"],
    ["you", "are", "idiot"]
]
correct_char_data_format = [
    "i am god",
    "you are idiot"
]
print("wrong char data format output - ")
tokenizer = char2int(wrong_char_data_format)
for word, index in tokenizer.word_index.items():
    print("{} - {}".format(word, index))
print("correct char data format output - ")
tokenizer = char2int(correct_char_data_format)
for word, index in tokenizer.word_index.items():
    print("{} - {}".format(word, index))
print("==================test char2int======================")