import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import sys
sys.path.append("..")
from common.lng_processor.english_preprocess import *
from common.lng_processor.eng_process import *
import gensim
import os


def prepare_movie_review_raw(line_num=1000):
    """
    返回原始数据集。后来有一个新的数据集。
    :param line_num:
    :return:
    """
    data_file = os.path.dirname(__file__) + "/input_data/IMDB_Dataset.csv"
    df = pd.read_csv(data_file)
    df.info()
    df = df[0:line_num]
    df.rename(columns={"review": "text", "sentiment": "label"}, inplace=True)
    df["text"] = df['text'].apply(lambda x: tokenize(x))
    df.head()
    return df


def prepare_movie_review_for_task(line_num=100, maxlen=100, embedding_size=10):
    """
    不需要在每个任务都自己读取数据
    :param line_num:
    :param maxlen:
    :param embedding_size:
    :return:
        tokenizer -
        embedding_mapping -
        X, Y - 原始数据
        X_train, X_test, Y_train, Y_test - ndarray
    所有的load数据的函数都应该返回ndarry的格式，底层的keras或者pytorch在上层再自己转。
    """
    # 1. 读取数据
    print("#1 load data")
    data_file = os.path.dirname(__file__) + "/input_data/IMDB_Dataset.csv"
    movie_reviews = pd.read_csv(data_file)
    # movie_reviews.info()
    movie_reviews = movie_reviews[0:line_num]
    # 2. 预处理数据
    print("#2 clean and processing data")
    movie_reviews["review"] = movie_reviews["review"].map(lambda x: preprocess_text(x))
    movie_reviews["sentiment"] = movie_reviews["sentiment"].map(lambda x: 1 if x=="positive" else 0)
    movie_reviews.head()
    X = movie_reviews["review"]
    Y = movie_reviews["sentiment"]
    # cannot -> can, not
    # data[13]
    data, word_num, word_set = column_text_to_sentence_array(X, False)
    print("#3 training with gensim")
    # 3. 使用gensim训练词向量
    cbow = gensim.models.Word2Vec(data, min_count=1, size=embedding_size, window=5)
    word_vector = cbow.wv
    print("word to vecotr number : {}".format(len(word_vector.index2word)))
    # 4. 将word数组转换为int数组，只保存5000个频率最高的映射？
    print("#4 turn word to int")
    tokenizer = Tokenizer(num_words=5000)
    """
   tokenizer.fit_on_texts(X_train) 这里不能这么写，要用data来fit_on_texts。
   原因是nltk.tokenize的word_tokenize和keras.preprocessing.fit_on_texts对于有些词不一样，例如cannot, nltk处理成了can和   not，但是fit_on_text的时候将cannot当成一个词了。会导致build_int_to_vector_mapping中，tokenizer有cannot，word_vector
   没有cannot(因为column_text_to_sentence_array将cannot分词了)
   """
    tokenizer.fit_on_texts(data)
    print("word to int number : {}".format(len(tokenizer.word_index)))
    X_int = tokenizer.texts_to_sequences(data)
    # 5. 填充成一样的长度，如果越过就是不填充
    print("#5 padding")
    X_int_padding = pad_sequences(X_int, padding='post', maxlen=maxlen)
    # 6. 获取输入层的weights
    print("#6 calculate embedding matrix")
    embedding_mapping = build_int_to_vector_mapping(tokenizer, word_vector)
    print("#7 print sample information")
    X_train_np, X_test_np, Y_train, Y_test = train_test_split(X_int_padding, Y, test_size=0.20, random_state=42)
    Y_train_np = Y_train.to_numpy()
    Y_test_np = Y_test.to_numpy()
    print(X_train_np[0])
    print(Y_train_np[0])
    return tokenizer, embedding_mapping, X, Y, X_train_np, X_test_np, Y_train_np, Y_test_np


if __name__ == "__main__":
    prepare_movie_review_for_task()
