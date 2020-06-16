import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import english_preprocess
import gensim

"""
不需要在每个子任务中都要读取一遍
"""
def prepare_movie_review_for_task(line_num=100, maxlen=100, embedding_size=10):
    # 1. 读取数据
    print("#1 load data")
    movie_reviews = pd.read_csv("/Users/luru/workspace/github/machine_learning/jupiter_notebook/common/input_data/IMDB_Dataset.csv")
    # movie_reviews.info()
    movie_reviews = movie_reviews[0:line_num]
    # 2. 预处理数据
    print("#2 clean data")
    movie_reviews["review"] = movie_reviews["review"].map(lambda x: english_preprocess.preprocess_text(x))
    movie_reviews["sentiment"] = movie_reviews["sentiment"].map(lambda x: 1 if x=="positive" else 0)
    movie_reviews.head()
    X = movie_reviews["review"]
    Y = movie_reviews["sentiment"]
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)
    data, word_num, word_set = english_preprocess.column_text_to_sentence_array(movie_reviews["review"], False)
    # cannot -> can, not
    # data[13]
    print("total word number : {}".format(word_num))
    print("total uniq word : {}".format(len(word_set)))
    print("#3 training with gensim")
    # 3. 使用gensim训练词向量
    cbow = gensim.models.Word2Vec(data, min_count=1, size=embedding_size, window=5)
    word_vector = cbow.wv
    # like word -> vector dict
    print("word to vecotr number : {}".format(len(word_vector.index2word)))
    # 4. 将word数组转换为int数组，只保存5000个频率最高的映射？
    print("#4 turn word to int")
    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(data)
    """
    correct_and_wrong()
    tokenizer.fit_on_texts(X_train) 这里不能这么写，要用data来fit_on_texts。
    原因是nltk.tokenize的word_tokenize和keras.preprocessing.fit_on_texts对于有些词不一样，例如cannot, nltk处理成了can和not，但是
    fit_on_text的时候将 cannot当成一个词了
    """
    print("word to int number : {}".format(len(tokenizer.word_index)))
    X_train = tokenizer.texts_to_sequences(X_train)
    X_test = tokenizer.texts_to_sequences(X_test)
    # 5. 填充成一样的长度，如果越过就是不填充
    print("#5 padding")
    X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
    X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
    # 6. 获取输入层的weights
    print("#6 calculate embedding matrix")
    embedding_mapping = english_preprocess.build_int_to_vector_mapping(tokenizer, word_vector)
    print("#7 print sample information")
    # 不能这样打印，因为X_train变为了一个打乱顺序的Series
    # sample_review, sample_sentiment = X_train[1], Y_train[1]
    # print(sample_review)
    # print(sample_sentiment)
    return tokenizer, embedding_mapping, X, Y, X_train, X_test, Y_train, Y_test


if __name__ == "__main__":
    prepare_movie_review_for_task()