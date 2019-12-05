# 是用预训练的word vector来构造DN

import pandas as pd
from numpy import zeros
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
import gensim
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Activation, Dropout, Dense, LSTM
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt


stopwords_en = set()
TAG_RE = re.compile(r'<[^>]+>')
max_row = 100


def column_text_to_sentence_array(lines):
    sentence_array = []
    word_num = 0
    word_set = set()
    for line in lines:
        temp = []
        for word in word_tokenize(line):
            if word not in stopwords_en:
                word_num += 1
                word_set.add(word)
                temp.append(word.lower())
        sentence_array.append(temp)
    uniq_word_num = len(word_set)
    print("total word : {}".format(word_num))
    print("unique word : {}".format(uniq_word_num))
    return sentence_array


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


def epoch_performance(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def simple_model(dict_size, embedding_matrix, maxlen, X_train, X_test, Y_train, Y_test):
    model = Sequential()
    """
    All that the Embedding layer does is to map the integer inputs to the vectors found at the
    corresponding index in the embedding matrix,
    i.e. the sequence [1, 2] would be converted to [embeddings[1], embeddings[2]]
    """
    embedding_layer = Embedding(dict_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    model.add(embedding_layer)
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    plot_model(model, to_file='simple_model.png', show_shapes=True)
    # if isinstance(Y, pd.core.series.Series):
    #     Y_train = Y.to_numpy()
    # if isinstance(Y_test, pd.core.series.Series):
    #     Y_test = Y_test.to_numpy()
    history = model.fit(X_train, Y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.2)
    epoch_performance(history)
    score, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print("Test Score:", score)
    print("Test Accuracy:", accuracy)
    return model


def LSTM_many_to_one_model(dict_size, embedding_matrix, maxlen, X_train, X_test, Y_train, Y_test):
    model2 = Sequential()
    embedding_layer = Embedding(dict_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
    model2.add(embedding_layer)
    model2.add(LSTM(128))
    model2.add(Dense(1, activation='sigmoid'))
    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    plot_model(model2, to_file='LSTM_many_to_one_model.png', show_shapes=True)
    history2 = model2.fit(X_train, Y_train, batch_size=128, epochs=3, verbose=1, validation_split=0.2)
    epoch_performance(history2)
    score2, accuracy2 = model2.evaluate(X_test, Y_test, verbose=0)
    print("Test Score:", score2)
    print("Test Accuracy:", accuracy2)
    return model2


def predict_single(review, model):
    review = tokenizer.texts_to_sequences(review)
    flat_list = []
    for sublist in review:
        for item in sublist:
            flat_list.append(item)
    flat_list = [flat_list]
    review = pad_sequences(flat_list, padding='post', maxlen=maxlen)
    print(model.predict(review))


def build_int_to_vector_mapping(tokenizer, word_vector):
    """
    现在有两个映射
    1. gensim 的 word->vector 存在word_vector
    2. tokenizer 的 word->int 存在tokenizer
    矩阵中存放 int->vector的映射
    最终的映射是 ["I" "am" "a" "super" "star"] -> [1, 3, 4, 5, 7] -> [[0.1, 0.4], [0.21, 0.233] ...]
    在Embedding layer embedding matrix被传入
    """
    embedding_matrix = zeros((vocab_size, 100))
    for word, index in tokenizer.word_index.items():
        try:
            embedding_vector = word_vector[word]
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector
        except KeyError:
            print("word {} is OOV".format(word))
            pass
    return embedding_matrix


# 1. 读取数据
movie_reviews = pd.read_csv("./input_data/IMDB_Dataset.csv")
movie_reviews.info()
movie_reviews = movie_reviews[0:max_row]
# 2. 预处理数据
movie_reviews["review"] = movie_reviews["review"].map(lambda x: preprocess_text(x))
movie_reviews["sentiment"] = movie_reviews["sentiment"].map(lambda x: 1 if x=="positive" else 0)
movie_reviews.head()
X = movie_reviews["review"]
Y = movie_reviews["sentiment"]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# 3. 使用gensim训练词向量
data = column_text_to_sentence_array(movie_reviews["review"])
model_COBW = gensim.models.Word2Vec(data, min_count=1, size=100, window=5)
word_vector = model_COBW.wv
# like word -> vector dict
print("total unique words : {}".format(len(word_vector.index2word)))

# 4. 将word数组转换为int数组，只保存5000个频率最高的映射？
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(data)
"""
tokenizer.fit_on_texts(X_train) 这里不能这么写，要用data来fit_on_texts。
原因是nltk.tokenize的word_tokenize和keras.preprocessing.fit_on_texts对于有些词不一样，例如cannot, nltk处理成了can和not，但是
fit_on_text的时候将 cannot当成一个词了
"""

print(len(tokenizer.word_index))
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index) + 1

# 5. 填充成一样的长度
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
print(len(X_train[0]))
print(len(X_train[1]))

# 6. 获取输入层的weights
embedding_mapping = build_int_to_vector_mapping(tokenizer, word_vector)

# 6. 训练和测试
sample_review = X[57]
print(sample_review)
# simple model
model1 = simple_model(vocab_size, embedding_mapping, maxlen, X_train, X_test, Y_train, Y_test)
predict_single(sample_review, model1)

model2 = simple_model(vocab_size, embedding_mapping, maxlen, X_train, X_test, Y_train, Y_test)
predict_single(sample_review, model2)