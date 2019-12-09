# 是用预训练的word vector来构造DN

import pandas as pd
from numpy import zeros
import re
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
import gensim
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Flatten
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Activation, Dropout, Dense, LSTM, Bidirectional
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras import regularizers
import nltk
nltk.download('punkt')


stopwords_en = set()
TAG_RE = re.compile(r'<[^>]+>')
max_row = 100

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type, file_name):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.savefig(file_name)


def correct_and_wrong():
    # fit_on_texts错误的用法
    tokenizer2 = Tokenizer(num_words=5000)
    tokenizer2.fit_on_texts([["I am god"], ["you are idiot"]])
    print(tokenizer2.word_index)
    # fit_on_texts正确的用法
    tokenizer3 = Tokenizer(num_words=5000)
    tokenizer3.fit_on_texts([["I", "am", "god"], ["you", "are", "idiot"]])
    print(tokenizer3.word_index)
    # gensim.models.Word2Vec的错误用法
    word_model2 = gensim.models.Word2Vec([["I am god"], ["you are idiot"]], min_count=1, size=5, window=5)
    word_model2.wv.save_word2vec_format("wrong_embedding")
    # gensim.models.Word2Vec的正确用法
    word_model3 = gensim.models.Word2Vec([["I", "am", "god"], ["you", "are", "idiot"]], min_count=1, size=5, window=5)
    word_model3.wv.save_word2vec_format("correct_embedding")


def column_text_to_sentence_array(lines):
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
    uniq_word_num = len(word_set)
    return sentence_array, word_num, word_set


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
data, word_num, word_set = column_text_to_sentence_array(movie_reviews["review"])
# cannot -> can, not
# data[13]
print("total word number : {}".format(word_num))
print("total uniq word : {}".format(len(word_set)))
model_COBW = gensim.models.Word2Vec(data, min_count=1, size=100, window=5)
word_vector = model_COBW.wv
# like word -> vector dict
print("word to vecotr number : {}".format(len(word_vector.index2word)))

# 4. 将word数组转换为int数组，只保存5000个频率最高的映射？
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
vocab_size = len(tokenizer.word_index) + 1


# 5. 填充成一样的长度
maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# 6. 获取输入层的weights
embedding_mapping = build_int_to_vector_mapping(tokenizer, word_vector)

# 6. 训练和测试
sample_review = X[57]
# print(sample_review)

def current_time():
    now = datetime.now()
    dt_string = now.strftime("%Y_%m_%d_%H_%M_%S")
    return dt_string

def epoch_performance(history, model_name):
    # plt.plot(history.history['acc'])
    # plt.plot(history.history['val_acc'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['train','validation'], loc='upper left')
    # plt.savefig(prefix+"_acc")
    # plt.close()
    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss'])
    # plt.title('model loss')
    # plt.ylabel('loss')
    # plt.xlabel('epoch')
    # plt.legend(['train', 'validation'], loc='upper left')
    # plt.savefig(prefix+"_loss")
    # plt.show()
    # plt.close()
    print("model name : {}".format(model_name))
    for i in history.epoch:
        print("epoch {}".format(i))
        print("train acc : {} validate acc : {}".format(history.history["acc"][i], history.history["val_acc"][i]))
        print("train loss : {} validate loss : {}".format(history.history["loss"][i], history.history["val_loss"][i]))


def simple_model(model_name, dict_size, embedding_matrix, maxlen, X_train, X_test, Y_train, Y_test):
    model = Sequential()
    """
    All that the Embedding layer does is to map the integer inputs to the vectors found at the
    corresponding index in the embedding matrix,
    i.e. the sequence [1, 2] would be converted to [embeddings[1], embeddings[2]]
    这种模型over fitting比较严重。
    """
    model_info_file_prefix = "./model_info/{}_{}_".format(model_name, current_time())
    model_structure_file = model_info_file_prefix + "model_structure.png"
    embedding_layer = Embedding(dict_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
    model.add(embedding_layer)
    model.add(Flatten())
    # 光通过这个解决不了overfitting的问题
    # model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.l2(0.01)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    plot_model(model, to_file=model_structure_file , show_shapes=True)
    if isinstance(Y, pd.core.series.Series):
        Y_train = Y_train.to_numpy()
    if isinstance(Y_test, pd.core.series.Series):
        Y_test = Y_test.to_numpy()
    history_plot = LossHistory()
    history = model.fit(X_train, Y_train, batch_size=64, epochs=10, verbose=1, validation_split=0.2, callbacks=[history_plot])
    epoch_performance(history, model_name)
    history_plot.loss_plot('epoch', model_info_file_prefix+"epoch.png")
    score, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    print("test set result")
    print("test score:", score)
    print("test acc:", accuracy)
    return model

print("start training simple model")
model1 = simple_model("simple_model" ,vocab_size, embedding_mapping, maxlen, X_train, X_test, Y_train, Y_test)