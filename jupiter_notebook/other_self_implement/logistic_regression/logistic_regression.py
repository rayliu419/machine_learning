# encoding=utf-8

import math
import random
import time

import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

MAX_VALUE = 255

# 逻辑回归定义分类为1的概率 P(x = 1) = exp{wx} / 1 + exp{wx}，所以相应的P(x = 0) = 1 / 1 + exp{wx}。
# 令P(x = 1) = f(z) = exp{z} / 1 + exp{z}，其中z = wx
# 逻辑回归对于单个sample的损失函数： loss = y * ln(yhat) + (1 - y)ln(1 - yhat)，
# 当y = 1时，若yhat -> 1，loss -> 0; y = 1时，若yhat -> 0，loss -> 负无穷。
# 对于f(z)，有f'(z) = f(z) * (1 - f(z))
# loss对某个w求导，有 loss'(wi) = (y - f(z))xi (这里需要做loss的导数推导，并利用到上一步的等式)。

class LogisticRegression(object):
    def __init__(self):
        self.learning_step = 0.001
        self.max_iteration = 5000

    def predict_single(self, x):
        try:
            wx = np.dot(self.w, x)
            exp_wx = math.exp(wx)
            predict1 = exp_wx / (1 + exp_wx)
            predict0 = 1 / (1 + exp_wx)
            if predict1 > predict0:
                return 1
            else:
                return 0
        except OverflowError:
            print x
            print self.w

    def train(self, features, labels):
        self.w = np.zeros(len(features[0]) + 1)
        time = 0
        while True:
            if (time > self.max_iteration):
                break
            time += 1
            index = random.randint(0, len(labels) - 1)
            cur_features = list(features[index])
            cur_features.append(1.0)
            y = labels[index]
            x = np.array(cur_features)
            # feature归一化到[0,1]，注意不归一化，有的时候会出现math.RangeError
            x = x / MAX_VALUE
            if y == self.predict_single(x):
                # 预测正确
                continue
            wx = np.dot(self.w, x)
            exp_wx = math.exp(wx)
            for i in xrange(len(self.w)):
                # 如果y = 1, 而yhat -> 0，即f(z) -> 0，说明预测有误，为了提高预测概率，要提高f(z)的值。
                # P(x = 1) = exp{wx} / 1 + exp{wx} = 1 - 1 / (1 + exp(wx))，要让P(X = 1)提高，下一轮需要提高wx的值。
                # 如果x[i] > 0，则+号使得w[i]增加了，则下一轮的wx增加，P(x = 1)提高，不论w[x]以前是正负，都是这个结论。
                # 如果x[i] < 0，则+号使得w[i]减小了，w[i]*x[i]减小了，不论w[x]以前是正负，都是这个结论。
                # 综上，+号更新是正确的。同理，可以假设y = 0来推导，应该同样成立。
                yhat = float(exp_wx) / float(1 + exp_wx)
                self.w[i] += self.learning_step * (y - yhat) * x[i]

    def predict_batch(self, features):
        labels = []
        for feature in features:
            cur_features = list(feature)
            cur_features.append(1.0)
            x = np.array(cur_features)
            x = x / 255
            labels.append(self.predict_single(x))
        return labels

if __name__ == "__main__":
    print 'Start read data'
    time_1 = time.time()
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values
    imgs = data[0:, 1:]
    labels = data[:, 0]
    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33,
                                                                                random_state=23323)
    time_2 = time.time()
    print 'read data cost ', time_2 - time_1, ' second', '\n'
    print 'Start training'
    lr = LogisticRegression()
    lr.train(train_features, train_labels)
    time_3 = time.time()
    print 'training cost ', time_3 - time_2, ' second', '\n'
    print 'Start predicting'
    test_predict = lr.predict_batch(test_features)
    time_4 = time.time()
    print 'predicting cost ', time_4 - time_3, ' second', '\n'
    score = accuracy_score(test_labels, test_predict)
    print "The accruacy socre is ", score
