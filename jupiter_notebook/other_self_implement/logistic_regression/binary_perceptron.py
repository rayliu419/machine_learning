# encoding=utf-8
import pandas as pd
import numpy as np
import random
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class Perceptron(object):
    def __init__(self):
        self.learning_step = .001
        self.max_iteration = 20000

    def predict_single(self, x):
        output = np.dot(self.w, np.array(x))
        return int(output > 0)

    def train(self, features, labels):
        # 权重初始化，包括偏置
        self.w = np.random.normal(0, 0.1, len(features[0]) + 1)
        time = 0
        correct_count = 0
        while True:
            if (time > self.max_iteration):
                break
            time += 1
            # 随机挑选一个sample
            index = random.randint(0, len(labels) - 1)
            cur_features = list(features[index])
            # 偏置上的总是x总是1
            cur_features.append(1.0)
            y = 2 * labels[index] - 1
            # 转换为numpy array以执行np.dot操作
            x = np.array(cur_features)
            output = np.dot(self.w, x)
            if (output > 0):
                yhat = 1
            else:
                yhat = -1
            if yhat * y > 0:
                correct_count += 1
                continue
            else:
                for i in xrange(len(self.w)):
                    # 感知器算法中，loss函数是无法对w求导的，或者说无法表达成函数的形式。
                    # 在感知器算法中，如果wx > 0, 则yhat = 1, 反之yhat = -1
                    # 如果yhat = 1, y = 1或者yhat =-1, y = -1, 则预测正确，不用调整权重。
                    # 如果yaht = 1, y = -1, 代表新的权重需要降低wx的值，假设x[i] > 0, y - yhat < 0, 此时learning_step * (y - yhat) * x[i] < 0, 所以w[i]会减小，wx也会减小，满足优化的方向。
                    # 同理可以分析x[i] < 0的情况。
                    # 对于一个训练例子更新所有的权重
                    self.w[i] += self.learning_step * (y - yhat) * x[i]

    def predict_batch(self, features):
        labels = []
        for feature in features:
            x = list(feature)
            x.append(1)
            labels.append(self.predict_single(x))
        return labels

if __name__ == '__main__':
    print 'Start read data'
    time_1 = time.time()
    raw_data = pd.read_csv('../data/train_binary.csv', header=0)
    data = raw_data.values
    imgs = data[0:, 1:]
    labels = data[:, 0]
    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(
        imgs, labels, test_size=0.33, random_state=23323)
    time_2 = time.time()
    print 'read data cost ', time_2 - time_1, ' second', '\n'
    print 'Start training'
    p = Perceptron()
    p.train(train_features, train_labels)
    time_3 = time.time()
    print 'training cost ', time_3 - time_2, ' second', '\n'
    print 'Start predicting'
    test_predict = p.predict_batch(test_features)
    time_4 = time.time()
    print 'predicting cost ', time_4 - time_3, ' second', '\n'
    score = accuracy_score(test_labels, test_predict)
    print "The accruacy socre is ", score
