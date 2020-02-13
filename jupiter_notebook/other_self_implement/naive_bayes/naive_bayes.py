#encoding=utf-8

import pandas as pd
import numpy as np
import cv2
import time

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# 朴素贝叶斯的思想 P(Y|x1x2...) = P(Y) * P(x1x2..|Y) / P(x1x2...)
# 由于朴素贝叶斯假设x1x2是相互独立的,所以P(x1x2..|Y) = P(x1|Y)*P(x2|Y)...
# 对于分类问题实际上是计算P(Y=yj|x1x2...), 对于这个概率, 实际上P(x1x2..)是一样的,所以不需要计算。
# 下面的实现其实没有考虑平滑化的问题,训练集中可能存在P(x1|Y)=0的情况。


def binaryzation(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img


def Train(trainset, train_labels):
    prior_count = np.zeros(class_num)
    conditional_probability = np.zeros((class_num, feature_len, 2))

    # 计算先验概率及条件概率
    for i in range(len(train_labels)):
        # 分析各个训练sample
        img = binaryzation(trainset[i])
        label = train_labels[i]
        # 标签的先验概率
        prior_count[label] += 1
        for j in range(feature_len):
            # 实际上是在计算count(Xj|Y), 把计数存在
            conditional_probability[label][j][img[j]] += 1
    # 将概率归到[1.10001], 估计是为了保证连乘的概率不上下越界。
    for i in range(class_num):
        for j in range(feature_len):
            # 经过二值化后图像只有0，1两种取值
            value_0 = conditional_probability[i][j][0]
            value_1 = conditional_probability[i][j][1]
            # 计算0，1像素点对应的条件概率
            probalility_0 = (float(value_0)/float(value_0 + value_1))*1000000 + 1
            probalility_1 = (float(value_1)/float(value_0 + value_1))*1000000 + 1

            conditional_probability[i][j][0] = probalility_0
            conditional_probability[i][j][1] = probalility_1
    return prior_count, conditional_probability


# 根据前面的资料，计算为某个标签的概率
def calculate_probability(img, label, prior_count, conditional_probability):
    # 这个地方使用的不是P(Yi)，估计是为了保证连乘的概率不上下越界
    probability = int(prior_count[label])
    for i in range(len(img)):
        probability *= int(conditional_probability[label][i][img[i]])
    return probability


def Predict(testset, prior_count, conditional_probability):
    predict = []
    for img in testset:
        # 图像二值化
        img = binaryzation(img)
        max_label = 0
        max_probability = calculate_probability(img, 0, prior_count, conditional_probability)
        for j in range(1, 10):
            probability = calculate_probability(img, j, prior_count, conditional_probability)
            if max_probability < probability:
                max_label = j
                max_probability = probability
        predict.append(max_label)
    return np.array(predict)

class_num = 10
feature_len = 784

if __name__ == '__main__':
    print 'Start read data'
    time_1 = time.time()
    raw_data = pd.read_csv('../data/train.csv',header=0)
    data = raw_data.values
    imgs = data[0:, 1:]
    labels = data[:, 0]
    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(imgs, labels, test_size=0.33, random_state=23323)
    time_2 = time.time()
    print 'read data cost ', time_2 - time_1, ' second', '\n'
    print 'Start training'
    prior_count, conditional_probability = Train(train_features, train_labels)
    time_3 = time.time()
    print 'training cost ', time_3 - time_2, ' second', '\n'
    print 'Start predicting'
    test_predict = Predict(test_features, prior_count, conditional_probability)
    time_4 = time.time()
    print 'predicting cost ', time_4 - time_3, ' second',' \n'
    score = accuracy_score(test_labels, test_predict)
    print "The accruacy socre is ", score