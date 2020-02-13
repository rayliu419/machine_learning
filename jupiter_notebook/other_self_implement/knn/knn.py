# encoding=utf-8

import time

import cv2
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

NEAREST_K_NUM = 10
CLASS_TOTAL = 10

# 将原始的一维的特征变为使用opencv转换的hog特征
def get_hog_features(training_set):
    hog_features = []
    hog = cv2.HOGDescriptor('../hog.xml')
    for img in training_set:
        # 将其变成28 * 28的图片
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)
        hog_feature = hog.compute(cv_img)
        hog_features.append(hog_feature)
    return np.array(hog_features).reshape((-1, 324))


def predict(testset, trainset, train_labels):
    predict = []
    for test_vec in testset:
        # 当前k个最近邻居
        knn_list = []
        # 当前k个最近邻居中距离最远点的坐标
        max_index = -1
        # 当前k个最近邻居中距离最远点的距离
        max_dist = 0
        # 先将前k个点放入k个最近邻居中，填充满knn_list
        for i in range(NEAREST_K_NUM):
            label = train_labels[i]
            train_vec = trainset[i]
            dist = np.linalg.norm(train_vec - test_vec)  # 计算两个点的欧氏距离
            knn_list.append((dist, label))
        # 剩下的点
        for i in range(NEAREST_K_NUM, len(train_labels)):
            label = train_labels[i]
            train_vec = trainset[i]
            dist = np.linalg.norm(train_vec - test_vec)  # 计算两个点的欧氏距离
            # 寻找10个邻近点钟距离最远的点
            if max_index < 0:
                for j in range(NEAREST_K_NUM):
                    if max_dist < knn_list[j][0]:
                        max_index = j
                        max_dist = knn_list[max_index][0]
            # 如果当前k个最近邻居中存在点距离比当前点距离远，则替换
            if dist < max_dist:
                knn_list[max_index] = (dist, label)
                max_index = -1
                max_dist = 0
        # 统计选票
        class_count = np.zeros(CLASS_TOTAL)
        for dist, label in knn_list:
            class_count[label] += 1
        # 仅仅返回第一个最大的
        max_class_index = np.argmax(class_count)
        predict.append(max_class_index)
    return np.array(predict)

if __name__ == '__main__':
    print 'Start read data'
    time_1 = time.time()
    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values
    imgs = data[0:, 1:]
    labels = data[0:, 0]
    features = get_hog_features(imgs)
    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
                                                                                random_state=23323)
    time_2 = time.time()
    print 'read data cost ', time_2 - time_1, ' second', '\n'
    print 'Start training'
    print 'knn do not need to train'
    time_3 = time.time()
    print 'training cost ', time_3 - time_2, ' second', '\n'
    print 'Start predicting'
    test_predict = predict(test_features, train_features, train_labels)
    time_4 = time.time()
    print 'predicting cost ', time_4 - time_3, ' second', '\n'
    score = accuracy_score(test_labels, test_predict)
    print "The accruacy socre is ", score
