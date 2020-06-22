#encoding=utf-8

import cv2
import time
import logging
import numpy as np
import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score


TOTAL_CLASS = 10
LEAF = 'leaf'
INTERNAL = 'internal'


# 用来打印日志的，可以封装函数，这样在函数中不用单独log
def log(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        logging.debug('start %s()' % func.__name__)
        ret = func(*args, **kwargs)
        end_time = time.time()
        logging.debug('end %s(), cost %s seconds' % (func.__name__,end_time-start_time))
        return ret
    return wrapper


# 这个二值化实际上是黑白色反转，首先根据是否大于50变为0或者1
def binarization(img):
    cv_img = img.astype(np.uint8)
    cv2.threshold(cv_img, 50, 1, cv2.THRESH_BINARY_INV, cv_img)
    return cv_img


# 原始的数据集是一个一维数组, reshape以后是28 * 28的二维数组，二值化的地方是将小于50的变为1，大于50的变为0。然后又变回一维数组
@log
def binarization_features(trainset):
    features = []
    for img in trainset:
        img = np.reshape(img, (28, 28))
        cv_img = img.astype(np.uint8)
        img_b = binarization(cv_img)
        features.append(img_b)
    return np.array(features).reshape(-1, 784)


class Tree(object):
    def __init__(self, node_type, Class=None, feature=None):
        self.node_type = node_type
        # 相当于普通树下的子树连接信息，某个feature等于某个值时的子树
        self.dict = {}
        self.Class = Class
        self.feature = feature

    def add_tree(self, val, tree):
        self.dict[val] = tree

    def predict(self, features):
        if self.node_type == LEAF:
            return self.Class
        tree = self.dict[features[self.feature]]
        return tree.predict(features)


# 计算熵
# 熵为0即代表事情是完全确定的。plog2(p)
def calc_ent(x):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        p = float(x[x == x_value].shape[0]) / x.shape[0]
        logp = np.log2(p)
        ent -= p * logp
    return ent


# 计算条件熵H(Y|X=x)
# 条件熵表示知道x的取值以后，我们对Y取值的确定程度
def calc_condition_ent(x, y):
    x_value_list = set([x[i] for i in range(x.shape[0])])
    ent = 0.0
    for x_value in x_value_list:
        sub_y = y[x == x_value]
        temp_ent = calc_ent(sub_y)
        ent += (float(sub_y.shape[0]) / y.shape[0]) * temp_ent
    return ent

def recurse_train(train_set, train_label, features, epsilon):
    # 如果train_set中的所有实例都属于同一类，直接返回一个叶节点
    label_set = set(train_label)
    if len(label_set) == 1:
        return Tree(LEAF, Class=label_set.pop())
    # 计算当前节点下各个类别的分布
    class_with_number = [(i, len(filter(lambda x: x == i, train_label))) for i in xrange(TOTAL_CLASS)]
    (max_class, max_len) = max(class_with_number, key=lambda x: x[1])

    # 虽然还有训练集不同的，但是已经没有feature可供选择了，直接返回，以最多的类别分布作为预测
    if len(features) == 0:
        return Tree(LEAF, Class=max_class)

    # 步骤3——计算信息增益
    max_feature = 0
    max_gda = 0
    entropy = calc_ent(train_label)
    for feature in features:
        cur_feature_list = np.array(train_set[:, feature].flat)
        # 信息增益 = 熵 - 条件熵
        # 信息增益越大表示当前这个feature对分类产生的效果越有用
        gda = entropy - calc_condition_ent(cur_feature_list, train_label)
        if gda > max_gda:
            # 有更好的feature选取
            max_gda, max_feature = gda, feature
    # 小于阈值，直接构造叶子节点，防止过拟合
    if max_gda < epsilon:
        return Tree(LEAF, Class = max_class)
    # 创建中间节点，减去当前选定这个feature
    sub_features = filter(lambda x: x != max_feature, features)
    tree = Tree(INTERNAL, feature=max_feature)
    # 选取当前选中的feature的取值集合，为了下面的构造子决策树提供依据，下面的子决策树不再使用这个feature作为分裂依据
    feature_col = np.array(train_set[:, max_feature].flat)
    feature_value_list = set([feature_col[i] for i in range(feature_col.shape[0])])
    for feature_value in feature_value_list:
        # 这里实际上根据当前选中的feature，分裂节点
        index = []
        for i in xrange(len(train_label)):
            if train_set[i][max_feature] == feature_value:
                # 收集在feature在这个取值下的训练集
                index.append(i)
        # 收集当前分裂的feature某个取值下的训练集并继续向下构造树
        sub_train_set = train_set[index]
        sub_train_label = train_label[index]
        sub_tree = recurse_train(sub_train_set, sub_train_label, sub_features, epsilon)
        tree.add_tree(feature_value, sub_tree)
    return tree

@log
def train(train_set, train_label, features, epsilon):
    return recurse_train(train_set, train_label, features, epsilon)

@log
def predict_batch(test_set, tree):
    result = []
    for features in test_set:
        tmp_predict = tree.predict(features)
        result.append(tmp_predict)
    return np.array(result)


if __name__ == '__main__':
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    raw_data = pd.read_csv('../data/train.csv', header=0)
    data = raw_data.values
    imgs = data[0:, 1:]
    labels = data[:, 0]
    # feature二值化
    features = binarization_features(imgs)
    # 选取 2/3 数据作为训练集， 1/3 数据作为测试集
    train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33, random_state=23323)
    tree = train(train_features, train_labels, [i for i in range(784)], 0.1)
    test_predict = predict_batch(test_features, tree)
    score = accuracy_score(test_labels, test_predict)
    print "The accuracy score is ", score

