# encoding=utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.datasets import make_hastie_10_2
from sklearn.tree import DecisionTreeClassifier


def get_error_rate(pred, Y):
    return sum(pred != Y) / float(len(Y))


def print_error_rate(err):
    print 'Error rate: Training: %.4f - Test: %.4f' % err


def generic_clf(Y_train, X_train, Y_test, X_test, clf):
    clf.fit(X_train, Y_train)
    pred_train = clf.predict(X_train)
    pred_test = clf.predict(X_test)
    return get_error_rate(pred_train, Y_train), \
           get_error_rate(pred_test, Y_test)


# 在这个函数中，没有将train出来的弱分类器模型存储起来，是一个一次性的算法
def adaboost_clf(Y_train, X_train, Y_test, X_test, classifier_num):
    n_train, n_test = len(X_train), len(X_test)
    # 初始化权重
    w = np.ones(n_train) / n_train
    accumulate_train_predict = np.zeros(n_train)
    accumulate_test_predict = np.zeros(n_test)
    # for debugging
    #err_m_array = []
    #alpha_m_array = []
    #w1 = []
    weight_model_pair = []
    for i in range(classifier_num):
        # Fit a classifier with the specific weights
        #w1.append(w[1])
        clf = DecisionTreeClassifier(max_depth=1)
        clf.fit(X_train, Y_train, sample_weight=w)
        cur_classifier_train_predict = clf.predict(X_train)
        cur_classifier_test_predict = clf.predict(X_test)
        miss = [int(predict_wrong) for predict_wrong in (cur_classifier_train_predict != Y_train)]
        # 这个数组主要是用来后面的sample权重计算的
        miss2 = [x if x == 1 else -1 for x in miss]
        # 错误率，注意这里predict错了的在miss2中是1，对的是-1
        err_m = sum(i for i in miss if i == 1) / float(len(cur_classifier_train_predict))
        # 分类器权重
        alpha_m = 0.5 * np.log((1 - err_m) / float(err_m))
        # 权重更新，注意权重是一个向量
        w = np.multiply(w, np.exp([float(x) * alpha_m for x in miss2]))
        # 权重归一化
        w_sum = np.sum(w)
        w_next = w / w_sum
        w = w_next
        #err_m_array.append(err_m)
        #alpha_m_array.append(alpha_m)
        weight_model_pair.append({alpha_m, clf})
        # 这个地方实际上将本轮使用的分类器的打分加到总得分中
        for index, item in enumerate(cur_classifier_train_predict):
            accumulate_train_predict[index] += alpha_m * item
        for index, item in enumerate(cur_classifier_test_predict):
            accumulate_test_predict[index] += alpha_m * item
    # 所有弱分类器投票的最终得分 
    # print "error rate : "
    # print err_m_array
    # print "alpha weight : "
    # print alpha_m_array
    # print "w"
    # print w1
    print weight_model_pair[0:5]
    train_Y_final = np.sign(accumulate_train_predict)
    test_Y_final = np.sign(accumulate_test_predict)
    # Return error rate in train and test set
    return get_error_rate(train_Y_final, Y_train), \
           get_error_rate(test_Y_final, Y_test)


def plot_error_rate(er_train, er_test):
    df_error = pd.DataFrame([er_train, er_test]).T
    df_error.columns = ['Training', 'Test']
    plot1 = df_error.plot(linewidth=3, figsize=(8, 6),
                          color=['lightblue', 'darkblue'], grid=True)
    plot1.set_xlabel('Number of iterations', fontsize=12)
    plot1.set_xticklabels(range(0, 450, 50))
    plot1.set_ylabel('Error rate', fontsize=12)
    plot1.set_title('Error rate vs number of iterations', fontsize=16)
    plt.axhline(y=er_test[0], linewidth=1, color='red', ls='dashed')


if __name__ == '__main__':
    # 读取数据
    print "read data"
    x, y = make_hastie_10_2(n_samples=10000)
    df = pd.DataFrame(x)
    df['Y'] = y
    train, test = train_test_split(df, test_size=0.2)
    X_train, Y_train = train.ix[:, :-1], train.ix[:, -1]
    X_test, Y_test = test.ix[:, :-1], test.ix[:, -1]
    print "base classifier training and predicting"
    clf_tree = DecisionTreeClassifier(max_depth=1, random_state=1)
    er_tree = generic_clf(Y_train, X_train, Y_test, X_test, clf_tree)
    # 使用一层决策树来做为弱分类器
    # 存起来主要是为了比较
    er_train, er_test = [er_tree[0]], [er_tree[1]]
    classifier_nums = [200, 400, 600, 800]
    # 这里的i代表弱分类器的个数，这里是为了比较弱分类器个数增加，效果会进一步提升
    # 这里有一个特别奇怪的现象，有的时候，会发现error_rate在从200个弱分类器的时候就开始停止增长了，一直保持不变，这是为什么？看起来
    # 跟产生的弱分类器都类似有关系，可以通过DecisionTreeClassifier设置随机的初始化来改善这个问题吗？
    for classifier_number in classifier_nums:
        print "classifier numbers: " + str(classifier_number)
        cur_err = adaboost_clf(Y_train, X_train, Y_test, X_test, classifier_number)
        er_train.append(cur_err[0])
        er_test.append(cur_err[1])
    print er_train
    print er_test