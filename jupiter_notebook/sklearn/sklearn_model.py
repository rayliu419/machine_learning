# encoding=utf-8

'''
比较不同模型的效果
每个都要使用kfold的方式来计算平均值
'''

import sklearn.cross_validation
import sklearn.ensemble
from sklearn import linear_model
from sklearn.ensemble import AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn_feature import *

classifiers = {
    "LR": linear_model.LogisticRegression(),
    "SVC": SVC(kernel="linear", C=0.025),
    "GNB": GaussianNB(),
    "RF": sklearn.ensemble.RandomForestClassifier(n_estimators=10),
    "AB": AdaBoostClassifier(n_estimators=100)
}


def evaluate_model_k_fold(model_name, X, Y, standardize=False, k=3):
    f1_result = []
    accuracy_result = []
    recall_result = []
    kf = sklearn.cross_validation.KFold(X.shape[0], n_folds=k, shuffle=True)
    for train_index, test_index in kf:
        if (standardize):
            # ndarray
            X_train = X.take(train_index, axis=0)
        else:
            X_train = X.ix[train_index]
            if "index" in X_train.columns:
                X_train = X_train.drop("index", axis=1)
        Y_train = Y.ix[train_index]
        if (standardize):
            # ndarray
            X_test = X.take(test_index, axis=0)
        else:
            X_test = X.ix[test_index]
            if "index" in X_test.columns:
                X_test = X_test.drop("index", axis=1)
        Y_test = Y.ix[test_index]
        model = classifiers[model_name]
        model.fit(X_train, Y_train.values.ravel())
        Y_predict = model.predict(X_test)
        _f1, _accuracy, _recall = metrics_models(Y_predict, Y_test)
        f1_result.append(_f1)
        accuracy_result.append(_accuracy)
        recall_result.append(_recall)
    return model_name, float(sum(f1_result)) / len(f1_result), \
           float(sum(accuracy_result)) / len(accuracy_result), \
           float(sum(recall_result)) / len(recall_result)


def metrics_models(Y_predict, Y_test):
    f1 = sklearn.metrics.f1_score(Y_predict, Y_test)
    accuracy = sklearn.metrics.accuracy_score(Y_predict, Y_test)
    recall = sklearn.metrics.recall_score(Y_predict, Y_test)
    return f1, accuracy, recall
