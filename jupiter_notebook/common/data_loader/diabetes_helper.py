import pandas as pd
import os
from sklearn.model_selection import train_test_split


def prepare_diabetes_raw_data_for_task():
    """
    返回diabetes的raw data
    :return: pandas data frame
    """
    print("load data")
    data_file = os.path.dirname(__file__) + "/input_data/diabetes.csv"
    pima_data = pd.read_csv(data_file)
    return pima_data


def prepare_diabetes_data_for_task(line_num=1000):
    """
    返回的依然是ndarray类型
    :param line_num:
    :return:
    """
    print("load data")
    data_file = os.path.dirname(__file__) + "/input_data/diabetes.csv"
    pima_data = pd.read_csv(data_file)
    pima_data = pima_data[0:line_num]
    # 前8列是feature, 第8列是标签
    X = pima_data.iloc[:, 0: 8]
    Y = pima_data.iloc[:, 8]
    X_np = X.to_numpy()
    Y_np = Y.to_numpy()
    X_train_np, X_test_np, Y_train_np, Y_test_np = train_test_split(X_np, Y_np, test_size=0.20, random_state=42)
    return X, Y, X_train_np, X_test_np, Y_train_np, Y_test_np

