import pandas as pd
import os


def prepare_adult_data_for_task():
    """
    返回的依然是ndarray类型
    :param line_num:
    :return:
    """
    print("load data")
    train_file = os.path.dirname(__file__) + "/input_data/adult.data"
    test_file = os.path.dirname(__file__) + "/input_data/adult.test"
    train_data = pd.read_csv(train_file)
    test_data = pd.read_csv(test_file)
    # 前8列是feature, 第8列是标签
    train_data.info()
    test_data.info()
    return train_data, test_data


if __name__ == "__main__":
    prepare_adult_data_for_task()

