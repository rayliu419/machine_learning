from keras.utils.np_utils import to_categorical
import numpy as np


def one_hot_encoding(int_labels, num_classes):
    """
    下面这两个函数将int的变成one-hot encoding的模式。
    例如:
    1 - [0, 1]
    5 - [0, 0, 0, 0, 1]
    """
    one_hot = to_categorical(int_labels, num_classes)
    return one_hot


def one_hot_encoding_2d(int_labels_2d, num_classes):
    """
    按道理还是得有相同的size的
    :param int_labels_2d:
    :param num_classes:
    :return:
    """
    one_hot_2d = []
    for int_labels in int_labels_2d:
        cur = one_hot_encoding(int_labels, num_classes)
        one_hot_2d.append(cur)
    return one_hot_2d


if __name__ == "__main__":
    print("one hot encoding, ndarray")
    one_hot_np = one_hot_encoding([0, 1, 4, 6], num_classes=7)
    print(type(one_hot_np))
    print(one_hot_np)

    print("2d one hot encoding, different size, list")
    one_hot_2d_list = one_hot_encoding_2d([[0, 1, 3], [4, 6], [6, 6, 5, 6, 4]], num_classes=7)
    print(type(one_hot_2d_list))
    print(one_hot_2d_list)

    # 不同长度的list应该不能转ndarray吧？
    print("#2d one hot encoding, different size, ndarray")
    one_hot_2d_list_np = np.array(one_hot_2d_list)
    print(type(one_hot_2d_list_np))
    print(one_hot_2d_list_np.shape)
    print(one_hot_2d_list_np)

    print("#2d one hot encoding, same size, list")
    one_hot_2d_list_with_same_size = one_hot_encoding_2d([[1, 3], [4, 6]], num_classes=7)
    print(type(one_hot_2d_list_with_same_size))
    print(one_hot_2d_list_with_same_size)

    print("2d one hot encoding, same size, ndarray")
    one_hot_2d_list_with_same_size_np = np.array(one_hot_2d_list_with_same_size)
    print(type(one_hot_2d_list_with_same_size_np))
    print(one_hot_2d_list_with_same_size_np)
