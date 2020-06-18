"""
Util class to help calculate probability.

"""


def sum_equal(tensor1, tensor2):
    """
    比较预测的类别正确率
    输入都是pytroch.tensor类型
    :param tensor1: [[1], [0], [2]]...
    :param tensor2: [[2], [1], [3]]...
    :return:
    """
    ndarray1 = tensor1.numpy()
    ndarray2 = tensor2.numpy()
    return (ndarray1 == ndarray2).astype(int).sum()

