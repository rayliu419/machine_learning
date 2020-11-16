import numpy as np
import torch
import pandas as pd


def print_type_value(item):
    print(type(item))
    print(item)


def list_to_ndarray(list):
    return np.array(list)


def list_to_tensor(list):
    tensor = torch.Tensor(list)
    return tensor


def var_list_to_tensor(list_of_list, padding):
    maxlen = max(len(l) for l in list_of_list)
    new_matrix = list(map(lambda l: l + [padding] * (maxlen - len(l)), list_of_list))
    return list_to_tensor(new_matrix)


def ndarray_to_list(ndarray):
    list = ndarray.tolist()
    return list


def ndarray_to_tensor(ndarray):
    return torch.from_numpy(ndarray)


def tensor_to_list(tensor):
    list = tensor.numpy().tolist()
    return list


def tensor_to_ndarray(tensor):
    np_arr = tensor.numpy()
    # gpu上的tensor不能直接转为numpy
    # np_arr = tensor.cpu().numpy()
    return np_arr


def series_to_list(series):
    return series.tolist()


def series_to_ndarray(series):
    list = series_to_list(series)
    ndarray = list_to_ndarray(list)
    return ndarray


def dataframe_to_list(df):
    mul_list = []
    for index in df.columns:
        series = df[index]
        cur_list = series_to_list(series)
        mul_list.append(cur_list)
    return mul_list


if __name__ == "__main__":
    # list to ndarray
    list = [1, 2, 3]
    print_type_value(list_to_ndarray(list))
    list2 = [[1], [2]]
    print_type_value(list_to_ndarray(list2))
    list3 = [[1], [2, 3]]
    # 多维对不齐的，其实不能真正的转成ndarray
    print_type_value(list_to_ndarray(list3))

    # ndarray to list
    ndarray = np.array([1, 2, 3])
    print_type_value(ndarray_to_list(ndarray))

    # 2.1 list 转 torch.Tensor
    print_type_value(list_to_tensor(list))

    # 2.2 torch.Tensor 转 list, 先转numpy，后转list
    tensor = torch.Tensor([1, 2, 3])
    print_type_value(tensor_to_list(tensor))

    # 3.1 torch.Tensor 转 numpy
    print_type_value(tensor_to_ndarray(tensor))

    # 3.2 numpy 转 torch.Tensor
    print_type_value(ndarray_to_tensor(ndarray))

    df = pd.DataFrame({'a': [1, 3, 5, 7, 4, 5, 6, 4, 7, 8, 9], 'b': [3, 5, 6, 2, 4, 6, 7, 8, 7, 8, 9]})

    print_type_value(series_to_list(df["a"]))
