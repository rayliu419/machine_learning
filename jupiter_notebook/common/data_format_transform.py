import numpy as np
import torch


def print_type_value(item):
    print(type(item))
    print(item)

list = [1, 2, 3]

#1.1 list 转 numpy
ndarray = np.array(list)
print_type_value(ndarray)

# 1.2 numpy 转 list
list = ndarray.tolist()
print_type_value(list)

# 2.1 list 转 torch.Tensor
tensor = torch.Tensor(list)
print_type_value(tensor)

# 2.2 torch.Tensor 转 list, 先转numpy，后转list
list = tensor.numpy().tolist()
print_type_value(list)

# 3.1 torch.Tensor 转 numpy
ndarray = tensor.numpy()
# *gpu上的tensor不能直接转为numpy
ndarray = tensor.cpu().numpy()
print_type_value(ndarray)

# 3.2 numpy 转 torch.Tensor
tensor = torch.from_numpy(ndarray)
print_type_value(tensor)