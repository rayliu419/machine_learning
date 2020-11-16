import torch


def partition_by_size(l, n):
    """
    返回的是generator，注意想多次遍历，要将结果转回list
    :param l:
    :param n:
    :return:
    """
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def sort_list_of_list_by_list_length(l):
    '''
    pytorch的pack_padded_sequence()要求list要从大到小排列
    :param l:
    :return:
    '''
    l.sort(key=len, reverse=True)
    return l


def get_length_of_list_of_list(l):
    return [len(i) for i in l]


def count_nonzero(tensor_2d):
    """
    pytorch 1.7 has this function, but now let's implement a simple version.
    :return:
    """
    _, column_number = tensor_2d.shape
    return column_number - (tensor_2d == 0).sum(dim=1)


if __name__ == '__main__':
    l = [
        ["ram", "mohan", "aman"],
        ["gaurav"],
        ["amy", "sima", "ankita", "rinku"]
    ]
    print(sort_list_of_list_by_list_length(l))
    x = torch.zeros(3, 3)
    x[torch.randn(3, 3) > 0.5] = 1
    print(x)
    print(count_nonzero(x))