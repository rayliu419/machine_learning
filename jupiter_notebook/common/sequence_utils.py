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
