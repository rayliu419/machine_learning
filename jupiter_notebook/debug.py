"""
textCNN做电影的情感分类任务
"""

import sys
sys.path.append("./common/")
import english_preprocess
import movie_review_helper
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


class TextCNN(nn.Module):
    def __init__(self, args):
        super(TextCNN, self).__init__()
        print(args)
        filter_num = args["filter_num"]
        filter_sizes = args["filter_sizes"]
        embedding_size = args["embedding_size"]
        drop_out_ratio = args["dropout"]
        class_num = args["class_num"]
        """
        Conv2d初始化参数
            in_channels - NLP中是几种embedding方式
            out_channels - filter的数目，多个filter可能是捕捉多种不同长度的模式
            kernel_size - NLP中是ngram的(n, embedding_size)
        """
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (cur, embedding_size)) for cur in filter_sizes])
        self.dropout = nn.Dropout(drop_out_ratio)
        self.fc = nn.Linear(filter_num * len(filter_sizes), class_num)

    def forward(self, x):
        out = x
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def conv_and_pool(self, x, conv):
        """
        ngram的n=某个值的卷积和max_pool，一次计算多个sample，多个filter
        Conv2d的输入解释：
            N - batch的大小
            C - 通道数量, 这里怎么理解？
            H - 输入的高度
            W - 输入的宽度
        Conv2d的输出
            N - batch大小，跟输入的是一样的。
            C - 根据例子来看，是跟Conv2d初始化的filter_num是一样的。
            H - 高度，在textCNN中其实就是filter在单个sample输入滑动产生的输出。
            W - 在textCNN中，应该就是1
        :param x:
        :param conv:
        :return:
        """
        # x - (8, 1, 5, 2), 8 - sample, 1 - 编码方式个数， 5 - 句子长度，2 - embedding空间大小
        output = conv(x)
        # output - (8, 2, 4, 1) 8 - sample, 2 - 每个filter的结果， 4 - 每个filter卷积的结果， 1 - 卷积结果的维度
        output = F.relu(output)
        # 这里的两个squeeze其实都是为了最后产生一个(sample, filter_num)的shape
        output = output.squeeze(3)
        # output - (8, 2, 4)
        output = F.max_pool1d(output, output.size(2))
        # output - (8, 2, 1)，每个filter产生一个4 * 1的结果，在4行里取最大值，代表的是模式抓取的语义
        output = output.squeeze(2)
        return output


def get_embedding_samples(samples, embedding_mapping, sentence_length, embedding_size):
    samples_mapping = torch.empty((len(samples), sentence_length, embedding_size))
    sample_index = 0
    for sample in samples:
        cur_sample = []
        for index in sample:
            index_embedding = embedding_mapping[index]
            cur_sample.append(index_embedding)
        sample_tensor = torch.from_numpy(np.array(cur_sample))
        samples_mapping[sample_index] = sample_tensor
        sample_index += 1
    return samples_mapping

"""
用来细致观察CNN的初始化，input，output
"""
line_num = 10
sentence_maxlen = 5
embedding_len = 2
args = {
    "class_num": 2,
    "filter_sizes": [2, 3, 4],
    "embedding_size": embedding_len,
    "filter_num": 2,
    "dropout": 0.5,

}
print("#0 load movie review data")
textCNN = TextCNN(args)
tokenizer, embedding_mapping, X, Y, X_train, X_test, Y_train, Y_test = \
    movie_review_helper.prepare_movie_review_for_task(line_num, sentence_maxlen, embedding_len)
X_train_embedding = get_embedding_samples(X_train, embedding_mapping, sentence_maxlen, embedding_len)
print("#1 calculate net")
Y_hat = textCNN(X_train_embedding)
# Y_hat 应该是(8, 2)
print("#2 call soft max")
softmax_func = nn.Softmax(dim=1)
Y_hat_softmax = softmax_func(Y_hat)
print(Y_hat_softmax)
print("#3 finish")


