"""
textCNN做电影的情感分类任务
"""

import sys
sys.path.append("./common/")
import english_preprocess
import movie_review_helper
import probability_utils
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data as data_utils


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


def trainset_compare(before, after, actual):
    before_predict = before.argmax(dim=1)
    after_predict = after.argmax(dim=1)
    before_correct = probability_utils.sum_equal(before_predict, actual)
    after_correct = probability_utils.sum_equal(after_predict, actual)
    # print("batch num - {}, before_prdict - {}, after_predict - {}".format(len(actual), before_correct, after_correct))

"""
用来细致观察CNN的初始化，input，output
"""
line_num = 100
sentence_maxlen = 100
embedding_len = 2
filter_num = 10
dropout = 0.3
epochs = 500
lr = 0.002
args = {
    "line_num": line_num,
    "sentence_maxlen": sentence_maxlen,
    "class_num": 2,
    "filter_sizes": [2, 3, 4],
    "embedding_size": embedding_len,
    "filter_num": filter_num,
    "dropout": dropout,
    "lr": lr,
    "epochs": epochs
}
print("#0 load movie review data")
tokenizer, embedding_mapping, _, _, X_train_np, X_test_np, Y_train_np, Y_test_np = \
    movie_review_helper.prepare_movie_review_for_task(line_num, sentence_maxlen, embedding_len)

X_train_embedding_tensor = get_embedding_samples(X_train_np, embedding_mapping, sentence_maxlen, embedding_len)
X_test_embedding_tensor = get_embedding_samples(X_test_np, embedding_mapping, sentence_maxlen, embedding_len)
Y_train_tensor = torch.from_numpy(Y_train_np).type(torch.LongTensor)
Y_test_tensor = torch.from_numpy(Y_test_np).type(torch.LongTensor)

print("#1 create textCNN")
textCNN = TextCNN(args)

print("# 2 create optimizer and loss")

optimizer = torch.optim.SGD(textCNN.parameters(), lr=lr)
loss_func = torch.nn.CrossEntropyLoss()

print("#3 batch loader creating")
train_data = data_utils.TensorDataset(X_train_embedding_tensor, Y_train_tensor)
train_loader = data_utils.DataLoader(train_data, batch_size=32, shuffle=True)

print("#4 start to train")
print(args)
for i in range(epochs):
    for index, data in enumerate(train_loader):
        X_batch_tensor, Y_batch_tensor = data
        net_output_tensor = textCNN(X_batch_tensor)
        loss = loss_func(net_output_tensor, Y_batch_tensor)
        before = net_output_tensor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        after = textCNN(X_batch_tensor)
        trainset_compare(before, after, Y_batch_tensor)
    if i % 10 == 0:
        Y_test_predict_net_output = textCNN(X_test_embedding_tensor)
        Y_test_predict = Y_test_predict_net_output.argmax(dim=1)
        total = len(Y_test_tensor)
        correct_prediction = probability_utils.sum_equal(Y_test_tensor, Y_test_predict)
        accuracy = float(correct_prediction) / total
        print("total - {}, epoch - {}, accuracy - {}".format(total, i, accuracy))
