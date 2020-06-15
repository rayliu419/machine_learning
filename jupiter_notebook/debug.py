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
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, filter_num, (cur, embedding_size)) for cur in filter_sizes])
        self.dropout = nn.Dropout(drop_out_ratio)
        self.fc = nn.Linear(filter_num * len(filter_sizes), class_num)

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

args = {
    "class_num": 2,
    "filter_sizes": [2, 3, 4],
    "embedding_size": 100,
    "filter_num": 16,
    "dropout": 0.5,

}

textCNN = TextCNN(args)
