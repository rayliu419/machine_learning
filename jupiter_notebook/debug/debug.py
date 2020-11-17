import sys
sys.path.append("..")
from common.utils import sequence_utils
from common.utils.type_cast_utils import *
from common.utils.model_utils import *
from common.utils.nlp_utils import *
from common.data_loader import movie_review_helper
from common.lng_processor.eng_process import *
import torch
from torch import nn

"""
演示怎么正确处理变长类型的输入，不使用截断的方法，而是使用padding + masking。
正确理解torch.nn.LSTM的输入和输出。
"""

"""
新的电影评论在这个上面的性能不好。
从优化的速度和效果来看，耗时间长且优化的loss下降慢，为什么? 可能的原因是:
1. 整体的sample数目还是有点少。2000个正例，2000个负例。
2. 每个sample的单词数比较多，几百上千个。
df = load_movie_review()
df['text_remove_punctuations'] = \
    df['text'].apply(lambda x: remove_punctuations_array(x))
df.info()

word_dict = build_word_to_int_dict(df['text_remove_punctuations'])
print(word_dict.total_token)
print(word_dict.total_uniq_word)

print("encoding")
df['encode'] = df['text_remove_punctuations'].apply(lambda x: word_dict.encode_tokenized_sentence(x))

print(df.loc[0:0, ['text_remove_punctuations']])
print(df.loc[0:0, ['encode']])

df.drop(columns=['text_remove_punctuations', 'text'], inplace=True)

df['label_encode'] = df['label'].apply(lambda x: 0 if x == 'neg' else 1)
df.drop(columns=['label'], inplace=True)

print(df.loc[0:1])
"""

print("loading and cleaning data")
df = movie_review_helper.prepare_movie_review_raw(5000)
df['text_remove_punctuations'] = \
    df['text'].apply(lambda x: remove_punctuations_array(x))
df.info()

word_dict = build_word_to_int_dictionary(df['text_remove_punctuations'])
print(word_dict.total_token)
print(word_dict.total_uniq_word)

print("encoding")
df['encode'] = df['text_remove_punctuations'].apply(lambda x: word_dict.encode_tokenized_sentence(x))

print(df.loc[0:0, ['text_remove_punctuations']])
print(df.loc[0:0, ['encode']])

df.drop(columns=['text_remove_punctuations', 'text'], inplace=True)

df['label_encode'] = df['label'].apply(lambda x: 0 if x == 'negative' else 1)
df.drop(columns=['label'], inplace=True)


class VariableNet(nn.Module):
    def __init__(self, word_dict_size, embedding_dimision, lstm_hidden_size):
        super(VariableNet, self).__init__()
        self.word_dict_size = word_dict_size
        self.embedding_dimision = embedding_dimision
        self.lstm_hidden_size = lstm_hidden_size
        # what's value of matrix[0]? - [.0, .0, .0...]
        self.embedding = torch.nn.Embedding(num_embeddings=word_dict_size,
                                            embedding_dim=embedding_dimision, padding_idx=0)
        self.lstm = torch.nn.LSTM(input_size=embedding_dimision, hidden_size=lstm_hidden_size,
                                  batch_first=True)
        self.dropout = torch.nn.Dropout()
        # batch_size * 1
        self.fc = torch.nn.Linear(lstm_hidden_size, 16)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(16, 1)
        # sigmoid
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        # 注意每次计算时，要重新初始化h0和c0。
        h0 = torch.zeros(x.size(0), self.lstm_hidden_size).view((1, x.size(0), -1))
        c0 = torch.zeros(x.size(0), self.lstm_hidden_size).view((1, x.size(0), -1))
        # 为了获取变长的每个输入的长度
        sample_number = x.size(0)
        # 获取变长的sample的长度
        sample_lengths = sequence_utils.count_nonzero(x)
        input_embedding = self.embedding(x.long())
        """
        转换输入的格式
        把上一个输入整平，把每个sample的第i个time step按顺序合并到一起，同时会有一个batch_sizes记录每个time step有多少个。
        例如上例说明第一个time step有2个输入，第二个也是2个，第三个有1个等。
        采用这样的方式输入到lstm中，可以提高lstm的计算效率。
        """
        packed_seq_batch = torch.nn.utils.rnn. \
            pack_padded_sequence(input_embedding, lengths=sample_lengths, batch_first=True)
        output, (hn, cn) = self.lstm(packed_seq_batch.float(), (h0.detach(), c0.detach()))
        """
        输出也得转回来，这里的问题是：我们不是从padded_output来取结果，而是从hn取结果。原因是因为padded_output对于不是那么长的
        序列，最后的一个实际上是padding位了。
        而hn, cs会自动考虑padding的问题, 输出是最后的实际有效位置的。
        使用实验对比了:
        padded_output[1:2, sample_lengths[1]:sample_lengths[1] + 1, :]与hn[1:2]的结果是相同的
        即手动将padded_output的元素自己通过有效位置取出来，不过用hn直接取要更加方便。
        """
        padded_output, output_lens = \
            torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        fc_input = hn.view((sample_number, -1))
        fc_input = self.dropout(fc_input)
        fc_output = self.fc(fc_input)
        fc2_input = self.relu(fc_output)
        fc2_output = self.fc2(fc2_input)
        sigmoid_output = self.sigmoid(fc2_output)
        return sigmoid_output


print("create model")
embedding_dimision = 8
lstm_hidden_size = 32
loss_func = torch.nn.BCELoss()
variable_net = VariableNet(word_dict.total_token, embedding_dimision, lstm_hidden_size)
optimizer = torch.optim.Adam(variable_net.parameters(), lr=0.01, weight_decay=0.0005)
epochs = 10
batch_size = 32
torch_model_parameters_number(variable_net)

print("partition")
df = df.sample(frac=1.0).reset_index(drop=True)
total_sample, _ = df.shape
partition_num = total_sample / batch_size
sub_df = np.array_split(df, partition_num)
print(len(sub_df))

for i in range(0, epochs):
    total_loss = 0
    print("epoch - {}".format(i))
    batch_num = 0
    for batch_df in sub_df:
        batch_df['sentence_len'] = batch_df['encode'].apply(lambda x: len(x))
        batch_df.sort_values('sentence_len', ascending=False, inplace=True)
        # X is list of variable list.
        X = series_to_list(batch_df['encode'])
        Y = series_to_list(batch_df['label_encode'])
        X_tensor = var_list_to_tensor(X, 0)
        Y_tensor = list_to_tensor(Y)
        sample_number, maxlen = X_tensor.shape
        Y_tensor_reshape = Y_tensor.view((sample_number, -1))
        # 清理上一轮的梯度
        optimizer.zero_grad()
        output = variable_net(X_tensor)
        loss = loss_func(output, Y_tensor_reshape.float())
        print("epoch - {}, batch - {}, loss - {}".format(i, batch_num, loss))
        # for predict, actual in zip(output, Y_tensor_reshape):
        #     print("predict 1 - {}, actual - {}".format(predict, actual))
        loss.backward()
        optimizer.step()
        # output2 = variable_net(X_tensor)
        # loss2 = loss_func(output2, Y_tensor_reshape.float())
        # for predict, actual in zip(output2, Y_tensor_reshape):
        #     print("predict 2 - {}, actual - {}, after loss - {}".format(predict, actual, loss2))
        # print()
        total_loss += loss
        batch_num += 1
    print("epoch - {}, total loss - {}".format(i, total_loss))