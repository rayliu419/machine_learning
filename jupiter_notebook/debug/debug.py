"""
see https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation.ipynb.

"""

import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
# %matplotlib inline
import sys
sys.path.append("../common/")
import english2franch_helper
import time_utils

MAX_LENGTH = 10
teacher_forcing_ratio = 0.5
clip = 5.0
SOS_token = 0
EOS_token = 1


"""
将word变成encoder/decoder的输入
"""
def indexes_from_sentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variable_from_sentence(lang, sentence):
    indexes = indexes_from_sentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    return var


def variables_from_pair(pair):
    input_variable = variable_from_sentence(input_lang, pair[0])
    target_variable = variable_from_sentence(output_lang, pair[1])
    return (input_variable, target_variable)


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1):
        super(EncoderRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        """
        num_embeddings - 嵌入字典的大小
        embedding_dim - 每个嵌入向量的大小
        """
        self.embedding = nn.Embedding(input_size, hidden_size)
        """
         input_size – 期望的输入x的特征值的维度 
         hidden_size – 隐状态的维度 
         num_layers – RNN的层数
        """
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers)

    def forward(self, word_inputs, hidden):
        """
        :param word_inputs: 是一个(seq, 1)的序列
        :param hidden:
        :return:
        """
        seq_len = len(word_inputs)
        """
        nn.embedding的输入 - LongTensor (N, W), 
            N - mini-batch的sample个数, 
            W - 序列长度
        一维tensor也可以接受？实际上用的时候更灵活，只要按照自己的语义来做传入就行，例如像本例，其实是传入的是(seq, 1)，reshape的时候
        将输入变成了GRU要求的形式。
        nn.embedding的输出 - (N, W, embedding_dim)
            N,W - 跟上面的一样
            embedding_dim - embedding的维度
        """
        embedded = self.embedding(word_inputs)
        # 变成GRU要求的形式。
        embedded_reshape = embedded.view(seq_len, 1, -1)
        """
        nn.GRU的输入 - input, h_0
            input(seq_len, batch, embedding_dim)
            h_0(num_layers * num_directions, batch, hidden_size) - 隐含状态，一般来说num_layers=1，num_directions=2是bi。
        nn.GRU的输出 - output, h_n
            output(seq_len, batch, hidden_size * num_directions) - 这个输出有点搞不懂
            h_n(num_layers * num_directions, batch, hidden_size) - 与输入相同
        """
        # 注意，对于encoder端，我们一次性将encoder的输入一次forward处理了。结果返回了seq个output和最后的hidden state。
        output, hidden = self.gru(embedded_reshape, hidden)
        return output, hidden

    def init_hidden(self):
        # encoder的初始hidden state为全0
        hidden = Variable(torch.zeros(self.n_layers, 1, self.hidden_size))
        return hidden


class Attn(nn.Module):
    def __init__(self, method, hidden_size, max_length=MAX_LENGTH):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.other = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = len(encoder_outputs)
        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(seq_len))
        # 计算相关性，有好几种计算方法，可以认为是每个word对应一个权重
        for i in range(seq_len):
            attn_energies[i] = self.score(hidden, encoder_outputs[i])
        # resize to 1 x 1 x seq_len
        return F.softmax(attn_energies, dim=0).unsqueeze(0).unsqueeze(0)

    def score(self, hidden, encoder_output):
        if self.method == 'dot':
            # 我估计这里也得像'general'那样改
            energy = hidden.dot(encoder_output)
            return energy
        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.view(-1).dot(energy.view(-1))
            return energy
        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.other.dot(energy)
            return energy


class AttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        """
         output_size – 词表大小 
         hidden_size – embedding的大小 
        """
        self.embedding = nn.Embedding(output_size, hidden_size)
        """
         期望的输入x的特征值的维度 
         隐状态的维度 
         RNN的层数
         为什么是hidden_size * 2呢，因为context也作为输入 - 待删除
        """
        # self.gru = nn.GRU(hidden_size * 2, hidden_size, n_layers, dropout=dropout_p)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size * 2, output_size)
        # Choose attention model
        if attn_model != 'none':
            self.attn = Attn(attn_model, hidden_size)

    # 实现的是Luong Attention的global版本
    # https://blog.floydhub.com/attention-mechanism/
    # 步骤是：
    # 1. 收集encoder的所有输出，作为context向量计算的源头之一。
    # 2. 前面decoder的hidden和输出传入一个decoder的RNN组件，为当前的time step产生一个新的output/hidden
    # 3. 计算Alignment Scores。使用新的output和encoder的所有output算分。有好几种方式。
    # 4. Softmax分数。
    # 5. 使用encoder的输出和计算出来的分数计算context向量。
    # 6. concat context向量和在第二步中生成的output作为输入传如到一个新的DNN并得到输出。
    # 7. 返回重复过程，直到最大长度。
    # 上述整个描述复合整个forward函数，除了使用了last_context，把整个去掉看看。
    def forward(self, word_input, last_context, last_hidden, encoder_outputs):
        """
        对于decoder，我们是一个单词一个单词的处理的，因为要加入attention
        感觉这个实现也不完全是按照Luong Attention的global方式做的？
        layer=1的情况
        :param word_input: 2维的，每次一个单词的int表示，例如<SOS_token>是[[0]]
        :param last_context: 2维，(1, hidden_size)
        :param last_hidden: 3维，(1, 1, hidden_size)
        :param encoder_outputs: 3维, (seq_len, 1, hidden_state)
        :return:
        """
        # 获取输入的embedding
        word_embedded = self.embedding(word_input).view(1, 1, -1) # S= 1 x B x N
        # 组合embedding input和last_context向量，这里的rnn_input为什么用到了last_context? - 拿掉last_context看看
        # rnn_input = torch.cat((word_embedded, last_context.unsqueeze(0)), 2)
        rnn_input = word_embedded
        rnn_output, hidden = self.gru(rnn_input, last_hidden)
        """
        计算attention的分数。先拿到了rnn_output，然后计算它与前面的encoder_output的相关性。
        调用的是Attn的forward()函数
        """
        attn_weights = self.attn(rnn_output.squeeze(0), encoder_outputs)
        # 这个bmm是做了什么啊？- 就是在计算context向量的矩阵运算方法
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x 1 x N
        # Final output layer (next word prediction) using the RNN hidden state and context vector
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        output = F.log_softmax(self.out(torch.cat((rnn_output, context), 1)), dim=1)
        # Return final output, hidden state, and attention weights (for visualization)
        return output, context, hidden, attn_weights


def train_single(input_variable, target_variable, encoder, decoder,
                 encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    """
    为了训练，我们将输入的句子一个个word进入，并且保存每个输出和最近的hidden state，decoder将encoder的最后一个hidden  state作为初始的hidden state，<SOS>token作为第一个输入。
    交替使用teacher forcing和decoder本身预测的输出作为下一个输入。
    """
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0
    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    """
    encoder的初始hidden state是[0, 0, 0...]
    encoder_outputs 作为decoder计算attention
    返回的encoder_hidden作为decoder的初始hidden state，这个hidden state是3维的 
    """
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    # SOS_token作为第一个输入
    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    # last hidden state from encoder to start decoder
    decoder_hidden = encoder_hidden
    use_teacher_forcing = random.random() < teacher_forcing_ratio
    if use_teacher_forcing:
        # Teacher forcing: Use the ground-truth target as the next input
        for di in range(target_length):
            # 调用AttnDecoderRNN的forward()函数
            decoder_output, decoder_context, decoder_hidden, decoder_attention = \
                decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di] # Next target is next input
    else:
        # Without teacher forcing: use network's own prediction as the next input
        for di in range(target_length):
            decoder_output, decoder_context, decoder_hidden, decoder_attention = \
                decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            # 使用概率最高的decoder的输出作为下一步的输入
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]
            decoder_input = Variable(torch.LongTensor([[ni]])) # Chosen word is next input
            # Stop at end of sentence (not necessary when using known targets)
            if ni == EOS_token: break
    # 这里说明整个句子被统一计算loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)
    encoder_optimizer.step()
    decoder_optimizer.step()
    return loss / target_length


print("#0 read data")
input_lang, output_lang, pairs = english2franch_helper.prepare_translation_raw_data_for_task()
print(random.choice(pairs))

print("#1 define parameters")
attn_model = 'general'
hidden_size = 128
# n_layers = 2
n_layers = 2
dropout_p = 0.05

encoder = EncoderRNN(input_lang.n_words, hidden_size, n_layers)
decoder = AttnDecoderRNN(attn_model, hidden_size, output_lang.n_words, n_layers, dropout_p=dropout_p)

learning_rate = 0.0001
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
criterion = nn.NLLLoss()
n_epochs = 40000
plot_every = 200
print_every = 500
start = time.time()
plot_losses = []
print_loss_total = 0
plot_loss_total = 0

print("#2 start to train")
for epoch in range(1, n_epochs + 1):
    training_pair = variables_from_pair(random.choice(pairs))
    # input_variable, target_variable是二维的
    input_variable = training_pair[0]
    target_variable = training_pair[1]
    loss = train_single(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    print_loss_total += loss
    plot_loss_total += loss
    if epoch == 0: continue
    if epoch % print_every == 0:
        print_loss_avg = print_loss_total / print_every
        print_loss_total = 0
        print_summary = '%s (%d %d%%) %.4f' % (time_utils.time_since(start, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg)
        print(print_summary)
    if epoch % plot_every == 0:
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0


def show_plot(points):
    plt.figure()
    fig, ax = plt.subplots()
    loc = ticker.MultipleLocator(base=0.2) # put ticks at regular intervals
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


show_plot(plot_losses)


def evaluate(sentence, max_length=MAX_LENGTH):
    """
    跟train_single的逻辑是类似的。
    """
    input_variable = variable_from_sentence(input_lang, sentence)
    input_length = input_variable.size()[0]
    encoder_hidden = encoder.init_hidden()
    encoder_outputs, encoder_hidden = encoder(input_variable, encoder_hidden)
    decoder_input = Variable(torch.LongTensor([[SOS_token]])) # SOS
    decoder_context = Variable(torch.zeros(1, decoder.hidden_size))
    decoder_hidden = encoder_hidden
    decoded_words = []
    decoder_attentions = torch.zeros(max_length, max_length)
    for di in range(max_length):
        decoder_output, decoder_context, decoder_hidden, decoder_attention = \
            decoder(decoder_input, decoder_context, decoder_hidden, encoder_outputs)
        decoder_attentions[di, :decoder_attention.size(2)] += \
            decoder_attention.squeeze(0).squeeze(0).cpu().data
        topv, topi = decoder_output.data.topk(1)
        ni = topi.numpy()[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])
        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([[ni]]))
    return decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]


def evaluate_randomly():
    pair = random.choice(pairs)
    output_words, decoder_attn = evaluate(pair[0])
    output_sentence = ' '.join(output_words)
    print('>', pair[0])
    print('=', pair[1])
    print('<', output_sentence)
    print('')


print("#3 random sample")
evaluate_randomly()


def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)
    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)
    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()
    plt.close()


def evaluate_and_show_attention(input_sentence, target_sentence):
    output_words, attentions = evaluate(input_sentence)
    print('input =', input_sentence)
    print('target=', target_sentence)
    print('output =', ' '.join(output_words))
    show_attention(input_sentence, output_words, attentions)


print("#4 show attention")
pair = random.choice(pairs)
evaluate_and_show_attention(pair[0], pair[1])

pair = random.choice(pairs)
evaluate_and_show_attention(pair[0], pair[1])



