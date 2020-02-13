# encoding=utf8

import numpy as np
import csv


class HMM(object):
    def __init__(self, real_state_num, observation_state_num):
        # 隐藏状态转移概率矩阵
        self.real_state_trans_matrix = np.zeros((real_state_num, real_state_num))
        # 观测概率矩阵
        self.real_to_observation_matrix = np.zeros((real_state_num, observation_state_num))
        # 初始状态概率矩阵
        self.initial_matrix = np.array([1.0 / real_state_num] * real_state_num)
        # 可能的状态数
        self.real_state_num = real_state_num
        # 可能的观测数
        self.observation_state_num = observation_state_num

    # 随机生成 real_state_trans_matrix，real_to_observation_matrix，initial_matrix并保证每行相加等于1
    def init(self):
        import random
        for i in range(self.real_state_num):
            randomlist = [random.randint(0, 100) for t in range(self.real_state_num)]
            Sum = sum(randomlist)
            for j in range(self.real_state_num):
                self.real_state_trans_matrix[i][j] = float(randomlist[j]) / Sum
        for i in range(self.real_state_num):
            randomlist = [random.randint(0,100) for t in range(self.observation_state_num)]
            Sum = sum(randomlist)
            for j in range(self.observation_state_num):
                self.real_to_observation_matrix[i][j] = float(randomlist[j]) / Sum

    def cal_probality(self, O):
        self.T = len(O)
        self.O = O
        self.forward()
        return sum(self.alpha[self.T-1])

    # 前向概率的定义：定义时刻t观察序列为O[0...t]，隐藏状态为qi为前向概率
    def forward(self):
        # O是观察状态， T是观察状态的个数
        self.alpha = np.zeros((self.T, self.real_state_num))
        # 计算初始隐藏状态为某个状态的概率，这些都是建立在观察状态已知的情况下
        for i in range(self.real_state_num):
            # 初始隐藏状态为i的概率且观察状态为O[0] = 初始概率i出现的情况 * 表现状态为O[0]时是i状态的概率
            self.alpha[0][i] = self.initial_matrix[i] * self.real_to_observation_matrix[i][self.O[0]]
        # 计算时间序列为t时
        for t in range(1, self.T):
            # 隐藏状态为i的概率
            for i in range(self.real_state_num):
                sum = 0
                for j in range(self.real_state_num):
                    # t时刻隐藏状态为i，可能有前面的任意隐藏状态转化而来，所以要概率要全部加上
                    sum += self.alpha[t-1][j]*self.real_state_trans_matrix[j][i]
                # t时刻隐藏状态为i的概率且观察状态为O[t]的概率 = t时刻隐藏状态为i * 观察状态为O[t]时是i状态的概率
                self.alpha[t][i] = sum * self.real_to_observation_matrix[i][self.O[t]]

    # 后向概率的定义：定义时刻t状态为qi的条件下，t+1到T的部分观测序列为O[t + 1...T]的概率为后向概率
    def backward(self):
        self.beta = np.zeros((self.T, self.real_state_num))

        # 对最终时刻为qi规定所有的概率都等于1
        for i in range(self.real_state_num):
            self.beta[self.T - 1][i] = 1

        # 要计算时刻t状态为qi的条件下，t+1到T的部分观测序列为O[t+1...T]的概率， 根据时刻t+1的各个隐藏状态为qj, t+2到T观测状态为
        # O[t + 2...T]之和，然后发生了观察状态为O[t+1]
        for t in range(self.T - 2, -1, -1):
            # t相当于从self.T -2 一直到0
            for i in range(self.real_state_num):
                # i表示t时刻状态为qi的概率
                for j in range(self.real_state_num):
                    # j状态由i状态转移过去 * 隐藏状态为j状态时表现为O[t+1]的概率
                    self.beta[t][i] += self.real_state_trans_matrix[i][j] * \
                                       self.real_to_observation_matrix[j][self.O[t + 1]] \
                                       * self.beta[t+1][j]

    def cal_gamma(self, i, t):
        # 10.24
        # 给定模型参数和观测O，在时刻t处于qi的概率
        numerator = self.alpha[t][i] * self.beta[t][i]
        denominator = 0
        for j in range(self.real_state_num):
            denominator += self.alpha[t][j] * self.beta[t][j]

        return float(numerator) / denominator

    # 给定模型参数和观测O，在时刻t处于状态qi在时刻t+1处于状态qj的概率
    def cal_ksi(self, i, j, t):
        # 10.26
        numerator = self.alpha[t][i] * self.real_state_trans_matrix[i][j] * \
                    self.real_to_observation_matrix[j][self.O[t + 1]] * self.beta[t + 1][j]
        denominator = 0
        for i in range(self.real_state_num):
            for j in range(self.real_state_num):
                denominator += self.alpha[t][i] * self.real_state_trans_matrix[i][j] * \
                               self.real_to_observation_matrix[j][self.O[t + 1]] * self.beta[t + 1][j]
        return float(numerator) / denominator

    # Baum-Welch算法：
    # 输入观测数据O，输出隐马尔克夫模型参数
    def train(self, O, observation_state_numax_steps = 100):
        self.T = len(O)
        self.O = O
        self.init()
        step = 0
        # 迭代求精
        while step < observation_state_numax_steps:
            step += 1
            print(step)
            tmp_real_state_trans_matrix = np.zeros((self.real_state_num, self.real_state_num))
            tmp_real_to_observation_matrix = np.zeros((self.real_state_num, self.observation_state_num))
            temp_initial_matrix = np.array([0.0] * self.real_state_num)
            # 注意这里计算前向和后向概率时，观察序列O都已经定下来了。
            self.forward()
            self.backward()
            # 隐藏状态转移矩阵估计aij
            for i in range(self.real_state_num):
                for j in range(self.real_state_num):
                    numerator = 0.0
                    denominator = 0.0
                    for t in range(self.T - 1):
                        numerator += self.cal_ksi(i, j, t)
                        denominator += self.cal_gamma(i, t)
                    tmp_real_state_trans_matrix[i][j] = numerator / denominator

            # 观测发射矩阵估计bj(k)
            for j in range(self.real_state_num):
                for k in range(self.observation_state_num):
                    numerator = 0.0
                    denominator = 0.0
                    for t in range(self.T):
                        if k == self.O[t]:
                            numerator += self.cal_gamma(j, t)
                        denominator += self.cal_gamma(j, t)
                    tmp_real_to_observation_matrix[j][k] = numerator / denominator

            # 初始状态矩阵估计pi
            for i in range(self.real_state_num):
                temp_initial_matrix[i] = self.cal_gamma(i, 0)

            self.real_state_trans_matrix = tmp_real_state_trans_matrix
            self.real_to_observation_matrix = tmp_real_to_observation_matrix
            self.initial_matrix = temp_initial_matrix

    # 这个generate函数的原理是什么？看起来也不像是预测算法中的维特比算法/近似算法
    def generate(self, length):
        import random
        I = []
        # start
        ran = random.randint(0, 1000)/1000.0
        i = 0
        while self.initial_matrix[i] < ran or self.initial_matrix[i] < 0.0001:
            ran -= self.initial_matrix[i]
            i += 1
        I.append(i)
        # 生成状态序列
        for i in range(1, length):
            last = I[-1]
            ran = random.randint(0, 1000) / 1000.0
            i = 0
            while self.real_state_trans_matrix[last][i] < ran or self.real_state_trans_matrix[last][i] < 0.0001:
                ran -= self.real_state_trans_matrix[last][i]
                i += 1
            I.append(i)
        # 生成观测序列
        Y = []
        for i in range(length):
            k = 0
            ran = random.randint(0, 1000) / 1000.0
            while self.real_to_observation_matrix[I[i]][k] < ran or self.real_to_observation_matrix[I[i]][k] < 0.0001:
                ran -= self.real_to_observation_matrix[I[i]][k]
                k += 1
            Y.append(k)
        return Y


def triangle(length):
    '''
    三角波
    '''
    X = [i for i in range(length)]
    Y = []

    for x in X:
        x = x % 6
        if x <= 3:
            Y.append(x)
        else:
            Y.append(6-x)
    return X,Y


def sin(length):
    '''
    三角波
    '''
    import math
    X = [i for i in range(length)]
    Y = []

    for x in X:
        x = x % 20
        Y.append(int(math.sin((x*math.pi)/10)*50)+50)
    return X,Y


def show_data(x,y):
    import matplotlib.pyplot as plt
    plt.plot(x, y, 'g')
    plt.show()
    return y


# 这个main函数其实是只给出观测序列，然后估算hmm的参数。接着使用hmm的参数来生成更长的O。
# 但是generate的那个函数好像没有严格的算法定义，是怎么来的？
if __name__ == '__main__':
    hmm = HMM(10, 4)
    # tri_x是隐含状态， tri_y是观察状态？triangle函数只会生成[0, 1, 2, 3]中的任意组合
    tri_x, tri_y = triangle(20)
    # hmm的学习问题，根据观测序列学习hmm的参数，使得P(O|hmm参数)最大
    hmm.train(tri_y)
    # 感觉是预测问题，但是预测问题应该还会给出原始的观测序列才对
    y = hmm.generate(100)
    x = [i for i in range(100)]
    show_data(tri_x, tri_y)
    show_data(x, y)

    # hmm = HMM(15,101)
    # sin_x, sin_y = sin(40)
    # show_data(sin_x, sin_y)
    # hmm.train(sin_y)
    # y = hmm.generate(100)
    # x = [i for i in range(100)]
    # show_data(x,y)

