import torch
import math


# 调用库计算
loss_func = torch.nn.CrossEntropyLoss(reduction='none')
predict = torch.FloatTensor([
    [0.1, 0.2, 0, 0.9],
    [0.2, 0.2, 0.8, 0.3]
])
target = torch.LongTensor([3, 2])
print(loss_func(predict, target))
loss_result = []
for index, target_value in enumerate(target):
    x_class = predict[index][target_value]
    e_sum = 0
    for class_score in predict[index]:
        e_sum += math.exp(class_score)
    log_e_sum = math.log(e_sum)
    cur_loss = - x_class + log_e_sum
    loss_result.append(cur_loss)
print(loss_result)

predict_soft_max = torch.nn.functional.softmax(predict, dim=1)
print(predict_soft_max)
loss_result2 = []
for index, target_value in enumerate(target):
    cur = 0
    for class_score, softmax_score in zip(predict[index], predict_soft_max[index]):
        cur += class_score * math.log(softmax_score)
    loss_result2.append(-cur)
print(loss_result2)