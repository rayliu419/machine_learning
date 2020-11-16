import torch

tensor1 = torch.Tensor([[0.4803],
        [0.9],
        [0.4906],
        [0.4804],
        [0.4925],
        [0.4789],
        [0.4760],
        [0.4803],
        [0.4786],
        [0.4727],
        [0.4827],
        [0.4727],
        [0.4789],
        [0.4749],
        [0.4778],
        [0.4811]])

tensor2 = torch.Tensor([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
tensor2_reshape = tensor2.view((16, 1))

loss_function = torch.nn.BCELoss()

loss = loss_function(tensor1, tensor2_reshape)
print(loss)