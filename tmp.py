'''
@author: jinglingzhiyu
'''
import torch
import torch.nn as nn
import numpy as np



percent = np.linspace(0, 100, 101)
print(percent)


# def my_loss(preds, labels):
#     loss, loss_ = 0.0, 0.0
#     for i in range(preds.shape[1]):
#         pred = preds[:, i]
#         loss += torch.exp(pred)
#     for i in range(preds.shape[1]):
#         pred = preds[:, i]
#         loss_ += labels[:, i] * torch.exp(pred)
#     loss = loss_ / loss
#     loss = - torch.log(loss)
#     return loss
# 
# a = torch.Tensor([0.1, 0.2, 0.3]).unsqueeze(0)
# # b = torch.Tensor([1]).long()
# # cr = nn.CrossEntropyLoss()
# b = torch.Tensor([0.1, 0.7, 0.2]).unsqueeze(0)
# c = my_loss(a,b)
# print(c)

# b = np.random.rand()
# nums = 3
# b = int(b * nums)
# print(b)

# b = torch.sparse.torch.eye(8)
# a = torch.ones(10).long()
# b = b.index_select(0, a)
# print(b.shape)
# print(b)

# b = [int(x) for x in np.linspace(0, 10, 4)]
# print(b)