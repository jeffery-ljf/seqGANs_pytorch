# -*- coding:utf-8 -*-
import torch
from datas.dataset import Quatrains
from torch.utils.data import DataLoader
# prob = torch.Tensor([[0, 0.2, 0.5, 0.3],[0.1, 0.2, 0.4, 0.3]])
# samples = torch.Tensor([[0., 0., 0., 0.],[0., 0., 0., 0.]])
# for i in range(5000):
#     sample = torch.multinomial(prob, 1)
#     samples[0][sample[0]]+=1
#     samples[1][sample[1]] += 1
# print(samples/5000)


# dataset = Quatrains(root_src=r'../datas/rnnpg_data_emnlp-2014/realdata_20Chars/data')
# dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=2)
# for epoch in range(5):
#     for i_batch, x_batch in enumerate(dataloader):
#         print(dataset.show(x_batch))

# a = torch.LongTensor(3, 2, 5).zero_()
# label = torch.LongTensor([[[3], [4]],
#                           [[0], [2]],
#                           [[1], [4]]])
# b = a.scatter_(dim=2, index=label, value=1)
# print(b)
# def onehot(self, label):
torch.tensor(61 * [0]).cuda()
print()