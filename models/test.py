# -*- coding:utf-8 -*-
import time
import torch
import torch.nn as nn
import visdom
from seqGANs import SEQGANs
from datas.dataset import Quatrains
from torch.utils.data import DataLoader
from datas.dictionary import Dictionary_Quatrains, Dictionary_Obama
def validate_G(model, epoch, win, vis):
    text_list = model.dataset.show(model.show_G())
    text = str(epoch) + ': '
    for b in range(model.batch_size):
        text += '<h5>'
        for word in text_list:
            if word[b] != '<R>':
                text += word[b] + ' '
            else:
                text += 'R' + ' '
        text += '</h5>'
    vis.text(text=text, win=win)
'''
output_layer = nn.Sequential(*[nn.Linear(in_features=100, out_features=1, bias=True),
                                    nn.Sigmoid()])
optimizer = torch.optim.Adam([
            {'params': output_layer.parameters()}
        ])

for i in range(500):
    optimizer.zero_grad()
    l2_loss = torch.tensor(0.)
    for param in output_layer.parameters():
        l2_loss += torch.norm(param, p=2)
    l2_loss.backward()
    optimizer.step()
    print(l2_loss)
'''
# vis = visdom.Visdom(port=2424, env='temp')
# seqGANs = SEQGANs().cuda()
# seqGANs.load_state_dict(torch.load('../save/completed_450.pkl'))
# text_list = seqGANs.dataset.show(seqGANs.show_G())
# validate_G(seqGANs, 0, 'GANs_samples', vis)