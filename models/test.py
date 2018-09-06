# -*- coding:utf-8 -*-
import time
import torch
import torch.nn as nn
import visdom
from seqGANs import SEQGANs
from datas.dataset import Quatrains
from torch.utils.data import DataLoader
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
vis = visdom.Visdom(port=2424, env='temp')
seqGANs = SEQGANs().cuda()
seqGANs.load_state_dict(torch.load('../save/completed_500.pkl'))
text_list = seqGANs.dataset.show(seqGANs.show_G())
for i in range(10000):
    time1 = time.time()
    J = seqGANs.backward_G()
    D_loss = seqGANs.backward_D(update=False, loss_f='MSE')
    validate_G(seqGANs, i, 'GANs_samples', vis)
    print(str(i)+': '+str(time.time()-time1))
    print("G_loss: " + str(J))
    print("D_loss: " + str(D_loss))
    vis.line(X=torch.cat([torch.tensor([[i]]), torch.tensor([[i]])], 1), Y=torch.cat(
        [torch.unsqueeze(torch.unsqueeze(torch.tensor(J), 0), 0),
         torch.unsqueeze(torch.unsqueeze(torch.tensor(D_loss), 0), 0)], 1), win='GD_loss',
             opts=dict(legend=['G_loss', 'D_loss']),
             update='append' if i > 0 else None)
print()