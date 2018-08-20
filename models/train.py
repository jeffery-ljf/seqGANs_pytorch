# -*- coding:utf-8 -*-
import time
import torch
import visdom
from discriminator import Discriminator
from generator import Generator
from seqGANs import SEQGANs
import numpy as np
if __name__ == '__main__':
    batch_size = 2
    filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]#判别器的窗口大小（也即每个窗口包含多少个单词）
    num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]#判别器channels数量
    sequence_length = 20#句子的长度
    num_classes = 1#判别器分类类别数量（输出结点数）
    vocab_size = 10000#字典大小
    embedding_size = 100#单词embedding大小
    hidden_size_gru = 100#GRU的隐藏层大小

    vis = visdom.Visdom(port=2424, env='seqGANs')
    seqGANs = SEQGANs(vis).cuda()
    seqGANs.load_state_dict(torch.load('../save/pretrained.pkl'))
    # seqGANs.pretraining()
    # torch.save(seqGANs.state_dict(), '../save/pretrained.pkl')
    seqGANs.backward_D(loss_f='MSE', is_epoch=True)
    for i in range(500):
        time1 = time.time()
        J = seqGANs.backward_G()
        D_loss = seqGANs.backward_D(loss_f='MSE')
        seqGANs.validate_G(i, 'GANs_samples')
        if(i%50==0):
            torch.save(seqGANs.state_dict(), '../save/completed_' + str(i) + '.pkl')
        print('Time: '+str(time.time()-time1))
        print('optimize G epoch ' + str(i) + ': ' + str(J))
        print('optimize D epoch ' + str(i) + ': ' + str(D_loss))
        vis.line(X=torch.cat([torch.tensor([[i]]), torch.tensor([[i]])], 1), Y=torch.cat([torch.unsqueeze(torch.unsqueeze(torch.tensor(J), 0), 0),torch.unsqueeze(torch.unsqueeze(torch.tensor(D_loss), 0), 0)], 1), win='GD_loss',
                      opts=dict(legend=['G_loss','D_loss']),
                      update='append' if i > 0 else None)
    torch.save(seqGANs.state_dict(), '../save/completed_' + str(i) + '.pkl')
    print()
