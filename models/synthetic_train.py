# -*- coding:utf-8 -*-
import time
import torch
import visdom
from discriminator import Discriminator
from generator import Generator
from synthetic_seqGANs import SEQGANs_syn
from oracle import Oracle
from datas.dataset import DataSet_Gen
import numpy as np
from torch.utils.data import DataLoader
def output_nll(seqGANs, oracle):
    seqGANs = seqGANs.cpu()
    # 利用生成器生成数据，构建dataloader
    data_gen = seqGANs.generate_data()
    dataset_gen = DataSet_Gen(data_gen)
    dataloader_gen = DataLoader(dataset_gen, batch_size=seqGANs.batch_size, shuffle=False, num_workers=2)
    nll = oracle.NLL(dataloader=dataloader_gen,
                     start_input=seqGANs.start_input,
                     start_h=seqGANs.start_h)
    print('nll: ' + str(nll))
    seqGANs = seqGANs.cuda()
    return nll
if __name__ == '__main__':
    vis = visdom.Visdom(port=2424, env='seqGANs-syn')
    seqGANs = SEQGANs_syn().cuda()
    oracle = Oracle().cuda()
    oracle.load_state_dict(torch.load('../datas/Synthetic_Data/oracle.pkl'))
    start_epoch = 0
    seqGANs.load_state_dict(torch.load('../syn_save_pretrained/pretrained_290.pkl'))
    '''
    for j in range(300):
        total_loss = seqGANs.pretraining()
        vis.line(X=torch.tensor([j]), Y=torch.unsqueeze(torch.tensor(total_loss), 0), win='G_pre_loss',
                      opts=dict(legend=['G_pre_loss']), update='append' if j > 0 else None)
        if j%10==0:
            torch.save(seqGANs.state_dict(), '../syn_save_pretrained/pretrained_'+str(j)+'.pkl')
        if j%5==0:
            nll = output_nll(seqGANs=seqGANs, oracle=oracle)
            vis.line(X=torch.tensor([j]), Y=torch.unsqueeze(torch.tensor(nll), 0), win='nll',
                     opts=dict(legend=['nll']), update='append' if j > 0 else None)
    '''
    seqGANs.backward_D(loss_f='MSE', is_epoch=True)#carryon的时候要注释掉
    for i in range(start_epoch, start_epoch+3000):
        time1 = time.time()
        J = seqGANs.backward_G()
        D_loss = seqGANs.backward_D(loss_f='MSE')
        if(i%5==0):
            torch.save(seqGANs.state_dict(), '../syn_save/completed_' + str(i) + '.pkl')
        print('Time: '+str(time.time()-time1))
        print('optimize G epoch ' + str(i) + ': ' + str(J))
        print('optimize D epoch ' + str(i) + ': ' + str(D_loss))
        vis.line(X=torch.cat([torch.tensor([[i]]), torch.tensor([[i]])], 1), Y=torch.cat([torch.unsqueeze(torch.unsqueeze(torch.tensor(J), 0), 0),torch.unsqueeze(torch.unsqueeze(torch.tensor(D_loss), 0), 0)], 1), win='GD_loss',
                      opts=dict(legend=['G_loss','D_loss']),
                      update='append' if i > 0 else None)
        if i % 5 == 0:
            nll = output_nll(seqGANs=seqGANs, oracle=oracle)
            vis.line(X=torch.tensor([i]), Y=torch.unsqueeze(torch.tensor(nll), 0), win='nll_adv',
                     opts=dict(legend=['nll_adv']), update='append' if i > 0 else None)
    torch.save(seqGANs.state_dict(), '../syn_save/completed_' + str(i) + '.pkl')
    print()