# -*- coding:utf-8 -*-
import time
import torch
import visdom
from discriminator import Discriminator
from generator import Generator
from seqGANs import SEQGANs
import numpy as np
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
if __name__ == '__main__':
    vis = visdom.Visdom(port=2424, env='seqGANs')
    seqGANs = SEQGANs().cuda()
    # start_epoch = 4000
    # seqGANs.load_state_dict(torch.load('../save/completed_3999.pkl'))
    for j in range(400):
        total_loss = seqGANs.pretraining()
        validate_G(seqGANs, j, 'pre_samples', vis)
        vis.line(X=torch.tensor([j]), Y=torch.unsqueeze(torch.tensor(total_loss), 0), win='G_pre_loss',
                      opts=dict(legend=['G_pre_loss']), update='append' if j > 0 else None)
        # if j%10==0:
        #     torch.save(seqGANs.state_dict(), '../save_pretrained/pretrained_'+str(j)+'.pkl')
    '''
    # seqGANs.backward_D(loss_f='MSE', is_epoch=True)#carryon的时候要注释掉
    for i in range(start_epoch, start_epoch+6000):
        time1 = time.time()
        J = seqGANs.backward_G()
        D_loss = seqGANs.backward_D(loss_f='MSE')
        validate_G(seqGANs, i, 'GANs_samples', vis)
        if(i%50==0):
            torch.save(seqGANs.state_dict(), '../save/completed_' + str(i) + '.pkl')
        print('Time: '+str(time.time()-time1))
        print('optimize G epoch ' + str(i) + ': ' + str(J))
        print('optimize D epoch ' + str(i) + ': ' + str(D_loss))
        vis.line(X=torch.cat([torch.tensor([[i]]), torch.tensor([[i]])], 1), Y=torch.cat([torch.unsqueeze(torch.unsqueeze(torch.tensor(J), 0), 0),torch.unsqueeze(torch.unsqueeze(torch.tensor(D_loss), 0), 0)], 1), win='GD_loss',
                      opts=dict(legend=['G_loss','D_loss']),
                      update='append' if i > 0 else None)
    torch.save(seqGANs.state_dict(), '../save/completed_' + str(i) + '.pkl')
    '''
    print()
