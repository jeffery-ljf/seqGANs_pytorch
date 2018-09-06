# -*- coding:utf-8 -*-
from oracle import Oracle
from torch import nn
import torch
import pickle
model = Oracle()
model.init_params()
fw = open(r'../datas/Synthetic_Data/data.pkl','wb')
data = model.synthesize_data()
pickle.dump(data, fw)
torch.save(model.state_dict(), '../datas/Synthetic_Data/oracle.pkl')
print(data.shape)