# -*- coding:utf-8 -*-
from torch.utils.data import Dataset
from datas.dictionary import Dictionary
import torch
import numpy as np
class Quatrains(Dataset):
    def __init__(self, root_src):
        '''

        :param root_src: src with all quatrains
        '''
        self.root_src = root_src
        self.dictionary = Dictionary(root_src)
        self.dictionary.build()
        data = []
        for doc in self.dictionary.data:
            doc_data = []
            for word in doc:
                doc_data.append(self.dictionary.word2id[word])
            data.append(doc_data)
        self.data = torch.tensor(data)
        print('Reading data is fine !')
    def __len__(self):
        return self.data.__len__()
    def show(self, data):
        #展现数据的原始样貌
        result = []
        for doc in data:
            result.append([self.dictionary.id2word[word] for word in doc])
        return result
    def __getitem__(self, item):
        return self.data[item]
