# -*- coding:utf-8 -*-
from torch.utils.data import Dataset
from datas.dictionary import Dictionary_Quatrains, Dictionary_Obama
import pickle
import torch
import numpy as np
class Quatrains(Dataset):
    def __init__(self, root_src, start_idx, end_idx, padding_idx):
        '''
        :param root_src: src with all quatrains
        :param padding_idx: idx of padding token
        '''
        self.root_src = root_src
        self.dictionary = Dictionary_Quatrains(root_src, start_idx, end_idx, padding_idx)
        self.dictionary.build()
        self.max_doclen = self.get_maxlen()
        data = []
        for doc in self.dictionary.data:
            doc_data = []
            for word in doc:
                doc_data.append(self.dictionary.word2id[word])
            doc_data.append(end_idx)
            while doc_data.__len__() != self.max_doclen+1:#为doc_data补全1，直到所有句子的最大长度+1为止。
                doc_data.append(padding_idx)
            data.append(doc_data)
        self.data = torch.tensor(data)#data的每一行是由原句+end token的id组成的，start token不在里面。
        print('Reading data is fine !')
    def get_maxlen(self):#获得data中所有句子的最大长度
        words_count = []
        for doc in self.dictionary.data:
            words_count.append(doc.__len__())
        words_count = np.array(words_count)
        return int(words_count.max())
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
class DataSet_Obama(Dataset):
    def __init__(self, root_src, start_idx, end_idx, padding_idx):
        '''
        :param root_src: src with all quatrains
        :param padding_idx: idx of padding token
        '''
        self.root_src = root_src
        self.dictionary = Dictionary_Obama(root_src, start_idx, end_idx, padding_idx)
        self.dictionary.build()
        self.max_doclen = self.get_maxlen()
        data = []
        for doc in self.dictionary.data:
            doc_data = []
            for word in doc:
                doc_data.append(self.dictionary.word2id[word])
            doc_data.append(end_idx)
            while doc_data.__len__() != self.max_doclen+1:#为doc_data补全1，直到所有句子的最大长度+1为止。
                doc_data.append(padding_idx)
            data.append(doc_data)
        self.data = torch.tensor(data)#data的每一行是由原句+end token的id组成的，start token不在里面。
        print('Reading data is fine !')
    def get_maxlen(self):#获得data中所有句子的最大长度
        words_count = []
        for doc in self.dictionary.data:
            words_count.append(doc.__len__())
        words_count = np.array(words_count)
        return int(words_count.max())
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
class DataSet_Syn(Dataset):
    def __init__(self, root_src):
        '''
        :param root_src: src with all datas
        '''
        self.root_src = root_src
        with open(self.root_src,'rb') as file:
            self.data = pickle.load(file)
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, item):
        return self.data[item]
class DataSet_Gen(Dataset):
    def __init__(self, data):
        '''
        :param data: data
        '''
        self.data = data
    def __len__(self):
        return self.data.shape[0]
    def __getitem__(self, item):
        return self.data[item]