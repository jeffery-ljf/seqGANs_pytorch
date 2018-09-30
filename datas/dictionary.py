# -*- coding:utf-8 -*-
import numpy as np
class Dictionary_Quatrains():
    def __init__(self, src, start_idx, end_idx, padding_idx):
        data = open(src).readlines()
        #应该把数据处理成doc*word的列表
        self.data = []#存放原始数据
        self.id2word = []
        self.word2id = {}
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.padding_idx = padding_idx
        for doc in data:
            temp = []
            lines = doc.replace('\n', '').split('\t')
            for line in lines:
                for word in line.split():
                    temp.append(word)
            self.data.append(temp)
    def __len__(self):
        if self.id2word.__len__()==self.word2id.__len__():
            return self.id2word.__len__()
        else:
            raise RuntimeError('dictionary size error')
    def build(self):
        temp = []#存放单词，唯一
        wordcount = []#存放单词以及它的频数
        for doc in self.data:
            for word in doc:
                if word in temp:
                    id = temp.index(word)#找出索引
                    if wordcount[id][0]==temp[id]:#计算频数
                        wordcount[id][1] = wordcount[id][1]+1
                    else:
                        raise RuntimeError('dictionary operation error')
                else:
                    temp.append(word)
                    wordcount.append([word,0])

        wordcount.sort(key=lambda item: item[1], reverse=True)#按照频数排序
        self.id2word.append('start token')#加入start token
        self.word2id['start token'] = self.start_idx#为start token安排id
        self.id2word.append('end token')#加入end token
        self.word2id['end token'] = self.end_idx#为end token安排id
        self.id2word.append('padding token')  # 加入padding token
        self.word2id['padding token'] = self.padding_idx  # 为padding token安排id
        for id in range(3, wordcount.__len__()+3):
            word = wordcount[id-3][0]
            self.id2word.append(word)
            self.word2id[word] = id
        print('Successfully building the dictionary !')
class Dictionary_Obama():
    def __init__(self, src, start_idx, end_idx, padding_idx):
        data = open(src).readlines()
        #应该把数据处理成doc*word的列表
        self.data = []#存放原始数据
        self.id2word = []
        self.word2id = {}
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.padding_idx = padding_idx
        # point = [0, 0, 0, 0, 0, 0]#记录句子长度的分界线
        for doc in data:
            if doc!='\n':
                line = doc.replace('\n', '').replace(',', ' ,').replace('.', ' .').replace(';', ' ;').replace(':', ' :')\
                    .replace('?', ' ?').replace('  ', ' ').replace('"', '').split(' ')#处理每一行，包括把标点符号分开。
                if line.__len__()>=4 and line.__len__()<=100:
                    self.data.append(line)
                # for i in range(point.__len__()):
                #     if(line.__len__()>i*4 and line.__len__()<=(i+1)*4):
                #         point[i] += 1
    def __len__(self):
        if self.id2word.__len__()==self.word2id.__len__():
            return self.id2word.__len__()
        else:
            raise RuntimeError('dictionary size error')
    def build(self):
        temp = []#存放单词，唯一
        wordcount = []#存放单词以及它的频数
        for doc in self.data:
            for word in doc:
                if word in temp:
                    id = temp.index(word)#找出索引
                    if wordcount[id][0]==temp[id]:#计算频数
                        wordcount[id][1] = wordcount[id][1]+1
                    else:
                        raise RuntimeError('dictionary operation error')
                else:
                    temp.append(word)
                    wordcount.append([word,0])

        wordcount.sort(key=lambda item: item[1], reverse=True)#按照频数排序
        self.id2word.append('start token')#加入start token
        self.word2id['start token'] = self.start_idx#为start token安排'0'id
        self.id2word.append('end token')#加入end token
        self.word2id['end token'] = self.end_idx#为end token安排'1'id
        self.id2word.append('padding token')  # 加入padding token
        self.word2id['padding token'] = self.padding_idx  # 为padding token安排'2'id
        for id in range(3, wordcount.__len__()+3):
            word = wordcount[id-3][0]
            self.id2word.append(word)
            self.word2id[word] = id
        print('Successfully building the dictionary !')