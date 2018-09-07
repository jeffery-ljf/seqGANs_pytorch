# -*- coding:utf-8 -*-
import numpy as np
class Dictionary():
    def __init__(self, src):
        data = open(src).readlines()
        #应该把数据处理成doc*line*word的列表
        self.data = []#存放原始数据
        self.id2word = []
        self.word2id = {}
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
        self.word2id['start token'] = 0#为start token安排'0'id
        self.id2word.append('end token')#加入end token
        self.word2id['end token'] = 1#为end token安排'1'id
        for id in range(2, wordcount.__len__()+2):
            word = wordcount[id-2][0]
            self.id2word.append(word)
            self.word2id[word] = id
        print('Successfully building the dictionary !')