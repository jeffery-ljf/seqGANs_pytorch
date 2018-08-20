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
        self.data = np.array(self.data)
    def __len__(self):
        if self.id2word.__len__()==self.word2id.__len__():
            return self.id2word.__len__()
        else:
            raise RuntimeError('dictionary size error')
    def build(self):
        temp = []
        wordcount = []
        for doc in self.data:
            for word in doc:
                if word in temp:
                    id = temp.index(word)
                    if wordcount[id][0]==temp[id]:
                        wordcount[id][1] = wordcount[id][1]+1
                    else:
                        raise RuntimeError('dictionary operation error')
                else:
                    temp.append(word)
                    wordcount.append([word,0])

        wordcount.sort(key=lambda item: item[1], reverse=True)
        self.id2word.append('start token')
        self.word2id['start token'] = 0
        for id in range(1, wordcount.__len__()+1):
            word = wordcount[id-1][0]
            self.id2word.append(word)
            self.word2id[word] = id
        print('Successfully building the dictionary !')