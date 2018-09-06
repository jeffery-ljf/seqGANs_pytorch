# -*- coding:utf-8 -*-
import time
import torch
from torch import nn
from generator import Generator
from torch.nn import init


class Oracle(nn.Module):
    def __init__(self):
        super().__init__()
        self.vocab_size = 5000
        self.embedding_size = 64
        self.hidden_size_gru = 32  # GRU的隐藏层大小
        self.sequence_length = 20  # 句子的长度
        self.start_token = 0  # 开始token的序号
        self.batch_size = 10000#生成数据时的batch
        self.start_input = torch.tensor(self.batch_size * [self.start_token])#Generator开始的输入
        self.start_h = torch.zeros(self.batch_size, self.hidden_size_gru)#Generator开始的状态
        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
        self.G = Generator(self.vocab_size, self.embedding_size, self.hidden_size_gru)
    def generate_X(self, start_input, start_h, sequence_length):
        '''

        :param start_input: batch
        :param start_h: batch * hidden_size
        :param sequence_length: int
        :return:samples: seq_len * batch||hs: seq_len * batch * hidden_size||predictions: seq_len * batch * vocab_size
        '''
        samples = []
        predictions = []
        hs = []
        input = self.embeddings(start_input)#设置初始输入,batch, input_size
        last_h = start_h#设置初始状态
        for i in range(sequence_length):
            # 迭代GRU
            next_token, h, prediction = self.G(input, last_h)#获得当前时间步预测的下一个token，隐藏状态和预测层
            samples.append(torch.unsqueeze(next_token, dim=0))
            hs.append(torch.unsqueeze(h, dim=0))
            predictions.append(torch.unsqueeze(prediction, dim=0))
            input = self.embeddings(next_token)
            last_h = h
        samples = torch.cat(samples, dim=0)
        hs = torch.cat(hs, dim=0)
        predictions = torch.cat(predictions, dim=0)
        return samples, hs, predictions#return seq_len, batch  -   seq_len, batch, hidden_size   -   seq_len, batch, vocab_size
    def synthesize_data(self):
        samples, _, _ = self.generate_X(start_input=self.start_input,
                                        start_h=self.start_h,
                                        sequence_length=self.sequence_length)
        result = torch.transpose(samples, dim0=0, dim1=1)
        return result#return batch, seq_len
    def generate_pretrained(self, start_input, start_h, sequence_length, groundtrues):#预训练阶段，输入为正确的单词，输出预测
        '''

        :param start_input: batch
        :param start_h: batch * hidden_size
        :param sequence_length: int
        :param groundtrues: sequence_length * batch
        :return:predictions: seq_len * batch * vocab_size
        '''
        predictions = []
        input = self.embeddings(start_input)#设置初始输入,batch, input_size
        last_h = start_h#设置初始状态
        for i in range(sequence_length):
            # 迭代GRU
            next_token, h, prediction = self.G(input, last_h)#获得当前时间步预测的下一个token，隐藏状态和预测层
            predictions.append(torch.unsqueeze(prediction, dim=0))
            input = self.embeddings(groundtrues[i])#输入正确的单词embedding
            last_h = h
        predictions = torch.cat(predictions, dim=0)
        return predictions#return seq_len, batch, vocab_size
    def NLL(self, dataloader, start_input, start_h):
        '''
        :param samples: seq_len, batch
        :param start_input:batch,1 这里的batch不是生成模拟数据的batch，而是samples的batch
        :param start_h:batch, hidden_size_gru
        :return:nll 1
        '''
        loss_func = nn.NLLLoss()
        total_loss = 0.0
        for i, x_batch in enumerate(dataloader):#x_batch: batch * seq_len
            x_groundtrues = torch.transpose(x_batch, dim0=0, dim1=1).cuda()#x_groundtrues: seq_len * batch
            if x_batch.size()[0] == start_input.shape[0]:
                predictions = self.generate_pretrained(#predictions: seq_len * batch * vocab_size
                    start_input=start_input,
                    start_h=start_h,
                    sequence_length=self.sequence_length,
                    groundtrues=x_groundtrues
                )
            else:
                predictions = self.generate_pretrained(  # predictions: seq_len * batch * vocab_size
                    start_input=torch.tensor(x_batch.size()[0] * [self.start_token]).cuda(),
                    start_h=torch.zeros(x_batch.size()[0], self.hidden_size_gru).cuda(),
                    sequence_length=self.sequence_length,
                    groundtrues=x_groundtrues
                )
            loss = 0.0
            for t in range(self.sequence_length):
                loss += loss_func(torch.log(torch.clamp(predictions[t], min=1e-20, max=1.0)), x_groundtrues[t])#tar*log(pre)
            loss = loss / self.sequence_length
            total_loss += loss.item()
        total_loss = total_loss/i
        return total_loss
    def init_params(self):
        for param in self.parameters():
            param.data.normal_(0.0, 0.1)
