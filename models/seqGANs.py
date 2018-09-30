# -*- coding:utf-8 -*-
import time
import torch
import visdom
from torch import nn
from discriminator import Discriminator
from generator import Generator
from datas.dataset import Quatrains,DataSet_Obama
from torch.utils.data import DataLoader
class SEQGANs(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_reg_lambda = 0.2
        self.batch_size = 8#batch的大小,为1的时候，过程有使用unsqueeze，可能会出错
        self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]  # 判别器的窗口大小（也即每个窗口包含多少个单词）
        self.num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]  # 判别器channels数量
        self.num_classes = 1  # 判别器分类类别数量（输出结点数）
        self.embedding_size = 100  # 单词embedding大小
        self.hidden_size_gru = 100  # GRU的隐藏层大小
        self.start_idx = 0#开始token的序号
        self.end_idx = 1#结束token的序号
        self.padding_idx = 2#填充token的序号
        self.start_input = torch.tensor(self.batch_size * [self.start_idx]).cuda()#Generator开始的输入
        self.start_h = torch.zeros(self.batch_size, self.hidden_size_gru).cuda()#Generator开始的状态
        self.rollout_num = 10#rollout的数量
        self.dataset = DataSet_Obama(root_src=r'../datas/obama/input.txt', start_idx=self.start_idx, end_idx=self.end_idx, padding_idx=self.padding_idx)#载入真实数据
        self.sequence_length = self.dataset.max_doclen + 1  # 真实数据集的最大句子长度+1(算上end token)
        self.vocab_size = self.dataset.dictionary.__len__()  # 字典大小
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False, num_workers=2)
        self.G = Generator(self.vocab_size, self.embedding_size, self.hidden_size_gru)
        self.D = Discriminator(self.sequence_length, self.num_classes, self.vocab_size, self.embedding_size, self.filter_sizes, self.num_filters)
        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size, padding_idx=self.padding_idx)
        self.pre_optimizer = torch.optim.Adam([
            {'params': self.G.parameters()},
            {'params': self.embeddings.parameters()}
        ])
        self.G_optimizer = torch.optim.Adam([
            {'params': self.G.parameters()},
            {'params': self.embeddings.parameters()}
        ])
        self.D_optimizer = torch.optim.Adam([
            {'params': self.D.parameters()},
            {'params': self.embeddings.parameters()}
        ])

    def forward(self, input):
        return -1
    def pad_data(self, samples, record, sequence_length):#对数据进行padding
        '''
        :param samples:seq_len, batch
        :param record:dictionary
        :return:
        '''
        for b in record.keys():
            for t in range(record[b]+1, sequence_length):
                samples[t][b] = 2
        return samples
    def generate_X_nofixedlen(self, start_input, start_h):#生成器生成不定长的句子（会使用padding token进行填充）
        '''
        :param start_input: batch
        :param start_h: batch * hidden_size
        :param sequence_length: int
        :return:samples: seq_len * batch||hs: seq_len * batch * hidden_size||predictions: seq_len * batch * vocab_size
        '''
        record = {}#记录已经生成出end token的batch idx,以及对应在samples中end token的位置序号
        now_len = 0#记录最新生成的长度
        samples = []
        predictions = []
        hs = []
        input = self.embeddings(start_input)  # 设置初始输入,batch, input_size
        last_h = start_h  # 设置初始状态
        while record.__len__() != start_h.shape[0]:#判断是否所有batch都生成初end token
            # 迭代GRU
            next_token, h, prediction = self.G(input, last_h)  # 获得当前时间步预测的下一个token，隐藏状态和预测层
            samples.append(torch.unsqueeze(next_token, dim=0))
            hs.append(torch.unsqueeze(h, dim=0))
            predictions.append(torch.unsqueeze(prediction, dim=0))
            input = self.embeddings(next_token)
            last_h = h
            for i in range(next_token.shape[0]):#判断每一个next token是否end token
                if next_token[i] == 1 and i not in record.keys():
                    record[i] = now_len
            now_len += 1
        samples = torch.cat(samples, dim=0)
        hs = torch.cat(hs, dim=0)
        predictions = torch.cat(predictions, dim=0)
        samples = self.pad_data(samples=samples, record=record)#对生成出来的token的end token后的位置进行padding。
        return samples, hs, predictions, record, now_len  # return seq_len, batch  -   seq_len, batch, hidden_size   -   seq_len, batch, vocab_size, list, int
    def generate_X(self, start_input, start_h, sequence_length):#生成样本，有最大长度
        '''

        :param start_input: batch
        :param start_h: batch * hidden_size
        :param sequence_length: int
        :return:samples: seq_len * batch||hs: seq_len * batch * hidden_size||predictions: seq_len * batch * vocab_size
        '''
        record = {}  # 记录已经生成出end token的batch idx,以及对应在samples中end token的位置序号
        now_len = 0  # 记录最新生成的长度
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
            for i in range(next_token.shape[0]):#判断每一个next token是否end token
                if next_token[i] == 1 and i not in record.keys():
                    record[i] = now_len
            now_len += 1
        samples = torch.cat(samples, dim=0)
        hs = torch.cat(hs, dim=0)
        predictions = torch.cat(predictions, dim=0)
        samples = self.pad_data(samples=samples, record=record, sequence_length = sequence_length)  # 对生成出来的token的end token后的位置进行padding。
        return samples, hs, predictions, record#return seq_len, batch  -   seq_len, batch, hidden_size   -   seq_len, batch, vocab_size   -  list
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
    def show_G(self):
        samples, _, _, _ = self.generate_X(
            start_input=self.start_input,
            start_h=self.start_h,
            sequence_length=self.sequence_length
        )
        return samples
    def pretraining(self):
        loss_func = nn.NLLLoss(ignore_index=self.padding_idx)
        for epoch in range(1):
            total_loss = 0.0
            for i, x_batch in enumerate(self.dataloader):#x_batch: batch * seq_len
                self.pre_optimizer.zero_grad()
                x_groundtrues = torch.transpose(x_batch, dim0=0, dim1=1).cuda()#x_groundtrues: seq_len * batch
                if x_batch.size()[0] == self.batch_size:
                    predictions = self.generate_pretrained(#predictions: seq_len * batch * vocab_size
                        start_input=self.start_input,
                        start_h=self.start_h,
                        sequence_length=self.sequence_length,
                        groundtrues=x_groundtrues
                    )
                else:
                    predictions = self.generate_pretrained(  # predictions: seq_len * batch * vocab_size
                        start_input=torch.tensor(x_batch.size()[0] * [self.start_idx]).cuda(),
                        start_h=torch.zeros(x_batch.size()[0], self.hidden_size_gru).cuda(),
                        sequence_length=self.sequence_length,
                        groundtrues=x_groundtrues
                    )
                loss = 0.0
                for t in range(self.sequence_length):
                    loss += loss_func(torch.log(torch.clamp(predictions[t], min=1e-20, max=1.0)), x_groundtrues[t])#tar*log(pre)
                loss = loss / self.sequence_length
                total_loss += loss.item()
                loss.backward()
                self.pre_optimizer.step()
            total_loss = total_loss/(i+1)
            #输出loss和生成的字符
            return total_loss
    def rollout(self):
        samples, hs, predictions, record = self.generate_X(
            start_input=self.start_input,
            start_h=self.start_h,
            sequence_length=self.sequence_length
        )
        result_rollout = []
        for given_num in range(self.sequence_length-1):#given < T, 遍历
            result_overtimes = []#存放每个时间步的rollout结果
            for i in range(self.rollout_num):
                sample_rollout, _, _, _ = self.generate_X(
                    start_input=samples[given_num],
                    start_h=hs[given_num],
                    sequence_length=self.sequence_length-given_num-1,
                )
                result_overtimes.append(torch.unsqueeze(torch.cat([samples[0:given_num+1], sample_rollout], 0), 0))
            result_overtimes = torch.cat(result_overtimes, 0)#result_overtimes: rollout_num * seq_len * batch
            result_rollout.append(torch.unsqueeze(result_overtimes, 0))
        result_rollout = torch.cat(result_rollout, 0)#result_rollout:(seq_len-1) * rollout_num * seq_len * batch
        return result_rollout, samples, predictions, record#result_rollout为1-（T-1）的rollout结果，samples为完整句子

    def onehot(self, label):
        a = torch.FloatTensor(self.sequence_length, self.batch_size, self.vocab_size).zero_().cuda()
        return a.scatter_(dim=2, index=label, value=1)
    def generate_code(self, record):
        '''

        :param record:
        :return code:seq_len * batch * 1
                    根据每个batch的句子长度来生成seq_len * batch * 1的0 1编码，1表示该位置的reward应该算上，0表示不算上。
                    如seq_len为4句子长度分别为1,3,2的batch(end token分别对应samples中的1,3,2位置)对应的code为：
                    1 1 1
                    0 1 1
                    0 1 0
                    1 1 1
                 num_elements:batch
                    为code中每个batch统计出为1的数量。
        '''
        num_elements = torch.zeros(self.batch_size, 1).new_full(size=(self.batch_size, 1), fill_value=self.sequence_length)
        code = torch.ones(self.sequence_length, self.batch_size, 1)
        for b in record.keys():
            num_elements[b][0] = record[b]+1
            for t in range(record[b], self.sequence_length-1):
                code[t][b][0] = 0
        return code, num_elements
    def backward_G(self):
        result_rollout, result, predictions, record = self.rollout()
        total_reward = []
        for t in range(self.sequence_length-1):#计算T-1的rollout奖励
            result_rollout_trans = torch.transpose(result_rollout[t], dim0=1, dim1=2)#result_rollout_trans: rollout_num * batch * seq_len
            input_D = self.embeddings(result_rollout_trans)
            input_D = torch.unsqueeze(input_D, 2)#input_D: rollout_num * batch * 1 * seq_len * embedding_size
            reward = 0.0
            for i in range(self.rollout_num):
                reward += self.D(input_D[i])
            reward = reward/self.rollout_num
            total_reward.append(torch.unsqueeze(reward, 0))
        #计算T时间的奖励
        result_trans = torch.transpose(result, dim0=0, dim1=1)#result_trans: batch * seq_len
        input_D = self.embeddings(result_trans)
        input_D = torch.unsqueeze(input_D, 1)#input_D: batch * 1 * seq_len * embedding_size
        total_reward.append(torch.unsqueeze(self.D(input_D), 0))
        total_reward = torch.cat(total_reward, 0)#total_reward: seq_len * batch * 1
        #计算J
        result_onehot = self.onehot(torch.unsqueeze(result, 2))
        policy = result_onehot * predictions
        policy = torch.unsqueeze(torch.sum(policy, 2), 2)#policy: seq_len * batch * 1
        code, num_elements = self.generate_code(record=record)
        code = code.cuda()
        num_elements = num_elements.cuda()
        J_temp = torch.sum(torch.log(torch.clamp(policy, min=1e-20, max=1.0)) * total_reward * code, 0)/num_elements#J_temp: batch * 1
        J = -(torch.sum(J_temp)/self.batch_size)
        self.G_optimizer.zero_grad()
        J.backward()
        self.G_optimizer.step()
        return J.item()
    def backward_D(self, update=True, loss_f='LOG', is_epoch = False):#is_epoch: 是否遍历整个真实样本
        total_loss = 0.0
        mse = nn.MSELoss()
        for i, x_batch_pos in enumerate(self.dataloader):
            self.D_optimizer.zero_grad()
            if x_batch_pos.size()[0] == self.batch_size:
                x_batch_neg, _, _, _ = self.generate_X(
                    start_input=self.start_input,
                    start_h=self.start_h,
                    sequence_length=self.sequence_length
                )
            else:#如果dataloader抽出来的不满足batch_size的大小要求
                x_batch_neg, _, _, _ = self.generate_X(
                    start_input=torch.tensor(x_batch_pos.size()[0] * [self.start_idx]).cuda(),
                    start_h=torch.zeros(x_batch_pos.size()[0], self.hidden_size_gru).cuda(),
                    sequence_length=self.sequence_length
                )
            x_batch_neg = torch.transpose(x_batch_neg, dim0=0, dim1=1)#x_batch_neg: batch * seq_len
            input_batch_pos = self.embeddings(x_batch_pos.cuda())
            input_batch_neg = self.embeddings(x_batch_neg)
            input_batch_pos = torch.unsqueeze(input_batch_pos, 1)#input_batch_pos: batch * 1 * seq_len * embedding_size
            input_batch_neg = torch.unsqueeze(input_batch_neg, 1)#input_batch_neg: batch * 1 * seq_len * embedding_size
            pre_pos = self.D(input=input_batch_pos)
            pre_neg = self.D(input=input_batch_neg)
            if loss_f=='LOG':
                loss = -torch.sum(torch.log(torch.clamp(pre_pos, min=1e-20, max=1.0))+torch.log(torch.clamp((1-pre_neg), min=1e-20, max=1.0)))/(2*pre_pos.size()[0])
            elif loss_f=='MSE':
                loss = (mse(pre_pos, torch.ones(x_batch_pos.size()[0], 1).cuda()) + mse(pre_neg, torch.zeros(x_batch_pos.size()[0], 1).cuda()))/2.0
            # 加入L2正则化
            l2_loss = torch.tensor(0.).cuda()
            for param in self.D.output_layer.parameters():
                l2_loss += torch.norm(param, p=2)
            loss += self.l2_reg_lambda * l2_loss

            total_loss += loss.item()
            loss.backward()
            if update:
                self.D_optimizer.step()
            if not is_epoch:
                return  total_loss#只训练一个batch
        total_loss = total_loss/(i+1)
        return total_loss







