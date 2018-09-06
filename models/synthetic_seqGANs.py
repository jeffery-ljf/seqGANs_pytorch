# -*- coding:utf-8 -*-
import time
import torch
import visdom
from torch import nn
from discriminator import Discriminator
from generator import Generator
from datas.dataset import DataSet_Syn
from torch.utils.data import DataLoader
class SEQGANs_syn(nn.Module):
    def __init__(self):
        super().__init__()
        self.l2_reg_lambda = 0.2
        self.batch_size = 64#batch的大小
        self.filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]  # 判别器的窗口大小（也即每个窗口包含多少个单词）
        self.num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]  # 判别器channels数量
        self.sequence_length = 20  # 句子的长度
        self.num_classes = 1  # 判别器分类类别数量（输出结点数）
        self.embedding_size = 64  # 单词embedding大小
        self.hidden_size_gru = 32  # GRU的隐藏层大小
        self.start_token = 0#开始token的序号
        self.start_input = torch.tensor(self.batch_size * [self.start_token]).cuda()#Generator开始的输入
        self.start_h = torch.zeros(self.batch_size, self.hidden_size_gru).cuda()#Generator开始的状态
        self.rollout_num = 10#rollout的数量
        self.dataset = DataSet_Syn(root_src=r'../datas/Synthetic_Data/data.pkl')#载入数据
        self.vocab_size = 5000  # 字典大小
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=2)
        self.G = Generator(self.vocab_size, self.embedding_size, self.hidden_size_gru)
        self.D = Discriminator(self.sequence_length, self.num_classes, self.vocab_size, self.embedding_size, self.filter_sizes, self.num_filters)
        self.embeddings = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_size)
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
        samples, _, _ = self.generate_X(
            start_input=self.start_input,
            start_h=self.start_h,
            sequence_length=self.sequence_length,
        )
        return samples
    def generate_data(self):
        #这里输出的是真实数据量大小*seq_len
        start_input = torch.tensor(self.dataset.__len__() * [self.start_token])#Generator开始的输入
        start_h = torch.zeros(self.dataset.__len__(), self.hidden_size_gru)#Generator开始的状态
        samples, _, _ = self.generate_X(
            start_input=start_input,
            start_h=start_h,
            sequence_length=self.sequence_length,
        )
        return torch.transpose(samples, dim0=0, dim1=1)#return batch, seq_len
    def pretraining(self):
        loss_func = nn.NLLLoss()
        for epoch in range(1):
            time1 = time.time()
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
                loss.backward()
                self.pre_optimizer.step()
            total_loss = total_loss/i
            #输出loss和生成的字符

            time2 = time.time()
            print('total_loss : '+str(total_loss)+'   Times: '+str(time2-time1))
            return total_loss
    def rollout(self):
        samples, hs, predictions = self.generate_X(
            start_input=self.start_input,
            start_h=self.start_h,
            sequence_length=self.sequence_length,
        )
        result_rollout = []
        for given_num in range(self.sequence_length-1):#given < T, 遍历
            result_overtimes = []#存放每个时间步的rollout结果
            for i in range(self.rollout_num):
                sample_rollout, _, _ = self.generate_X(
                    start_input=samples[given_num],
                    start_h=hs[given_num],
                    sequence_length=self.sequence_length-given_num-1,
                )
                result_overtimes.append(torch.unsqueeze(torch.cat([samples[0:given_num+1], sample_rollout], 0), 0))
            result_overtimes = torch.cat(result_overtimes, 0)#result_overtimes: rollout_num * seq_len * batch
            result_rollout.append(torch.unsqueeze(result_overtimes, 0))
        result_rollout = torch.cat(result_rollout, 0)#result_rollout:(seq_len-1) * rollout_num * seq_len * batch
        return result_rollout, samples, predictions#result_rollout为1-（T-1）的rollout结果，samples为完整句子

    def onehot(self, label):
        a = torch.FloatTensor(self.sequence_length, self.batch_size, self.vocab_size).zero_().cuda()
        return a.scatter_(dim=2, index=label, value=1)
    def backward_G(self):
        result_rollout, result, predictions = self.rollout()
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
        J_temp = torch.sum(torch.log(torch.clamp(policy, min=1e-20, max=1.0)) * total_reward, 0)/self.sequence_length#J_temp: batch * 1
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
                x_batch_neg, _, _ = self.generate_X(
                    start_input=self.start_input,
                    start_h=self.start_h,
                    sequence_length=self.sequence_length
                )
            else:#如果dataloader抽出来的不满足batch_size的大小要求
                x_batch_neg, _, _ = self.generate_X(
                    start_input=torch.tensor(x_batch_pos.size()[0] * [self.start_token]).cuda(),
                    start_h=torch.zeros(x_batch_pos.size()[0], self.hidden_size_gru).cuda(),
                    sequence_length=self.sequence_length
                )
            x_batch_neg = torch.transpose(x_batch_neg, dim0=0, dim1=1)#x_batch_neg: batch * seq_len
            input_batch_pos = self.embeddings(x_batch_pos.cuda())
            input_batch_neg = self.embeddings(x_batch_neg)
            input_batch_pos = torch.unsqueeze(input_batch_pos, 1)#input_batch_pos: batch * 1 * seq_len * embedding_size
            input_batch_neg = torch.unsqueeze(input_batch_neg, 1)#input_batch_neg: batch * 1 * seq_len * embedding_size
            pre_pos = self.D(input_batch_pos)
            pre_neg = self.D(input_batch_neg)
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
        total_loss = total_loss/i
        return total_loss







