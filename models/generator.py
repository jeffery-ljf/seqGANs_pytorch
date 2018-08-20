# -*- coding:utf-8 -*-
from torch import nn
import torch
class Generator(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size, bias=True):
        super().__init__()
        self.grucell = nn.GRUCell(input_size=embedding_size, hidden_size=hidden_size, bias=bias)
        self.output_layer = nn.Sequential(*[nn.Linear(in_features=hidden_size, out_features=vocab_size, bias=True),
                                            nn.Softmax(dim=1)])
    def forward(self, input, h_last):
        try:
            h = self.grucell(input, h_last)#return batch, hidden_size
            output = self.output_layer(h)#return batch, vocab_size
            sample = torch.squeeze(torch.multinomial(output, 1))
            return sample, h, output#return batch  -   batch, hidden_size   -   batch, vocab_size
        except Exception as e:
            print(repr(e))
