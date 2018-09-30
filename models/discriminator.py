# -*- coding:utf-8 -*-
from torch import nn
import torch
class Discriminator(nn.Module):
    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size, filter_sizes, num_filters):
        super().__init__()
        self.vocab_size = vocab_size
        self.conv_pools = nn.ModuleList()
        for filter_size, num_filter in zip(filter_sizes,num_filters):
            # Convolution Layer and Pool Layer
            model = [nn.Conv2d(in_channels=1, out_channels=num_filter, kernel_size=(filter_size, embedding_size), stride=1),
                     nn.ReLU(True),
                     nn.MaxPool2d(kernel_size=(sequence_length - filter_size + 1, 1))]
            self.conv_pools.append(nn.Sequential(*model))
        total_features = sum(num_filters)
        # Highway Layer
        self.highway_linear_H = nn.Sequential(*[nn.Linear(in_features=total_features, out_features=total_features, bias=True),
                                                nn.ReLU(True)])
        self.highway_linear_G = nn.Sequential(*[nn.Linear(in_features=total_features, out_features=total_features, bias=True),
                                                nn.Sigmoid()])
        # Dropout Layer
        self.dropout = nn.Sequential(*[nn.Dropout(0.5)])
        # Output Layer
        self.output_layer = nn.Sequential(*[nn.Linear(in_features=total_features, out_features=num_classes, bias=True),
                                            nn.Sigmoid()])
    def __highway(self, input_, num_layers=1):
        """Highway Network (cf. http://arxiv.org/abs/1505.00387).
            t = sigmoid(Wy + b)
            z = t * g(Wy + b) + (1 - t) * y
            where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
        """
        for idx in range(num_layers):
            g = self.highway_linear_H(input_)
            t = self.highway_linear_G(input_)
            output = t*g+(1.-t)*input_
            return output
    def forward(self, input):
        result = []
        for i in range(self.conv_pools.__len__()):
            result.append(self.conv_pools[i](input))
        self.h_pooled_flat = torch.squeeze(torch.cat(result, dim=1))
        self.h_highway = self.__highway(input_=self.h_pooled_flat, num_layers=1)
        self.h_dropout = self.dropout(self.h_highway)
        self.y_pre = self.output_layer(self.h_dropout)
        return self.y_pre