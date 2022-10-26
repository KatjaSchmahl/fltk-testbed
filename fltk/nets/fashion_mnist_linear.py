# pylint: disable=missing-function-docstring,missing-class-docstring,invalid-name
import logging

import torch

# Source: https://www.kaggle.com/code/pankajj/fashion-mnist-with-pytorch-93-accuracy/notebook
class FashionMNISTLIN(torch.nn.Module):
    def __init__(self):
        super(FashionMNISTLIN, self).__init__()

        self.linear = torch.nn.Linear(28*28, 10)


    def forward(self, x):
        xb = x.reshape(-1, 784)
        out = self.linear(xb)
        return out