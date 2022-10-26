# pylint: disable=missing-function-docstring,missing-class-docstring,invalid-name
import logging

import torch

# Source: https://www.kaggle.com/code/pankajj/fashion-mnist-with-pytorch-93-accuracy/notebook
class FashionMNISTCNN(torch.nn.Module):
    def __init__(self):
        super(FashionMNISTCNN, self).__init__()

        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.fc1 = torch.nn.Linear(in_features=64 * 6 * 6, out_features=600)
        self.drop = torch.nn.Dropout2d(0.25)
        self.fc2 = torch.nn.Linear(in_features=600, out_features=120)
        self.fc3 = torch.nn.Linear(in_features=120, out_features=10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out


    def change_size(self, learning_params):
        if learning_params.model_size == 0:
            scale = 0.5
        elif learning_params.model_size == 1:
            scale = 0.75
        elif learning_params.model_size == 2:
            scale = 1

        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=32, out_channels=int(64*scale), kernel_size=3),
            torch.nn.BatchNorm2d(int(64*scale)),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )

        self.fc1 = torch.nn.Linear(in_features=int(64 * 6 * 6 * scale), out_features=int(600 * scale))
        self.drop = torch.nn.Dropout2d(0.25)
        self.fc2 = torch.nn.Linear(in_features=int(600 * scale), out_features=int(120 * scale))
        self.fc3 = torch.nn.Linear(in_features=int(120 * scale), out_features=10)
        pass