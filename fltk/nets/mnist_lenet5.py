# pylint: disable=missing-class-docstring,invalid-name,missing-function-docstring
import logging

import torch.nn as nn
import torch.nn.functional as F

class MNIST_LENET5(nn.Module):
    def __init__(self):
        super().__init__()
        # logging.info("TEST NEW NETWORK")
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)

        self.fc1 = nn.Linear(in_features=256, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x): # pylint: disable=missing-function-docstring
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # logging.info(f"{x}")
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        # logging.info(f"{x}")
        x = x.view(-1, 256)
        # logging.info(f"{x}")
        x = F.relu(self.fc1(x))
        # logging.info(f"{x}")
        x = F.relu(self.fc2(x))
        # logging.info(f"{x}")
        x = F.relu(self.fc3(x))
        # logging.info(f"{x}")
        return F.log_softmax(x)

