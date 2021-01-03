# -*- coding: utf-8 -*-
import torch.nn as nn
from resnet import resnet18


class ANGCLS(nn.Module):
    def __init__(self, nclass, pretrained=False):
        super(ANGCLS, self).__init__()
        self.blocknet = resnet18(pretrained=pretrained)
        fc_features = self.blocknet.fc.in_features
        self.blocknet.fc = nn.Linear(fc_features, nclass)

    def forward(self, x):
        output = self.blocknet(x)

        return output