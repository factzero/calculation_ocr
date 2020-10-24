
# -*- coding: utf-8 -*-
'''
用基础网络预训练参数初始化网络
'''
import os
import torch
import torch.nn as nn
import torchvision.models as models


if __name__ == '__main__':
    base_model = models.vgg16(pretrained=False)
    layers = list(base_model.features)[:-1]
    model = nn.Sequential(*layers)
    backbone = torch.load('D:/06DL/OCR/vgg16_bn-6c64b313.pth')
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k, v in backbone.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    check_path = os.path.join('./checkpoints', 'vgg16_bn_base.pth')
    torch.save(model.state_dict(), check_path) 
