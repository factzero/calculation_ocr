# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import datetime
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import glob
from angle_class import ANGCLS


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--image_root', default='./data/', type=str, help='test image root dir')
parser.add_argument('--resume_net', default='./checkpoints/angle_class_best.pth', type=str, help='net')


if __name__ == "__main__":
    opt = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ANGCLS(nclass=4, pretrained=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    if opt.resume_net !='' and os.path.exists(opt.resume_net):
        print('loading pretrained model from %s' % opt.resume_net)
        model.load_state_dict(torch.load(opt.resume_net))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    transform_test = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    labels_index = {0:'0', 1:'90', 2:'180', 3:'270'}
    angle_images = glob.glob(os.path.join(opt.image_root, '*.jpg'))
    for image_name in angle_images:
        image_ori = Image.open(image_name).convert("RGB")
        image = transform_test(image_ori).to(device)
        image = image.view(1, *image.size())
        preds = model(image)
        _, pred_label = preds.max(1)
        # p = torch.nn.functional.softmax(preds, dim=1)
        # print(p)

        angle_label = labels_index[pred_label.item()]

        image_show = image_ori.resize((224, 224))
        font = ImageFont.truetype("SansSerif.ttf", 40) 
        draw = ImageDraw.Draw(image_show)
        draw.text((10, 10), angle_label, fill=(255, 0, 0), font=font)
        plt.imshow(image_show)
        plt.show()
