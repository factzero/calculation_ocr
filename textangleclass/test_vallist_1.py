# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import datetime
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from tqdm import tqdm
from angle_class import ANGCLS
from angle_dataset import imgDataset


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--image_root', default='./data/', type=str, help='test image root dir')
parser.add_argument('--resume_net', default='./checkpoints/angle_class_best.pth', type=str, help='net')
parser.add_argument('--val_list', default='', type=str, help='net')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')


def val(model, loader, device):
    print('Start val')
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    num_correct = 0
    num_total = 0
    for i, (image, label, index) in tqdm(enumerate(loader), total=len(loader), desc='test model'):
        image = image.to(device)
        preds = model(image)
        label = label.to(device)
        _, pred_label = preds.max(1)
        num_correct += (pred_label == label).sum().item()
        num_total += preds.shape[0]

    accuracy = num_correct / num_total
    print('accuracy: %0.4f, %d/%d' % (accuracy, num_correct, num_total))
    return accuracy


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

    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    val_dataset = imgDataset(opt.image_root, opt.val_list, transform_val)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)
    accuracy = val(model, val_dataloader, device)
    print('accuracy: %0.4f' % (accuracy))