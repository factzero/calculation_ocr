# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import datetime
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from angle_class import ANGCLS
from angle_dataset import imgDataset
from tqdm import tqdm


parser = argparse.ArgumentParser(description='train')
parser.add_argument('--image_root', default='./data/', type=str, help='train image root dir')
parser.add_argument('--train_label', default='/data/data_train.list', type=str, help='train label')
parser.add_argument('--val_label', default='./data/data_test.list', type=str, help='val label')
parser.add_argument('--save_folder', default='./checkpoints/', help='Location to save checkpoint models')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--resume_net', default='', help='resume net')
parser.add_argument('--resume_iter', default=0, type=int, help='resume Iteration')
parser.add_argument('--nepoch', default=120, type=int, help='number of epoch')

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


def get_acc(output, label):
    total = output.shape[0]
    _, pred_label = output.max(1)
    num_correct = (pred_label == label).sum().item()
    return num_correct / total


def train(model, loader, criterion, optimizer, iteration, device):
    for p in model.parameters():
        p.requires_grad = True
    model.train()

    for i_batch, (image, label, index) in enumerate(loader):
        image = image.to(device)
        preds = model(image)
        label = label.to(device)
        batch_size = image.size(0)
        cost = criterion(preds, label)
        model.zero_grad()
        cost.backward()
        optimizer.step()
        train_acc = get_acc(preds, label)

        theTime = datetime.datetime.now()
        print('%s [%d/%d][%d/%d] ACC: %f' % (theTime, iteration, opt.nepoch, i_batch, len(loader), train_acc))


if __name__ == "__main__":
    opt = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ANGCLS(nclass=4, pretrained=True)
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.5, 0.999))
    criterion = torch.nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    if opt.resume_net !='' and os.path.exists(opt.resume_net):
        print('loading pretrained model from %s' % opt.resume_net)
        model.load_state_dict(torch.load(opt.resume_net))

    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    train_dataset = imgDataset(opt.image_root, opt.train_label, transform_train)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)
    val_dataset = imgDataset(opt.image_root, opt.val_label, transform_val)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False)

    if not os.path.exists(opt.save_folder):
        os.mkdir(opt.save_folder)
    
    best_accuracy = 0
    Iteration = opt.resume_iter
    while Iteration < opt.nepoch:
        train(model, train_dataloader, criterion, optimizer, Iteration, device)
        accuracy = val(model, val_dataloader, device)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), '{0}/angle_class_done_{1:04d}_{2:.4f}.pth'.format(opt.save_folder, Iteration, accuracy))
            torch.save(model.state_dict(), '{0}/angle_class_best.pth'.format(opt.save_folder))
        Iteration += 1
