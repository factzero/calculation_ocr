# -*- coding: utf-8 -*-
import argparse
import numpy as np
import os
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from textrecognition import params
from textrecognition.crnn import CRNN
from tools.dataset import imgDataset
import tools.utils as utils


def val(model, loader, criterion, iteration, device, max_i=1000):
    print('Start val')
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    
    loss_avg = utils.averager()
    alphabet = ''.join([chr(uni) for uni in params.alphabet])
    converter = utils.strLabelConverter(alphabet)
    n_total = 0
    n_correct = 0
    for i_batch, (image, label, index) in enumerate(loader):
        image = image.to(device)
        preds = model(image)
        batch_size = image.size(0)
        index = np.array(index.data.numpy())
        text, length = converter.encode(label)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        cost = criterion(preds, text, preds_size, length) / batch_size
        loss_avg.add(cost)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        for pred, target in zip(sim_preds, label):
            if pred == target:
                n_correct += 1

        if (i_batch+1)%params.displayInterval == 0:
            print('[%d/%d][%d/%d]' % (iteration, params.niter, i_batch, len(loader)))

        n_total += batch_size
        if i_batch == max_i:
            break

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:params.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, label):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    print(n_correct, n_total)
    accuracy = n_correct / float(n_total)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

    return accuracy


def train(model, loader, criterion, optimizer, iteration, device):
    for p in model.parameters():
        p.requires_grad = True
    model.train()

    loss_avg = utils.averager()
    alphabet = ''.join([chr(uni) for uni in params.alphabet])
    converter = utils.strLabelConverter(alphabet)
    for i_batch, (image, label, index) in enumerate(loader):
        image = image.to(device)
        preds = model(image)
        batch_size = image.size(0)
        text, length = converter.encode(label)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        cost = criterion(preds, text, preds_size, length) / batch_size
        model.zero_grad()
        cost.backward()
        optimizer.step()
        loss_avg.add(cost)

        if (i_batch+1) % params.displayInterval == 0:
            print('[%d/%d][%d/%d] Loss: %f' % (iteration, params.niter, i_batch, len(loader), loss_avg.val()))
            loss_avg.reset()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def backward_hook(self, grad_input, grad_output):
    for g in grad_input:
        g[g != g] = 0   # replace all nan/inf in gradients to zero


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nclass = len(params.alphabet) + 1
    nc = 1
    model = CRNN(params.imgH, nc, nclass, params.nh)
    criterion = torch.nn.CTCLoss(reduction='sum', zero_infinity=True)
    optimizer = optim.Adam(model.parameters(), lr=params.lr, betas=(params.beta1, 0.999))
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
        criterion = criterion.cuda()
    model.apply(weights_init)
    model.register_backward_hook(backward_hook)
    # model_path = './checkpoints/CRNN.pth'
    # model.load_state_dict(torch.load(model_path))

    train_dataset = imgDataset('D:/80dataset/ocr/DataSet/testxx/images', 'D:/80dataset/ocr/DataSet/testxx/train.list', 
                                params.alphabet, (params.imgW, params.imgH), params.mean, params.std)
    train_dataloader = DataLoader(train_dataset, batch_size=params.batchSize, shuffle=True, num_workers=params.workers)
    val_dataset = imgDataset('D:/80dataset/ocr/DataSet/testxx/images', 'D:/80dataset/ocr/DataSet/testxx/train.list', 
                                params.alphabet, (params.imgW, params.imgH), params.mean, params.std)
    val_dataloader = DataLoader(val_dataset, batch_size=params.batchSize, shuffle=False)

    if not os.path.exists(params.expr_dir):
        os.mkdir(params.expr_dir)
    
    best_accuracy = 0
    Iteration = 0
    while Iteration < params.niter:
        train(model, train_dataloader, criterion, optimizer, Iteration, device)
        accuracy = val(model, val_dataloader, criterion, Iteration, device, max_i=2000)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), '{0}/crnn_Rec_done_{1}_{2}.pth'.format(params.expr_dir, Iteration, accuracy))
            torch.save(model.state_dict(), '{0}/crnn_best.pth'.format(params.expr_dir))
        Iteration += 1
