# -*- coding: utf-8 -*-
import argparse
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import crnn_params
from crnn_model import CRNN
from crnn_dataset import imgDataset
import crnn_utils as utils


parser = argparse.ArgumentParser(description='train')
parser.add_argument('--image_root', default='./data/', type=str, help='train image root dir')
parser.add_argument('--val_label', default='./data/data_test.list', type=str, help='val label')
parser.add_argument('--trained_net', default='', help='trained net')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')


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

        if (i_batch+1)%crnn_params.displayInterval == 0:
            print('[%d/%d][%d/%d]' % (iteration, crnn_params.niter, i_batch, len(loader)))

        n_total += batch_size
        # if i_batch == max_i:
        #     break

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:crnn_params.n_test_disp]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, label):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    print(n_correct, n_total)
    accuracy = n_correct / float(n_total)
    print('Test loss: %f, accuray: %f' % (loss_avg.val(), accuracy))

    return accuracy


if __name__ == "__main__":
    opt = parser.parse_args()
    image_root = opt.image_root
    val_label = opt.val_label
    trained_net = opt.trained_net
    batch_size = opt.batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    nclass = len(crnn_params.alphabet) + 1
    nc = 1
    model = CRNN(32, nc, nclass, crnn_params.nh)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    criterion = torch.nn.CTCLoss(reduction='sum')
    if trained_net!='' and os.path.exists(trained_net):
        print('loading pretrained model from %s' % trained_net)
        model.load_state_dict(torch.load(trained_net))

    val_dataset = imgDataset(image_root, val_label, crnn_params.alphabet, (crnn_params.imgW, crnn_params.imgH), crnn_params.mean, crnn_params.std, is_aug=False)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=crnn_params.workers)

    accuracy = val(model, val_dataloader, criterion, 1, device, max_i=20)
