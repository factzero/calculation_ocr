# -*- coding: utf-8 -*-
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tools.dataset_det import vocDataset
from textdetection.ctpn_model import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss
from textdetection import params


parser = argparse.ArgumentParser(description='train')
parser.add_argument('--image_root', default='D:/04download/VOC2007_v0', type=str, help='train image root dir')
parser.add_argument('--save_folder', default='./checkpoints/', help='Location to save checkpoint models')
parser.add_argument('--batch_size', default=64, type=int, help='batch size')
parser.add_argument('--resume_net', default='', help='resume net')
parser.add_argument('--resume_iter', default=0, type=int, help='resume Iteration')


def train(model, loader, criterion_cls, criterion_regr, optimizer, iteration, device):
    for p in model.parameters():
        p.requires_grad = True
    model.train()

    epoch_loss_cls = 0
    epoch_loss_regr = 0
    epoch_loss = 0
    epoch_size = len(loader)
    for i_batch, (image, gt_boxes, clss, index) in enumerate(loader):
        gt_boxes = gt_boxes.to(device)
        clss = clss.to(device)
        image = image.to(device)
        out_cls, out_regr = model(image)
        loss_cls = criterion_cls(out_cls, clss)
        loss_regr = criterion_regr(out_regr, gt_boxes)
        loss = loss_cls + loss_regr   

        model.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss_cls += loss_cls.item()
        epoch_loss_regr += loss_regr.item()
        epoch_loss += loss.item()
        mmp = i_batch + 1

        print(f'Batch:{i_batch}/{epoch_size}\n'
              f'batch: loss_cls:{loss_cls.item():.4f}--loss_regr:{loss_regr.item():.4f}--loss:{loss.item():.4f}\n'
              f'Epoch: loss_cls:{epoch_loss_cls/mmp:.4f}--loss_regr:{epoch_loss_regr/mmp:.4f}--'
              f'loss:{epoch_loss/mmp:.4f}\n')
    
    epoch_loss_cls /= epoch_size
    epoch_loss_regr /= epoch_size
    epoch_loss /= epoch_size

    return epoch_loss_cls, epoch_loss_regr, epoch_loss


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


if __name__ == "__main__":
    opt = parser.parse_args()
    image_root = opt.image_root
    save_folder = opt.save_folder
    batch_size = opt.batch_size
    resume_net = opt.resume_net

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CTPN_Model()

    criterion_cls = RPN_CLS_Loss(device)
    criterion_regr = RPN_REGR_Loss(device)
    optimizer = optim.SGD(model.parameters(), lr=params.lr, momentum=0.9)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    
    if resume_net!='' and os.path.exists(resume_net):
        print('loading pretrained model from %s' % resume_net)
        model.load_state_dict(torch.load(resume_net))
    else:
        model.apply(weights_init)
    
    train_dataset = vocDataset(image_root)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
	
    Iteration = opt.resume_iter
    best_loss_cls = 100
    best_loss_regr = 100
    best_loss = 100
    while Iteration < params.niter:
        print(f'Epoch {Iteration}/{params.niter}')
        print('#'*50)
        epoch_loss_cls, epoch_loss_regr, epoch_loss = train(model, train_dataloader, criterion_cls, criterion_regr, optimizer, Iteration, device)
        if best_loss_cls > epoch_loss_cls or best_loss_regr > epoch_loss_regr or best_loss > epoch_loss:
            best_loss = epoch_loss
            best_loss_regr = epoch_loss_regr
            best_loss_cls = epoch_loss_cls
            torch.save(model.state_dict(), '{0}/ctpn_done_{1}.pth'.format(save_folder, Iteration))
        Iteration += 1