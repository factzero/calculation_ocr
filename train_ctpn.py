# -*- coding: utf-8 -*-
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import argparse
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from textdetection.ctpn.ctpn_dataset import vocDataset
from textdetection.ctpn.ctpn_model import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss
from textdetection.ctpn import ctpn_params


parser = argparse.ArgumentParser(description='train')
parser.add_argument('--image_root', default='D:/04download/VOC2007_v0', type=str, help='train image root dir')
parser.add_argument('--save_folder', default='./checkpoints/', help='Location to save checkpoint models')
parser.add_argument('--batch_size', default=4, type=int, help='batch size')
parser.add_argument('--resume_net', default='', help='resume net weights')
parser.add_argument('--backbone_net', default='', help='backbone net weights')


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

        optimizer.zero_grad()

        out_cls, out_regr = model(image)
        loss_cls = criterion_cls(out_cls, clss)
        loss_regr = criterion_regr(out_regr, gt_boxes)
        loss = loss_cls + loss_regr   

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

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CTPN_Model()

    criterion_cls = RPN_CLS_Loss(device)
    criterion_regr = RPN_REGR_Loss(device)
    optimizer = optim.SGD([{'params': model.parameters(), 'initial_lr': 1e-3}], lr=ctpn_params.lr, momentum=0.9)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    
    resume_epoch = 0
    # ������ѵ��ģ�ͣ����ڶϵ����ѵ��
    if opt.resume_net !='' and os.path.exists(opt.resume_net):
        print('loading pretrained model from %s' % opt.resume_net)
        cc = torch.load(opt.resume_net, map_location=device)
        model.load_state_dict(cc['model_state_dict'])
        optimizer.load_state_dict(cc['optimizer'])
        resume_epoch = cc['epoch'] + 1
    # ���ػ�������Ԥѵ��ģ�ͣ�����Ȩ��ʹ��Ĭ�ϳ�ʼ��
    elif opt.backbone_net !='' and os.path.exists(opt.backbone_net):
        print('loading pretrained model from %s' % opt.backbone_net)
        backbone = torch.load(opt.backbone_net, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k, v in backbone.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    # ��ʼ��Ȩ��
    else:
        model.apply(weights_init)
    
    train_dataset = vocDataset(opt.image_root)
    train_dataloader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
	
    epoch = resume_epoch
    best_loss_cls = 100
    best_loss_regr = 100
    best_loss = 100
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1, last_epoch=epoch)
    while epoch < ctpn_params.niter:
        print(f'Epoch {epoch}/{ctpn_params.niter}')
        print('#'*50)
        epoch_loss_cls, epoch_loss_regr, epoch_loss = train(model, train_dataloader, criterion_cls, criterion_regr, 
            optimizer, epoch, device)
        if best_loss_cls > epoch_loss_cls or best_loss_regr > epoch_loss_regr or best_loss > epoch_loss:
            best_loss = epoch_loss
            best_loss_regr = epoch_loss_regr
            best_loss_cls = epoch_loss_cls
            check_path = os.path.join(opt.save_folder,
                                      f'ctpn_done_ep{epoch:02d}_'
                                      f'{best_loss_cls:.4f}_{best_loss_regr:.4f}_{best_loss:.4f}.pth')
            torch.save({'model_state_dict': model.state_dict(), 
                        'epoch': epoch,
                        'optimizer': optimizer.state_dict(),
                        'best_loss': best_loss,
                        'version': 'vgg'}, check_path)
        scheduler.step()
        epoch += 1