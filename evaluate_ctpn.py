# -*- coding: utf-8 -*-
import os
import argparse
import time
import torch
from torch.utils.data import Dataset, DataLoader
from textdetection.ctpn.ctpn_dataset import vocDataset
from textdetection.ctpn.ctpn_model import CTPN_Model, RPN_CLS_Loss, RPN_REGR_Loss


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--data_root', default='', type=str, help='data root in voc2007')
parser.add_argument('--model_path', default='./checkpoints/CTPN.pth', type=str, help='trained model path')
parser.add_argument('--batch_size', default=2, type=int, help='batch size')


def val(model, loader, criterion_cls, criterion_regr, device):
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    total_cls_loss = 0
    total_v_reg_loss = 0
    total_loss = 0
    epoch_size = len(loader)
    start_time = time.time()
    for i_batch, (image, gt_boxes, clss, index) in enumerate(loader):
        gt_boxes = gt_boxes.to(device)
        clss = clss.to(device)
        image = image.to(device)

        out_cls, out_regr = model(image)
        loss_cls = criterion_cls(out_cls, clss)
        loss_regr = criterion_regr(out_regr, gt_boxes)
        loss = loss_cls + loss_regr   


        total_cls_loss += loss_cls.item()
        total_v_reg_loss += loss_regr.item()
        total_loss += loss.item()
        print(f'Batch:{i_batch}/{epoch_size}')
    end_time = time.time()
    total_time = end_time - start_time
    
    total_cls_loss /= epoch_size
    total_v_reg_loss /= epoch_size
    total_loss /= epoch_size

    print('####################  Start evaluate  ####################')
    print('loss: {0}'.format(total_loss))
    print('classification loss: {0}'.format(total_cls_loss))
    print('vertical regression loss: {0}'.format(total_v_reg_loss))
    print('{0} iterations for {1} seconds, avg {2} seconds.'.format(epoch_size, total_time, total_time/epoch_size))
    print('#####################  Evaluate end  #####################')
    print('\n')

    return total_cls_loss, total_v_reg_loss, total_loss


if __name__ == "__main__":
    opt = parser.parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CTPN_Model()
    criterion_cls = RPN_CLS_Loss(device)
    criterion_regr = RPN_REGR_Loss(device)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()

    val_dataset = vocDataset(opt.data_root)
    val_dataloader = DataLoader(val_dataset, batch_size=opt.batch_size, shuffle=False, num_workers=4)
    
    if opt.model_path !='' and os.path.exists(opt.model_path):
        print('loading pretrained model from %s' % opt.model_path)
        cc = torch.load(opt.model_path, map_location=device)
        model.load_state_dict(cc['model_state_dict'])
        loss_cls, loss_regr, loss = val(model, val_dataloader, criterion_cls, criterion_regr, device)

