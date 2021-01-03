# -*- coding: utf-8 -*-
import os
import argparse
import yaml
from easydict import EasyDict as edict
import time
import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter
from tqdm import tqdm
from crnn_keys import alphabet
import crnn_utils as utils
from crnn_model import get_crnn
from crnn_dataset import OcrDataset


def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")
    parser.add_argument('--cfg', default='./textrecognition/crnn/config.yaml', type=str, help='experiment configuration filename')
    parser.add_argument('--resume_net', default='', help='resume net')
    parser.add_argument('--resume_iter', default=0, type=int, help='resume epoch')
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    
    if args.resume_net != '':
        config.TRAIN.RESUME.IS_RESUME = True
        config.TRAIN.RESUME.FILE = args.resume_net
    config.TRAIN.BEGIN_EPOCH = args.resume_iter

    config.DATASET.ALPHABETS = alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    print('alphabet length : ', config.MODEL.NUM_CLASSES)

    return config


def train(config, loader, converter, model, criterion, optimizer, device, epoch, writer_dict=None):
    model.train()

    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    losses = utils.AverageMeter()

    start_time = time.time()
    for i, (image, label, index) in enumerate(loader):
        data_time.update(time.time() - start_time)

        optimizer.zero_grad()
        model.zero_grad()
        
        image = image.to(device)
        preds = model(image)
        
        batch_size = image.size(0)
        text, length = converter.encode(label)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        loss = criterion(preds, text, preds_size, length)
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), image.size(0))
        batch_time.update(time.time() - start_time)

        if i % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                  'Speed {speed:.1f} samples/s\t' \
                  'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                  'Loss {loss.val:.5f} ({loss.avg:.5f})\t'.format(
                      epoch, i, len(loader), batch_time=batch_time,
                      speed=image.size(0)/batch_time.val,
                      data_time=data_time, loss=losses)
            print(msg)

            if writer_dict:
                writer = writer_dict['writer']
                global_steps = writer_dict['train_global_steps']
                writer.add_scalar('train_loss', losses.avg, global_steps)
                writer_dict['train_global_steps'] = global_steps + 1
        
        start_time = time.time()


def validate(config, loader, converter, model, criterion, device, epoch, writer_dict=None):
    losses = utils.AverageMeter()
    model.eval()

    n_correct = 0
    total_n = 0
    with torch.no_grad():
        for i, (image, labels, index) in tqdm(enumerate(loader), total=len(loader), desc='test model'):
            image = image.to(device)
            preds = model(image)
            
            batch_size = image.size(0)
            text, length = converter.encode(labels)
            preds_size = torch.IntTensor([preds.size(0)] * batch_size)
            loss = criterion(preds, text, preds_size, length)

            losses.update(loss.item(), image.size(0))

            total_n += image.shape[0]
            _, preds = preds.max(2)
            preds = preds.transpose(1, 0).contiguous().view(-1)
            sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
            for pred, target in zip(sim_preds, labels):
                if pred == target:
                    n_correct += 1

    raw_preds = converter.decode(preds.data, preds_size.data, raw=True)[:config.TEST.NUM_TEST_DISP]
    for raw_pred, pred, gt in zip(raw_preds, sim_preds, labels):
        print('%-20s => %-20s, gt: %-20s' % (raw_pred, pred, gt))

    accuracy = float(n_correct) / total_n
    print('Test loss: {:.4f}, accuray: {:.4f}'.format(losses.avg, accuracy))

    if writer_dict:
        writer = writer_dict['writer']
        global_steps = writer_dict['valid_global_steps']
        writer.add_scalar('valid_acc', accuracy, global_steps)
        writer_dict['valid_global_steps'] = global_steps + 1

    return accuracy


def main():
    config = parse_arg()

    output_dict = utils.create_log_folder(config, phase='train')
    # writer dict
    writer_dict = {
        'writer': SummaryWriter(log_dir=output_dict['tb_dir']),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    last_epoch = config.TRAIN.BEGIN_EPOCH

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_crnn(config).to(device)
    if config.TRAIN.RESUME.IS_RESUME:
        model_state_file = config.TRAIN.RESUME.FILE
        if model_state_file != '' and os.path.exists(model_state_file):
            print('loading pretrained model from %s' % model_state_file)
            model.load_state_dict(torch.load(model_state_file))

    criterion = torch.nn.CTCLoss(reduction='sum').to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.TRAIN.LR)
    if isinstance(config.TRAIN.LR_STEP, list):
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch - 1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR, last_epoch - 1)
    
    train_dataset = OcrDataset(config, is_train=True)
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = OcrDataset(config, is_train=False)
    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    best_acc = 0.01
    converter = utils.strLabelConverter(alphabet)
    for epoch in range(last_epoch, config.TRAIN.END_EPOCH):
        train(config, train_dataloader, converter, model, criterion, optimizer, device, epoch, writer_dict)
        lr_scheduler.step()
        acc = validate(config, val_dataloader, converter, model, criterion, device, epoch, writer_dict)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), '{0}/crnn_Rec_done_{1:04d}_{2:.4f}.pth'.format(output_dict['chs_dir'], epoch, acc))
            torch.save(model.state_dict(), '{0}/crnn_Rec_best.pth'.format(output_dict['chs_dir']))
    
    writer_dict['writer'].close()
        

if __name__ == "__main__":
    main()