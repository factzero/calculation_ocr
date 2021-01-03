# -*- coding: utf-8 -*-
import argparse
import yaml
from easydict import EasyDict as edict
import time
import torch
import torch.utils.data as data
from tqdm import tqdm
from crnn_keys import alphabet
import crnn_utils as utils
from crnn_model import get_crnn
from crnn_dataset import OcrDataset


def parse_arg():
    parser = argparse.ArgumentParser(description="train crnn")
    parser.add_argument('--cfg', default='./textrecognition/crnn/config.yaml', type=str, help='experiment configuration filename')
    parser.add_argument('--image_root', default='', type=str, help='train image root dir')
    parser.add_argument('--val_labels', default='', type=str, help='val label')
    parser.add_argument('--resume_net', default='', help='resume net')
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    
    if args.resume_net != '':
        config.TRAIN.RESUME.IS_RESUME = True
        config.TRAIN.RESUME.FILE = args.resume_net
    
    if args.image_root != '':
        config.DATASET.ROOT = args.image_root

    if args.val_labels != '':
        config.DATASET.JSON_FILE['val'] = args.val_labels
        
    config.DATASET.ALPHABETS = alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    print('alphabet length : ', config.MODEL.NUM_CLASSES)

    return config


def validate(config, loader, converter, model, criterion, device):
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

    return accuracy


def main():
    config = parse_arg()

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

    val_dataset = OcrDataset(config, is_train=False)
    val_dataloader = data.DataLoader(
        dataset=val_dataset,
        batch_size=config.TEST.BATCH_SIZE_PER_GPU,
        shuffle=config.TEST.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    converter = utils.strLabelConverter(alphabet)
    acc = validate(config, val_dataloader, converter, model, criterion, device)
        

if __name__ == "__main__":
    main()