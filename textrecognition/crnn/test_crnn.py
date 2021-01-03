# -*- coding: utf-8 -*-
import os
import argparse
import yaml
from easydict import EasyDict as edict
from PIL import Image
import numpy as np
import torch
from crnn_keys import alphabet
from crnn_model import get_crnn
import crnn_utils as utils


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--cfg', default='./textrecognition/crnn/config.yaml', type=str, help='experiment configuration filename')
parser.add_argument('--image_name', default='./data/img_0987732.jpg', type=str, help='image for processing')
parser.add_argument('--model_path', default='./checkpoints/crnn_Rec_best.pth', type=str, help='trained model path')

if __name__ == "__main__":
    args = parser.parse_args()
    with open(args.cfg, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)
    config.DATASET.ALPHABETS = alphabet
    config.MODEL.NUM_CLASSES = len(config.DATASET.ALPHABETS)
    print('alphabet length : ', config.MODEL.NUM_CLASSES)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = get_crnn(config).to(device)
    if args.model_path != '' and os.path.exists(args.model_path):
        print('loading pretrained model from %s' % args.model_path)
        model.load_state_dict(torch.load(args.model_path))
    
    image = Image.open(args.image_name).convert("L")
    w, h = image.size
    new_w = int(w / h * config.MODEL.IMAGE_SIZE.H)
    image = image.resize((new_w, config.MODEL.IMAGE_SIZE.H))
    image = np.array(image).astype(np.float32)
    image = (image/255.0 - config.DATASET.MEAN)/config.DATASET.STD
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=0)
    image = torch.from_numpy(image)

    converter = utils.strLabelConverter(alphabet)
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        preds = model(image)
        preds_size = torch.IntTensor([preds.size(0)] * image.size(0))
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        sim_preds = converter.decode(preds.data, preds_size.data, raw=False)
        print(sim_preds)