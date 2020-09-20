# -*- coding: utf-8 -*-
import cv2
import numpy as np
import torch
from textrecognition import params
from textrecognition.crnn import CRNN
from tools import utils


class OcrTextRec():
    def __init__(self, model_path='./checkpoints/CRNN.pth'):
        self.alphabet = ''.join([chr(uni) for uni in params.alphabet])
        self.nclass = len(self.alphabet) + 1
        self.model = CRNN(params.imgH, 1, self.nclass, 256)
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
        self.model.load_state_dict(torch.load(model_path))
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.converter = utils.strLabelConverter(self.alphabet)

    def inference(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = image.shape
        w_new = int(w/h*params.imgH)
        image = cv2.resize(image, (w_new, params.imgH), interpolation=cv2.INTER_CUBIC)
        image = (np.reshape(image, (params.imgH, w_new, 1))).transpose(2, 0, 1)
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image.sub_(params.mean).div_(params.std)
        image = image.view(1, *image.size())
        if self.use_gpu:
            image = image.cuda()
        
        preds = self.model(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        preds_size = torch.IntTensor([preds.size(0)])
        sim_pred = self.converter.decode(preds.data, preds_size.data, raw=False)

        return sim_pred


if __name__ == "__main__":
    model_path = './checkpoints/CRNN.pth'
    recognizer = OcrTextRec(model_path)
    image_name = './data/20436218_1024524228.jpg'
    image = cv2.imread(image_name)
    pred = recognizer.inference(image)
    print(pred)

