# -*- coding: utf-8 -*-
import argparse
import cv2
import time
from textrecognition.crnnCTC.crnn_inference import OcrTextRec


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--image_name', default='./data/img_0987732.jpg', type=str, help='image for processing')
parser.add_argument('--model_path', default='./checkpoints/crnn_Rec_best.pth', type=str, help='trained model path')


if __name__ == '__main__':
    opt = parser.parse_args()
    image_name = opt.image_name
    model_path = opt.model_path
    recognizer = OcrTextRec(model_path)
    image = cv2.imread(image_name)
    started = time.time()
    pred = recognizer.inference(image)
    finished = time.time()
    print('results: {0} '.format(pred))
    print('elapsed time: {0}'.format(finished-started))
    