# -*- coding: utf-8 -*-
import argparse
import cv2
import time
from textdetection.ctpn.ctpn_inference import OcrDetCTPN


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--image_name', default='./data/tx6.jpg', type=str, help='image for processing')
parser.add_argument('--model_path', default='./checkpoints/CTPN.pth', type=str, help='trained model path')


if __name__ == '__main__':
    opt = parser.parse_args()
    detect = OcrDetCTPN(opt.model_path)
    image = cv2.imread(opt.image_name)
    started = time.time()
    det_recs = detect.inference(image)
    finished = time.time()
    for res in det_recs:
        cv2.line(image, (int(res[0]), int(res[1])), (int(res[2]), int(res[3])), (0, 0, 255), 8)
        cv2.line(image, (int(res[0]), int(res[1])), (int(res[4]), int(res[5])), (0, 0, 255), 8)
        cv2.line(image, (int(res[6]), int(res[7])), (int(res[2]), int(res[3])), (0, 0, 255), 8)
        cv2.line(image, (int(res[4]), int(res[5])), (int(res[6]), int(res[7])), (0, 0, 255), 8)
    image = cv2.resize(image, (1000, 800))
    cv2.imshow("image", image)
    cv2.waitKey(0)
    print('elapsed time: {0}'.format(finished-started))