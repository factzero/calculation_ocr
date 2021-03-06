# -*- coding: utf-8 -*-
import sys, os
sys.path.append("./textdetection")
sys.path.append("./textdetection/cal_recall")
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from ctpn_inference import OcrDetCTPN
from cal_recall.script import cal_recall_precison_f1


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--image_root', default='D:/80dataset/ocr/icdar2015TextLocaltion/test_im', type=str, help='image root')
parser.add_argument('--model_path', default='./checkpoints/CTPN.pth', type=str, help='trained model path')
parser.add_argument('--test_gt', default='D:/80dataset/ocr/icdar2015TextLocaltion/test_gt', type=str, help='test image groud truth')


if __name__ == "__main__":
    opt = parser.parse_args()
    txt_save_path = './pre_gt'
    img_save_path = './result'
    if not os.path.exists(txt_save_path):
        os.mkdir(txt_save_path)
    if not os.path.exists(img_save_path):
        os.mkdir(img_save_path)

    detect = OcrDetCTPN(opt.model_path)
    image_files = os.listdir(opt.image_root)
    for i in tqdm(range(len(image_files))):
        image_file = image_files[i]
        im_name = image_file.split('.')[0]
        fid = open(os.path.join(txt_save_path, 'res_' + im_name + '.txt'), 'w+', encoding='utf-8')
        image_file = os.path.join(opt.image_root, image_file)
        image = cv2.imread(image_file)
        det_recs = detect.inference(image)
        for i, box in enumerate(det_recs):
            x3, y3 = box[6], box[7]
            box[6], box[7] = box[4], box[5]
            box[4], box[5] = x3, y3
            box = box[:8].reshape(4, 2).astype(np.int32)
            cv2.polylines(image, [box[:8].reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=2)
            box = [str(x) for x in box.reshape(-1).tolist()]
            fid.write(','.join(box) + '\n')

        cv2.imwrite(os.path.join(img_save_path, im_name + '.jpg'), image)
        fid.close()

    pre = cal_recall_precison_f1(opt.test_gt, txt_save_path, show_result=False)
    print(pre)