# -*- coding: utf-8 -*-
import cv2
import os
import argparse
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from xml.dom.minidom import parse
import ctpn_params 
from ctpn_utils import gen_gt_from_quadrilaterals, cal_rpn, resize_image2square, adj_gtboxes


class vocDataset(Dataset):
    def __init__(self, voc_dir, is_aug=True, is_debug=False):
        super(vocDataset, self).__init__()
        self.voc_dir = voc_dir
        self.imgs = self.get_trainval_imgs(os.path.join(self.voc_dir, "ImageSets", "Main", "trainval.txt"))
        self.is_debug = is_debug

    def get_trainval_imgs(self, trainval_txt):
        imgs = []
        with open(trainval_txt, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            imgs.append(line.rstrip("\n"))
			
        return imgs

    def __len__(self):
        return len(self.imgs)

    def read_xml(self, p):
        domTree = parse(p)
        rootNode = domTree.documentElement
        imgname = rootNode.getElementsByTagName("image")[0].childNodes[0].data

        gt_boxes = []
        labels = []
        objects = rootNode.getElementsByTagName("object")
        for obj in objects:
            label = obj.getElementsByTagName("label")[0].childNodes[0].data
            bndbox = obj.getElementsByTagName("bndbox")
            x1 = int(obj.getElementsByTagName("x1")[0].childNodes[0].data)
            y1 = int(obj.getElementsByTagName("y1")[0].childNodes[0].data)
            x2 = int(obj.getElementsByTagName("x2")[0].childNodes[0].data)
            y2 = int(obj.getElementsByTagName("y2")[0].childNodes[0].data)
            x3 = int(obj.getElementsByTagName("x3")[0].childNodes[0].data)
            y3 = int(obj.getElementsByTagName("y3")[0].childNodes[0].data)
            x4 = int(obj.getElementsByTagName("x4")[0].childNodes[0].data)
            y4 = int(obj.getElementsByTagName("y4")[0].childNodes[0].data)
            labels.append((label))
            gt_boxes.append((x1, y1, x2, y2, x3, y3, x4, y4))
            
        return np.array(gt_boxes), labels, imgname 

    def __getitem__(self, index):
        img = self.imgs[index]
        xml_file = os.path.join(self.voc_dir, "Annotations", (img + ".xml"))
        gt_boxes, labels, image_name = self.read_xml(xml_file)
        image_name = os.path.join(self.voc_dir, "JPEGImages", image_name)
        # 读取图片，如果尺寸不是ctpn_params.IMAGE_HEIGHT*ctpn_params.IMAGE_HEIGHT，等比例缩放至该尺寸
        image = cv2.imread(image_name)
        if image.shape[0] != ctpn_params.IMAGE_HEIGHT or image.shape[1] != ctpn_params.IMAGE_HEIGHT:
            image, rescale_fac, padding = resize_image2square(image, ctpn_params.IMAGE_HEIGHT)
            gt_boxes = adj_gtboxes(gt_boxes, rescale_fac, padding)
        # 将大标定框分割成宽度是ctpn_params.ANCHORS_WIDTH的小框
        gt_boxes, class_ids = gen_gt_from_quadrilaterals(gt_boxes, labels, image.shape, ctpn_params.ANCHORS_WIDTH)
        
        h, w, c = image.shape
        if self.is_debug == False:
            image = image - ctpn_params.IMAGE_MEAN
        image = torch.from_numpy(image.transpose([2, 0, 1])).float()

        # 计算rpn
        [clss, regr], base_anchors = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gt_boxes)
        # 数据按[(label, Vc, Vh)]存放
        regr = np.hstack([clss.reshape(clss.shape[0], 1), regr])
        regr = torch.from_numpy(regr).float()

        clss = np.expand_dims(clss, axis=0)
        clss = torch.from_numpy(clss).float()

        if self.is_debug:
            return image, gt_boxes, clss, index
        else:
            return image, regr, clss, index


def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)): # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train')
    parser.add_argument('--image_root', default='D:/80dataset/ocr/VOC2007_test', type=str, help='image root dir')
    opt = parser.parse_args()

    train_dataset = vocDataset(opt.image_root, is_debug=True)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    dataset_size = len(train_dataloader)
	
    # cv2.namedWindow('input_image', 0)
    for i_batch, (image, gtboxes, labels, index) in enumerate(train_dataloader):
        # cv2.resizeWindow('input_image', 1000, 800)
        image = image.squeeze(0).numpy().transpose([1, 2, 0]).astype(np.uint8)
        gtboxes = gtboxes.squeeze(0).numpy()
        image = np.ascontiguousarray(image)
        # for gtbox, label in zip(gtboxes, labels):
        # for gtbox in gtboxes:
        #     cv2.line(image, (int(gtbox[0]), int(gtbox[1])), (int(gtbox[2]), int(gtbox[3])), (255, 0, 0), 4)
        #     cv2.line(image, (int(gtbox[2]), int(gtbox[3])), (int(gtbox[4]), int(gtbox[5])), (255, 0, 0), 4)
        #     cv2.line(image, (int(gtbox[4]), int(gtbox[5])), (int(gtbox[6]), int(gtbox[7])), (255, 0, 0), 4)
        #     cv2.line(image, (int(gtbox[6]), int(gtbox[7])), (int(gtbox[0]), int(gtbox[1])), (255, 0, 0), 4)
        #     image = cv2ImgAddText(image, str(label[0]), gtbox[2], gtbox[3], (255, 0, 0), 80)
        for gtbox in gtboxes:
            cv2.line(image, (int(gtbox[0]), int(gtbox[1])), (int(gtbox[2]), int(gtbox[1])), (0, 0, 255), 2)
            cv2.line(image, (int(gtbox[2]), int(gtbox[1])), (int(gtbox[2]), int(gtbox[3])), (0, 0, 255), 2)
            cv2.line(image, (int(gtbox[2]), int(gtbox[3])), (int(gtbox[0]), int(gtbox[3])), (0, 0, 255), 2)
            cv2.line(image, (int(gtbox[0]), int(gtbox[3])), (int(gtbox[0]), int(gtbox[1])), (0, 0, 255), 2)
        cv2.imshow("input_image", image)
        cv2.waitKey(0)
        print(f'Batch:{i_batch}/{dataset_size}\n')
        print(image.shape)