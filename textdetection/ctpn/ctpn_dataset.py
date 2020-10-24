# -*- coding: utf-8 -*-
import cv2
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from xml.dom.minidom import parse
import ctpn_params 
from ctpn_utils import gen_gt_from_quadrilaterals, cal_rpn, resize_image2square, adj_gtboxes


class vocDataset(Dataset):
    def __init__(self, voc_dir, is_aug=True):
        super(vocDataset, self).__init__()
        self.voc_dir = voc_dir
        self.imgs = self.get_trainval_imgs(os.path.join(self.voc_dir, "ImageSets", "Main", "trainval.txt"))

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
        image = image - ctpn_params.IMAGE_MEAN
        image = torch.from_numpy(image.transpose([2, 0, 1])).float()

        # 计算rpn
        [clss, regr], base_anchors = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gt_boxes)
        # 数据按[(label, Vc, Vh)]存放
        regr = np.hstack([clss.reshape(clss.shape[0], 1), regr])
        regr = torch.from_numpy(regr).float()

        clss = np.expand_dims(clss, axis=0)
        clss = torch.from_numpy(clss).float()

        return image, regr, clss, index
