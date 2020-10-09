# -*- coding: utf-8 -*-
import cv2
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from xml.dom.minidom import parse
from textdetection import params
from tools.utils_ctpn import gen_gt_from_quadrilaterals, cal_rpn


class resizeNormalize(object):
    def __init__(self, size, interpolation=cv2.INTER_CUBIC, is_test=True):
        self.size = size
        self.interpolation = interpolation
        self.is_test = is_test

    def __call__(self, img):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h0, w0 = img.shape
        w, h = self.size
        if w <= (w0 / h0 * h):
            image = cv2.resize(img, (w, h), interpolation=self.interpolation)
            image = image.astype(np.float32) / 255.
            image = torch.from_numpy(image).type(torch.FloatTensor)
            image.sub_(params.mean).div_(params.std)
            image = image.view(1, *image.size())
        else:
            w_real = int(w0 / h0 * h)
            image = cv2.resize(img, (w_real, h), interpolation=self.interpolation)
            image = image.astype(np.float32) / 255.
            image = torch.from_numpy(image).type(torch.FloatTensor)
            image.sub_(params.mean).div_(params.std)
            tmp = torch.zeros([1, h, w])
            start = random.randint(0, w - w_real)
            if self.is_test:
                start = 0
            tmp[:, :, start:start + w_real] = image
            image = tmp

        return image


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
        image = cv2.imread(image_name)
        h, w, c = image.shape
        gt_boxes, class_ids = gen_gt_from_quadrilaterals(gt_boxes, labels, image.shape, params.ANCHORS_WIDTH)

        rescale_fac = max(h, w) / 800
        if rescale_fac > 1.0:
            h = int(h / rescale_fac)
            w = int(w / rescale_fac)
            image = cv2.resize(image,(w,h))
            gt_boxes = gt_boxes / rescale_fac
        
        image = image - params.IMAGE_MEAN
        image = torch.from_numpy(image.transpose([2, 0, 1])).float()
        [clss, regr], base_anchors = cal_rpn((h, w), (int(h / 16), int(w / 16)), 16, gt_boxes)
        regr = np.hstack([clss.reshape(clss.shape[0], 1), regr])
        regr = torch.from_numpy(regr).float()
        clss = np.expand_dims(clss, axis=0)
        clss = torch.from_numpy(clss).float()

        return image, regr, clss, index


if __name__ == '__main__':
    train_dataset = imgDataset('D:/80dataset/ocr/DataSet/testxx/images', 'D:/80dataset/ocr/DataSet/testxx/train.list', 
                                params.alphabet, (params.imgW, params.imgH), params.mean, params.std)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)
	
    for i_batch, (image, label, index) in enumerate(train_dataloader):
        print(image.shape)
        print(label)
