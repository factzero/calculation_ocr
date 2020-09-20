# -*- coding: utf-8 -*-
import cv2
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from textrecognition import params
from PIL import Image


class imgDataset(Dataset):
    def __init__(self, imgs_dir, labels_dir, alphabet, resize, mean, std):
        super(imgDataset, self).__init__()
        self.imgs_dir = imgs_dir
        self.labels = self.get_labels(labels_dir)
        self.alphabet = alphabet
        self.width, self.height = resize
        self.mean = mean
        self.std = std

    def get_labels(self, label_path):
        with open(label_path, 'r', encoding='utf-8') as file:
            labels = [ {c.split(' ')[0]:c.split(' ')[-1][:-1]}for c in file.readlines()]	
			
        return labels

    def __len__(self):
        return len(self.labels)

    def preprocessing(self, image):
        image = image.astype(np.float32) / 255.
        image = torch.from_numpy(image).type(torch.FloatTensor)
        image.sub_(self.mean).div_(self.std)

        return image

    def __getitem__(self, index):
        label = list(self.labels[index].values())[0]
        image_name = list(self.labels[index].keys())[0]
        image_name = os.path.join(self.imgs_dir, image_name)
        image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
        h, w = image.shape
        w_new = int(w/h*params.imgH)
        image = cv2.resize(image, (w_new, self.height), interpolation=cv2.INTER_CUBIC)
        image = (np.reshape(image, (self.height, w_new, 1))).transpose(2, 0, 1)
        image = self.preprocessing(image)

        return image, label, index


class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(params.mean).div_(params.std)        
        return img


if __name__ == '__main__':
    train_dataset = imgDataset('D:/80dataset/ocr/DataSet/testxx/images', 'D:/80dataset/ocr/DataSet/testxx/train.list', 
                                params.alphabet, (params.imgW, params.imgH), params.mean, params.std)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)
	
    for i_batch, (image, label, index) in enumerate(train_dataloader):
        print(image.shape)
        print(label)