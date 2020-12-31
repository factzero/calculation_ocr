# -*- coding: utf-8 -*-
import cv2
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from textrecognition.crnnCTC import crnn_params
from data_augment import dataAugment


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
            image.sub_(crnn_params.mean).div_(crnn_params.std)
            image = image.view(1, *image.size())
        else:
            w_real = int(w0 / h0 * h)
            image = cv2.resize(img, (w_real, h), interpolation=self.interpolation)
            image = image.astype(np.float32) / 255.
            image = torch.from_numpy(image).type(torch.FloatTensor)
            image.sub_(crnn_params.mean).div_(crnn_params.std)
            tmp = torch.zeros([1, h, w])
            start = random.randint(0, w - w_real)
            if self.is_test:
                start = 0
            tmp[:, :, start:start + w_real] = image
            image = tmp

        return image


class imgDataset(Dataset):
    def __init__(self, imgs_dir, labels_dir, alphabet, resize, mean, std, is_aug=True):
        super(imgDataset, self).__init__()
        self.imgs_dir = imgs_dir
        self.labels = self.get_labels(labels_dir)
        self.alphabet = alphabet
        self.width, self.height = resize
        self.mean = mean
        self.std = std
        self.is_aug = is_aug

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

    # def __getitem__(self, index):
    #     label = list(self.labels[index].values())[0]
    #     image_name = list(self.labels[index].keys())[0]
    #     image_name = os.path.join(self.imgs_dir, image_name)
    #     image = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    #     transformer = resizeNormalize((params.imgW, params.imgH))
    #     image = transformer(image)

    #     return image, label, index
    def __getitem__(self, index):
        label = list(self.labels[index].values())[0]
        image_name = list(self.labels[index].keys())[0]
        image_name = os.path.join(self.imgs_dir, image_name)
        image = cv2.imread(image_name)
        if self.is_aug:
            data_aug = dataAugment()
            image = data_aug(image)
        transformer = resizeNormalize((self.width, self.height))
        image = transformer(image)

        return image, label, index



if __name__ == '__main__':
    train_dataset = imgDataset('D:/80dataset/ocr/DataSet/testxx/images', 'D:/80dataset/ocr/DataSet/testxx/train.list', 
                               crnn_params.alphabet, (crnn_params.imgW, crnn_params.imgH), crnn_params.mean, crnn_params.std)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=False)
	
    for i_batch, (image, label, index) in enumerate(train_dataloader):
        print(image.shape)
        print(label)
