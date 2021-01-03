# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


class imgDataset(Dataset):
    def __init__(self, imgs_dir, labels_dir, transform=None, is_train=True):
        super(imgDataset, self).__init__()
        self.imgs_dir = imgs_dir
        self.labels = self.get_labels(labels_dir)
        self.transform = transform
        self.is_train = is_train
        self.labels_index = {'0':0, '90':1, '180':2, '270':3}

    def get_labels(self, label_path):
        with open(label_path, 'r', encoding='utf-8') as file:
            labels = [ {c.split(' ')[0]:c.split(' ')[-1][:-1]}for c in file.readlines()]	
			
        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = list(self.labels[index].values())[0]
        image_name = list(self.labels[index].keys())[0]
        image_name = os.path.join(self.imgs_dir, image_name)
        image = Image.open(image_name).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return image, self.labels_index[label], index


if __name__ == '__main__':
    print('testing angle class dataset')
    # transform_train = transforms.Compose([
    #     transforms.Resize((256, 256)),
    #     transforms.RandomRotation(10),
    #     transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
    #     transforms.RandomCrop((224, 224)),
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    # ])
    transform_train = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.RandomCrop((224, 224)),
        transforms.ToTensor()
    ])
    train_dataset = imgDataset('D:/80dataset/ocr/angle/images', 'D:/80dataset/ocr/angle/train.list', transform_train)
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
	
    for i_batch, (image, label, index) in enumerate(train_dataloader):
        print(image.shape)
        print(label)
        arrayShow = transforms.ToPILImage()(image[0])
        print(arrayShow.size)
        plt.imshow(arrayShow)
        plt.show()
        arrayShow = arrayShow.transpose(Image.ROTATE_90)
        plt.imshow(arrayShow)
        plt.show()
