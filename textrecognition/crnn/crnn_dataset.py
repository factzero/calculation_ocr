# -*- coding: utf-8 -*-
import os
import torch.utils.data as data
from PIL import Image
import numpy as np
from data_augment import dataAugment


class resizeNormalize():
    def __init__(self, cfg, is_train=True):
        self.MEAN = cfg.DATASET.MEAN
        self.STD = cfg.DATASET.STD
        self.img_w = cfg.MODEL.IMAGE_SIZE.W
        self.img_h = cfg.MODEL.IMAGE_SIZE.H
        self.is_train = is_train

    def __call__(self, img):
        w, h = img.size
        new_h = self.img_h
        new_w = int(w / h * new_h)
        if self.is_train:
            if new_w < w:
                img_sz = img.resize((new_w, new_h))
                image = Image.new('L', (self.img_w, self.img_h), (128))
                image.paste(img_sz)
            else:
                image = img.resize((self.img_w, self.img_h))
        else:
            image = img.resize((new_w, new_h))
        image = np.array(image).astype(np.float32)
        image = (image/255.0 - self.MEAN)/self.STD

        return image


class OcrDataset(data.Dataset):
    def __init__(self, cfg, transform=None, is_train=True):
        super(OcrDataset, self).__init__()
        self.img_root = cfg.DATASET.ROOT
        txt_file = cfg.DATASET.JSON_FILE['train'] if is_train else cfg.DATASET.JSON_FILE['val']
        self.labels = self.get_labels(txt_file)
        self.transform = transform
        self.is_train = is_train
        if self.is_train:
            self.data_aug = dataAugment()
        self.resizenorm = resizeNormalize(cfg, is_train)

    def get_labels(self, label_path):
        with open(label_path, 'r', encoding='utf-8') as file:
            labels = [ {c.split(' ')[0]:c.split(' ')[-1][:-1]}for c in file.readlines()]	
        print("load {} images! ".format(len(labels)))

        return labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label = list(self.labels[index].values())[0]
        image_name = list(self.labels[index].keys())[0]
        image_name = os.path.join(self.img_root, image_name)
        image = Image.open(image_name).convert("L")
        if self.transform is not None:
            image = self.transform(image)
        else:
            if self.is_train:
                image = self.data_aug(image)
            image = self.resizenorm(image)
            image = np.expand_dims(image, axis=0)

        return image, label, index


if __name__ == "__main__":
    import yaml
    from easydict import EasyDict as edict
    import torchvision.transforms as transforms
    import matplotlib.pyplot as plt

    with open('./textrecognition/crnn/config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config = edict(config)

    train_transform = transforms.Compose([
        transforms.RandomRotation(3),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        transforms.Resize((config.MODEL.IMAGE_SIZE.H, config.MODEL.IMAGE_SIZE.W)),
        transforms.ToTensor(),
        transforms.Normalize((config.DATASET.MEAN), (config.DATASET.STD))
    ])
    
    # train_dataset = OcrDataset(config, transform=train_transform, is_train=True)
    train_dataset = OcrDataset(config, is_train=True)
    train_dataloader = data.DataLoader(
        dataset=train_dataset,
        batch_size=config.TRAIN.BATCH_SIZE_PER_GPU,
        shuffle=config.TRAIN.SHUFFLE,
        num_workers=config.WORKERS,
        pin_memory=config.PIN_MEMORY,
    )

    val_dataset = OcrDataset(config, transform=train_transform, is_train=False)

    for i, (image, label, index) in enumerate(train_dataloader):
        print(image.shape)
        print(label)
        arrayShow = transforms.ToPILImage()(image[0])
        print(arrayShow.size)
        plt.imshow(arrayShow)
        plt.show()