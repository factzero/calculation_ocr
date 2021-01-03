# -*- coding: utf-8 -*-
import argparse
import os
import glob
from PIL import Image
import matplotlib.pyplot as plt
import random
from tqdm import tqdm


parser = argparse.ArgumentParser(description='gen data')
parser.add_argument('--image_root', default='./data/', type=str, help='image root dir')


if __name__ == "__main__":
    opt = parser.parse_args()
    angle0_image_root = os.path.join(opt.image_root, '0')
    angle90_image_root = os.path.join(opt.image_root, '90')
    angle180_image_root = os.path.join(opt.image_root, '180')
    angle270_image_root = os.path.join(opt.image_root, '270')

    angle0_images = glob.glob(os.path.join(angle0_image_root, '*.jpg'))
    num_images = len(angle0_images)
    for i, angle0_image_name in tqdm(enumerate(angle0_images), total=len(angle0_images), desc='test model'):
    # for angle0_image_name in angle0_images:
        angle0_image = Image.open(angle0_image_name)
        angle90_image = angle0_image.transpose(Image.ROTATE_270)
        angle180_image = angle0_image.transpose(Image.ROTATE_180)
        angle270_image = angle0_image.transpose(Image.ROTATE_90)
        new_image_name = str(random.randint(0, 1000*num_images)) + '.jpg'
        angle90_image.save(os.path.join(angle90_image_root, new_image_name))
        angle180_image.save(os.path.join(angle180_image_root, new_image_name))
        angle270_image.save(os.path.join(angle270_image_root, new_image_name))

        # plt.figure()
        # plt.subplot(2,2,1)
        # plt.imshow(angle0_image)
        # plt.subplot(2,2,2)
        # plt.imshow(angle90_image)
        # plt.subplot(2,2,3)
        # plt.imshow(angle180_image)
        # plt.subplot(2,2,4)
        # plt.imshow(angle270_image)
        # plt.show()


