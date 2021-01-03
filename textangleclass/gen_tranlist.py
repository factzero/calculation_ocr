# -*- coding: utf-8 -*-
import argparse
import os
import glob
import random


parser = argparse.ArgumentParser(description='gen list')
parser.add_argument('--image_root', default='./data/', type=str, help='image root dir')
parser.add_argument('--train_list', default='train.list', type=str, help='train list name')
parser.add_argument('--val_list', default='val.list', type=str, help='val list name')


if __name__ == "__main__":
    opt = parser.parse_args()

    angles = ['0', '90', '180', '270']
    train_lines = []
    val_lines = []
    for angle in angles:
        angle_image_root = os.path.join(opt.image_root, angle)
        list_dir = os.listdir(angle_image_root)
        for name in list_dir:
            name = os.path.join(angle, name) + ' ' + angle + '\n' 
            if 0 == random.randint(0, 4):
                val_lines.append(name)
            else:
                train_lines.append(name)

    trainlist_file = os.path.join(opt.image_root, opt.train_list)
    with open(trainlist_file, 'w') as f:
        f.writelines(train_lines)
    
    vallist_file = os.path.join(opt.image_root, opt.val_list)
    with open(vallist_file, 'w') as f:
        f.writelines(val_lines)
        
            
    
