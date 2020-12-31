# -*- coding: utf-8 -*-
import argparse
import os
import glob
import random


parser = argparse.ArgumentParser(description='gen list')
parser.add_argument('--total', default='labels.txt', type=str, help='total labels')
parser.add_argument('--train_list', default='train.list', type=str, help='train list name')
parser.add_argument('--val_list', default='val.list', type=str, help='val list name')


if __name__ == "__main__":
    opt = parser.parse_args()
    
    train_lines = []
    val_lines = []
    with open(opt.total, 'r', encoding='utf-8') as f:
        labels = f.readlines()
    
    for label in labels:
        if 0 == random.randint(0, 19):
            val_lines.append(label)
        else:
            train_lines.append(label)

    with open(opt.train_list, 'w') as f:
        f.writelines(train_lines)
    
    with open(opt.val_list, 'w') as f:
        f.writelines(val_lines)
        
            
    
