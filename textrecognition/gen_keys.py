# -*- coding: utf-8 -*-
import os
import argparse
import pickle as pkl
from tqdm import tqdm
from alphabets_v1 import alphabet_v1


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--labels', default='', type=str, help='list(image_name characters)')
parser.add_argument('--alphabet', default='alphabet_v1.pkl', type=str, help='primeval alphabet')


if __name__ == "__main__":
    opt = parser.parse_args()

    if not os.path.exists(opt.alphabet):
        alphabet = [ch for ch in alphabet_v1]
    else:
        alphabet = pkl.load(open(opt.alphabet,'rb'))
    print('ori alphabet length : ', len(alphabet))
    if opt.labels != '':
        with open(opt.labels, 'r', encoding='utf-8') as f:
            labels = [ {c.split(' ')[0]:c.split(' ')[-1][:-1]}for c in f.readlines()]
            num_total = len(labels)
            for i in tqdm(range(num_total)):
            # for i in range(num_total):
                values = list(labels[i].values())[0]
                a = [v for v in values if v not in alphabet ]
                if len(a) != 0:
                    alphabet = alphabet + a

    print('done alphabet length : ', len(alphabet))
    with open('alphabet_v1.pkl', 'wb') as f:
        pkl.dump(alphabet, f)