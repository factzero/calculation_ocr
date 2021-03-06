# -*- coding: utf-8 -*-
import os, sys
import glob
import cv2
import numpy as np
import xml.dom.minidom
import argparse
import ctpn_params
from ctpn_utils import resize_image2square, adj_gtboxes


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--icdar_root', default='D:/80dataset/ocr/share/data_ready', type=str, help='source')
parser.add_argument('--voc_root', default='D:/80dataset/ocr', help='target dir')


def WriterXMLFiles(filename, img_name, box_list, labels, w, h, d):
    doc = xml.dom.minidom.Document()
    root = doc.createElement('annotation')
    doc.appendChild(root)

    foldername = doc.createElement("folder")
    foldername.appendChild(doc.createTextNode("JPEGImages"))
    root.appendChild(foldername)

    sourcename=doc.createElement("source")
    databasename = doc.createElement("database")
    databasename.appendChild(doc.createTextNode("icdar2017rctw_train_v1.2"))
    sourcename.appendChild(databasename)
    imagename = doc.createElement("image")
    imagename.appendChild(doc.createTextNode(img_name))
    sourcename.appendChild(imagename)
    root.appendChild(sourcename)

    nodesize = doc.createElement('size')
    nodewidth = doc.createElement('width')
    nodewidth.appendChild(doc.createTextNode(str(w)))
    nodesize.appendChild(nodewidth)
    nodeheight = doc.createElement('height')
    nodeheight.appendChild(doc.createTextNode(str(h)))
    nodesize.appendChild(nodeheight)
    nodedepth = doc.createElement('depth')
    nodedepth.appendChild(doc.createTextNode(str(d)))
    nodesize.appendChild(nodedepth)
    root.appendChild(nodesize)

    segname = doc.createElement("segmented")
    segname.appendChild(doc.createTextNode("0"))
    root.appendChild(segname)

    for box, label in zip(box_list, labels):
        nodeobject = doc.createElement('object')
        nodename = doc.createElement('name')
        nodename.appendChild(doc.createTextNode(label))
        nodeobject.appendChild(nodename)
        nodelabel = doc.createElement('label')
        nodelabel.appendChild(doc.createTextNode('1'))
        nodeobject.appendChild(nodelabel)
        nodebndbox = doc.createElement('bndbox')
        nodex1 = doc.createElement('x1')
        nodex1.appendChild(doc.createTextNode(str(box[0])))
        nodebndbox.appendChild(nodex1)
        nodey1 = doc.createElement('y1')
        nodey1.appendChild(doc.createTextNode(str(box[1])))
        nodebndbox.appendChild(nodey1)
        nodex2 = doc.createElement('x2')
        nodex2.appendChild(doc.createTextNode(str(box[2])))
        nodebndbox.appendChild(nodex2)
        nodey2 = doc.createElement('y2')
        nodey2.appendChild(doc.createTextNode(str(box[3])))
        nodebndbox.appendChild(nodey2)
        nodex3 = doc.createElement('x3')
        nodex3.appendChild(doc.createTextNode(str(box[4])))
        nodebndbox.appendChild(nodex3)
        nodey3 = doc.createElement('y3')
        nodey3.appendChild(doc.createTextNode(str(box[5])))
        nodebndbox.appendChild(nodey3)
        nodex4 = doc.createElement('x4')
        nodex4.appendChild(doc.createTextNode(str(box[6])))
        nodebndbox.appendChild(nodex4)
        nodey4 = doc.createElement('y4')
        nodey4.appendChild(doc.createTextNode(str(box[7])))
        nodebndbox.appendChild(nodey4)

        # ang = doc.createElement('angle')
        # ang.appendChild(doc.createTextNode(str(angle)))
        # nodebndbox.appendChild(ang)
        nodeobject.appendChild(nodebndbox)
        root.appendChild(nodeobject)

    with open(filename, 'w', encoding='utf-8') as fp:
        doc.writexml(fp, indent='\n')


def load_annoataion(p):
    '''
    load annotation from the text file
    '''
    text_polys = []
    text_tags = []
    if not os.path.exists(p):
        return np.array(text_polys, dtype=np.float32)
    with open(p, 'r', encoding='utf-8') as f:
        gt_txt = f.read()
        gt_split = gt_txt.split('\n')
        for gt_line in gt_split:
            gt_ind = gt_line.split(',')
            if len(gt_ind) < 8:
                break
            x1, y1, x2, y2, x3, y3, x4, y4 = [int(float(gt_ind[i])) for i in range(8)]
            text_polys.append([x1, y1, x2, y2, x3, y3, x4, y4])
            label = 1
            text_tags.append(label)

        return np.array(text_polys, dtype=np.int32), np.array(text_tags, dtype=np.str)


if __name__ == "__main__":
    opt = parser.parse_args()
    DATASET_LIST = ["ali_icpr", "MSRA_TD500", "icdar2015Text Localtion"]
    # target dir
    base_dir = opt.voc_root
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    target_img_dir = os.path.join(base_dir, "JPEGImages")
    if not os.path.exists(target_img_dir):
        os.mkdir(target_img_dir)
    target_ann_dir = os.path.join(base_dir, "Annotations")
    if not os.path.exists(target_ann_dir):
        os.mkdir(target_ann_dir)
    target_set_dir = os.path.join(base_dir, "ImageSets")
    if not os.path.exists(target_set_dir):
        os.mkdir(target_set_dir)
    
    for dataset in DATASET_LIST:
        # source
        train_img_dir = os.path.join(opt.icdar_root, dataset, "train_im")
        train_txt_dir = os.path.join(opt.icdar_root, dataset, "train_gt")

        if not os.path.exists(train_img_dir) or not os.path.exists(train_txt_dir):
            continue

        gt_list = []
        img_list = []
        # rename and move img to target_img_dir
        for file_name in os.listdir(train_img_dir):
            if file_name.split('.')[-1] == 'jpg':
                os.rename(os.path.join(train_img_dir, file_name), 
                        os.path.join(target_img_dir, dataset + os.path.basename(file_name)))
                img_list.append(dataset + os.path.basename(file_name))
                # img_list.append(file_name)
                gt_list.append("gt_" + file_name.replace('.jpg', '.txt'))

        for idx in range(len(img_list)):
            img_name = os.path.join(target_img_dir, img_list[idx])
            gt_name = os.path.join(train_txt_dir, gt_list[idx])
            print(img_name)
            print(gt_name)
            boxes, labels = load_annoataion(gt_name)
            img = cv2.imread(img_name)
            if img is None or len(boxes) == 0:
                continue
            img, rescale_fac, padding = resize_image2square(img, ctpn_params.IMAGE_HEIGHT)
            boxes = adj_gtboxes(boxes, rescale_fac, padding)
            h, w, d = img.shape
            cv2.imwrite(img_name, img)
            xml_file = os.path.join(target_ann_dir, (img_list[idx].split('.')[0] + '.xml'))
            print(xml_file)
            WriterXMLFiles(xml_file, img_list[idx], boxes, labels, w, h, d)

    # write info into target_set_dir
    img_lists = glob.glob(target_ann_dir + '/*.xml')
    img_names = []
    for item in img_lists:
        temp1, temp2 = os.path.splitext(os.path.basename(item))
        img_names.append(temp1)

    target_set_dir_main = os.path.join(target_set_dir, 'Main')
    if not os.path.exists(target_set_dir_main):
        os.mkdir(target_set_dir_main)
    with open(os.path.join(target_set_dir_main, 'trainval.txt'), 'w', encoding='utf-8') as f:
        for item in img_names:
            f.write(str(item) + '\n')