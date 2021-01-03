# -*- coding: utf-8 -*-
import os
import argparse
import onnxruntime as rt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--onnx_net', default='./checkpoints/angle_class.onnx', type=str, help='onnx net')
parser.add_argument('--image_root', default='./data/', type=str, help='test image root dir')
parser.add_argument('--val_list', default='', type=str, help='net')


if __name__ == "__main__":
    PIXEL_MEANS =(0.485, 0.456, 0.406)  #RGB format mean and variances
    PIXEL_STDS = (0.229, 0.224, 0.225)

    opt = parser.parse_args()

    sess = rt.InferenceSession(opt.onnx_net)
    labels_index = {0:'0', 1:'90', 2:'180', 3:'270'}

    labels = []
    with open(opt.val_list, 'r', encoding='utf-8') as f:
        labels = [ {c.split(' ')[0]:c.split(' ')[-1][:-1]}for c in f.readlines()]
    num_total = len(labels)
    num_correct = 0
    for i in tqdm(range(num_total)):
        image_name = list(labels[i].keys())[0]
        label = list(labels[i].values())[0]
        image_name = os.path.join(opt.image_root, image_name)
        image_ori = Image.open(image_name).convert("RGB").resize((224, 224))
        image = np.array(image_ori).astype(np.float32)
        image /= 255.0
        image -= np.array(PIXEL_MEANS)
        image /= np.array(PIXEL_STDS)
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        image = np.expand_dims(image, axis=0)
        input_name = sess.get_inputs()[0].name
        label_name = sess.get_outputs()[0].name
        out = sess.run([label_name], {input_name:image})[0]
        pred_label = np.argmax(out,axis=1)[0]

        angle_label = labels_index[pred_label]

        if angle_label == label:
            num_correct += 1

    accuracy = num_correct / num_total
    print('accuracy: %0.4f, %d/%d' % (accuracy, num_correct, num_total))

    # angle_images = glob.glob(os.path.join(opt.image_root, '*.jpg'))
    # for image_name in angle_images:
    #     image_ori = Image.open(image_name).convert("RGB").resize((224, 224))
    #     image = np.array(image_ori).astype(np.float32)
    #     image /= 255.0
    #     image -= np.array(PIXEL_MEANS)
    #     image /= np.array(PIXEL_STDS)
    #     # HWC to CHW format:
    #     image = np.transpose(image, [2, 0, 1])
    #     image = np.expand_dims(image, axis=0)
    #     input_name = sess.get_inputs()[0].name
    #     label_name = sess.get_outputs()[0].name
    #     out = sess.run([label_name], {input_name:image})[0]
    #     pred_label = np.argmax(out,axis=1)[0]

    #     angle_label = labels_index[pred_label]

    #     image_show = image_ori
    #     font = ImageFont.truetype("SansSerif.ttf", 40) 
    #     draw = ImageDraw.Draw(image_show)
    #     draw.text((10, 10), angle_label, fill=(255, 0, 0), font=font)
    #     plt.imshow(image_show)
    #     plt.show()
