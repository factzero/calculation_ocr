# -*- coding: utf-8 -*-
import argparse
import os
import torch
import cv2
import numpy as np
import math
from config import Configurable, Config
from db_model import SegDetectorModel
from seg_detector_representer import SegDetectorRepresenter


def main():
    parser = argparse.ArgumentParser(description='Text Recognition Testing')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--image_path', type=str, help='image path')
    parser.add_argument('--result_dir', type=str, default='./demo_results/', help='path to save results')
    parser.add_argument('--data', type=str, help='The name of dataloader which will be evaluated on.')
    parser.add_argument('--image_short_side', type=int, default=736, help='The threshold to replace it in the representers')
    parser.add_argument('--thresh', type=float, help='The threshold to replace it in the representers')
    parser.add_argument('--box_thresh', type=float, default=0.6, help='The threshold to replace it in the representers')
    parser.add_argument('--visualize', action='store_true', help='visualize maps in tensorboard')
    parser.add_argument('--resize', action='store_true', help='resize')
    parser.add_argument('--polygon', action='store_true', help='output polygons if true')
    parser.add_argument('--eager', '--eager_show', action='store_true', dest='eager_show', help='Show iamges eagerly')

    args = parser.parse_args()
    args = vars(args)
    args = {k: v for k, v in args.items() if v is not None}

    Demo(args=args).inference(args['image_path'], args['visualize'])


class Demo:
    def __init__(self, args):
        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.args = args
        self.model_path = self.args['resume']
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model = SegDetectorModel(device=self.device)
        states = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(states, strict=False)
        model.eval()
        if torch.cuda.is_available():
            # torch.backends.cudnn.benchmark = True
            self.model = model.cuda()
        self.representer = SegDetectorRepresenter()
        

    def resize_image(self, img):
        height, width, _ = img.shape
        if height < width:
            new_height = self.args['image_short_side']
            new_width = int(math.ceil(new_height / height * width / 32) * 32)
        else:
            new_width = self.args['image_short_side']
            new_height = int(math.ceil(new_width / width * height / 32) * 32)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img
        
    def load_image(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR).astype('float32')
        original_shape = img.shape[:2]
        img = self.resize_image(img)
        img -= self.RGB_MEAN
        img /= 255.
        img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0)
        return img, original_shape
        
    def format_output(self, batch, output):
        batch_boxes, batch_scores = output
        for index in range(batch['image'].size(0)):
            original_shape = batch['shape'][index]
            filename = batch['filename'][index]
            result_file_name = 'res_' + filename.split('/')[-1].split('.')[0] + '.txt'
            result_file_path = os.path.join(self.args['result_dir'], result_file_name)
            boxes = batch_boxes[index]
            scores = batch_scores[index]
            if self.args['polygon']:
                with open(result_file_path, 'wt') as res:
                    for i, box in enumerate(boxes):
                        box = np.array(box).reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        score = scores[i]
                        res.write(result + ',' + str(score) + "\n")
            else:
                with open(result_file_path, 'wt') as res:
                    for i in range(boxes.shape[0]):
                        score = scores[i]
                        if score < self.args['box_thresh']:
                            continue
                        box = boxes[i,:,:].reshape(-1).tolist()
                        result = ",".join([str(int(x)) for x in box])
                        res.write(result + ',' + str(score) + "\n")
        
    def inference(self, image_path, visualize=False):
        all_matircs = {}
        batch = dict()
        batch['filename'] = [image_path]
        img, original_shape = self.load_image(image_path)
        batch['shape'] = [original_shape]
        with torch.no_grad():
            batch['image'] = img
            pred = self.model.forward(batch, training=False)
            output = self.representer.represent(batch, pred, is_output_polygon=self.args['polygon']) 
            if not os.path.isdir(self.args['result_dir']):
                os.mkdir(self.args['result_dir'])
            self.format_output(batch, output)

            if visualize and self.structure.visualizer:
                vis_image = self.structure.visualizer.demo_visualize(image_path, output)
                cv2.imwrite(os.path.join(self.args['result_dir'], image_path.split('/')[-1].split('.')[0]+'.jpg'), vis_image)


if __name__ == "__main__":
    main()