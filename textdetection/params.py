# -*- coding: utf-8 -*-
import os

IMAGE_MEAN = [123.68, 116.779, 103.939]
IMAGE_HEIGHT = 720

ANCHORS_HEIGHT = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
ANCHORS_WIDTH = 16
IOU_NEGATIVE = 0.3
IOU_POSITIVE = 0.7
IOU_SELECT = 0.7
RPN_POSITIVE_NUM = 128

lr = 0.0001 # learning rate for Critic, not used by adadealta
niter = 300

