# -*- coding: utf-8 -*-
import numpy as np
from textdetection import params


def gen_anchor(featuresize, scale):
    """
        gen base anchor from feature map [HXW][10][4]
        reshape  [HXW][10][4] to [HXWX10][4]
    """
    heights = [11, 16, 23, 33, 48, 68, 97, 139, 198, 283]
    widths = [16, 16, 16, 16, 16, 16, 16, 16, 16, 16]

    # gen k=10 anchor size (h,w)
    heights = np.array(heights).reshape(len(heights), 1)
    widths = np.array(widths).reshape(len(widths), 1)

    base_anchor = np.array([0, 0, 15, 15])
    # center x,y
    xt = (base_anchor[0] + base_anchor[2]) * 0.5
    yt = (base_anchor[1] + base_anchor[3]) * 0.5

    # x1 y1 x2 y2
    x1 = xt - widths * 0.5
    y1 = yt - heights * 0.5
    x2 = xt + widths * 0.5
    y2 = yt + heights * 0.5
    base_anchor = np.hstack((x1, y1, x2, y2))

    h, w = featuresize
    shift_x = np.arange(0, w) * scale
    shift_y = np.arange(0, h) * scale
    # apply shift
    anchor = []
    for i in shift_y:
        for j in shift_x:
            anchor.append(base_anchor + [j, i, j, i])
    return np.array(anchor).reshape((-1, 4))


def cal_iou(box1, box1_area , boxes2, boxes2_area):
    """
    box1 [x1,y1,x2,y2]
    boxes2 [Msample,x1,y1,x2,y2]
    """
    x1 = np.maximum(box1[0], boxes2[:, 0])
    x2 = np.minimum(box1[2], boxes2[:, 2])
    y1 = np.maximum(box1[1], boxes2[:, 1])
    y2 = np.minimum(box1[3], boxes2[:, 3])

    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    iou = intersection / (box1_area + boxes2_area[:] - intersection[:])
    return iou


def cal_overlaps(boxes1, boxes2):
    """
    boxes1 [Nsample,x1,y1,x2,y2]  anchor
    boxes2 [Msample,x1,y1,x2,y2]  grouth-box

    """
    area1 = (boxes1[:, 0] - boxes1[:, 2]) * (boxes1[:, 1] - boxes1[:, 3])
    area2 = (boxes2[:, 0] - boxes2[:, 2]) * (boxes2[:, 1] - boxes2[:, 3])

    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))

    # calculate the intersection of  boxes1(anchor) and boxes2(GT box)
    for i in range(boxes1.shape[0]):
        overlaps[i][:] = cal_iou(boxes1[i], area1[i], boxes2, area2)

    return overlaps
    

def bbox_transfrom(anchors, gtboxes):
    """
     compute relative predicted vertical coordinates Vc ,Vh
        with respect to the bounding box location of an anchor
    """
    regr = np.zeros((anchors.shape[0], 2))
    Cy = (gtboxes[:, 1] + gtboxes[:, 3]) * 0.5
    Cya = (anchors[:, 1] + anchors[:, 3]) * 0.5
    h = gtboxes[:, 3] - gtboxes[:, 1] + 1.0
    ha = anchors[:, 3] - anchors[:, 1] + 1.0

    Vc = (Cy - Cya) / ha
    Vh = np.log(h / ha)

    return np.vstack((Vc, Vh)).transpose()


def cal_rpn(imgsize, featuresize, scale, gtboxes):
    imgh, imgw = imgsize

    # gen base anchor
    base_anchor = gen_anchor(featuresize, scale)

    # calculate iou
    overlaps = cal_overlaps(base_anchor, gtboxes)

    # init labels -1 don't care  0 is negative  1 is positive
    labels = np.empty(base_anchor.shape[0])
    labels.fill(-1)

    # for each GT box corresponds to an anchor which has highest IOU
    gt_argmax_overlaps = overlaps.argmax(axis=0)

    # the anchor with the highest IOU overlap with a GT box
    anchor_argmax_overlaps = overlaps.argmax(axis=1)
    anchor_max_overlaps = overlaps[range(overlaps.shape[0]), anchor_argmax_overlaps]

    # IOU > IOU_POSITIVE
    labels[anchor_max_overlaps > params.IOU_POSITIVE] = 1
    # IOU <IOU_NEGATIVE
    labels[anchor_max_overlaps < params.IOU_NEGATIVE] = 0
    # ensure that every GT box has at least one positive RPN region
    labels[gt_argmax_overlaps] = 1

    # only keep anchors inside the image
    outside_anchor = np.where(
        (base_anchor[:, 0] < 0) |
        (base_anchor[:, 1] < 0) |
        (base_anchor[:, 2] >= imgw) |
        (base_anchor[:, 3] >= imgh)
    )[0]
    labels[outside_anchor] = -1

    # subsample positive labels ,if greater than RPN_POSITIVE_NUM(default 128)
    fg_index = np.where(labels == 1)[0]
    if (len(fg_index) > params.RPN_POSITIVE_NUM):
        labels[np.random.choice(fg_index, len(fg_index) - params.RPN_POSITIVE_NUM, replace=False)] = -1

    # calculate bbox targets
    bbox_targets = bbox_transfrom(base_anchor, gtboxes[anchor_argmax_overlaps, :])

    return [labels, bbox_targets], base_anchor


def linear_fit_y(xs, ys, x_list):
    """
    线性函数拟合两点(x1,y1),(x2,y2)；并求得x_list在的取值
    :param xs:  [x1,x2]
    :param ys:  [y1,y2]
    :param x_list: x轴坐标点,numpy数组 [n]
    :return:
    """
    if xs[0] == xs[1]:  # 垂直线
        return np.ones_like(x_list) * np.mean(ys)
    elif ys[0] == ys[1]:  # 水平线
        return np.ones_like(x_list) * ys[0]
    else:
        fn = np.poly1d(np.polyfit(xs, ys, 1))  # 一元线性函数
        return fn(x_list)


def get_min_max_y(quadrilateral, xs):
    """
    获取指定x值坐标点集合四边形上的y轴最小值和最大值
    :param quadrilateral: 四边形坐标；x1,y1,x2,y2,x3,y3,x4,y4
    :param xs: x轴坐标点,numpy数组 [n]
    :return:  x轴坐标点在四边形上的最小值和最大值
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = quadrilateral.tolist()
    y_val_1 = linear_fit_y(np.array([x1, x2]), np.array([y1, y2]), xs)
    y_val_2 = linear_fit_y(np.array([x2, x3]), np.array([y2, y3]), xs)
    y_val_3 = linear_fit_y(np.array([x3, x4]), np.array([y3, y4]), xs)
    y_val_4 = linear_fit_y(np.array([x4, x1]), np.array([y4, y1]), xs)
    y_val_min = []
    y_val_max = []
    for i in range(len(xs)):
        y_val = []
        if min(x1, x2) <= xs[i] <= max(x1, x2):
            y_val.append(y_val_1[i])
        if min(x2, x3) <= xs[i] <= max(x2, x3):
            y_val.append(y_val_2[i])
        if min(x3, x4) <= xs[i] <= max(x3, x4):
            y_val.append(y_val_3[i])
        if min(x4, x1) <= xs[i] <= max(x4, x1):
            y_val.append(y_val_4[i])
        # print("y_val:{}".format(y_val))
        y_val_min.append(min(y_val))
        y_val_max.append(max(y_val))

    return np.array(y_val_min), np.array(y_val_max)


def get_xs_in_range(x_array, x_min, x_max):
    """
    获取分割坐标点
    :param x_array: 宽度方向分割坐标点数组；0~image_width,间隔16 ；如:[0,16,32,...608]
    :param x_min: 四边形x最小值
    :param x_max: 四边形x最大值
    :return:
    """
    indices = np.logical_and(x_array >= x_min, x_array <= x_max)
    xs = x_array[indices]
    # 处理两端的值
    if xs.shape[0] == 0 or xs[0] > x_min:
        xs = np.insert(xs, 0, x_min)
    if xs.shape[0] == 0 or xs[-1] < x_max:
        xs = np.append(xs, x_max)
    return xs


def gen_gt_from_quadrilaterals(gt_quadrilaterals, input_gt_class_ids, image_shape, width_stride, box_min_size=3):
    """
    从gt 四边形生成，宽度固定的gt boxes
    :param gt_quadrilaterals: GT四边形坐标,[n,(x1,y1,x2,y2,x3,y3,x4,y4)]
    :param input_gt_class_ids: GT四边形类别，一般就是1 [n]
    :param image_shape:
    :param width_stride: 分割的步长，一般16
    :param box_min_size: 分割后GT boxes的最小尺寸
    :return:
            gt_boxes：[m,(x1,y1,x2,y2)]
            gt_class_ids: [m]
    """
    h, w = list(image_shape)[:2]
    x_array = np.arange(0, w + 1, width_stride, np.float32)  # 固定宽度间隔的x坐标点
    # 每个四边形x 最小值和最大值
    x_min_np = np.min(gt_quadrilaterals[:, ::2], axis=1)
    x_max_np = np.max(gt_quadrilaterals[:, ::2], axis=1)
    gt_boxes = []
    gt_class_ids = []
    for i in np.arange(len(gt_quadrilaterals)):
        xs = get_xs_in_range(x_array, x_min_np[i], x_max_np[i])  # 获取四边形内的x中坐标点
        ys_min, ys_max = get_min_max_y(gt_quadrilaterals[i], xs)
        # 为每个四边形生成固定宽度的gt
        for j in range(len(xs) - 1):
            x1, x2 = xs[j], xs[j + 1]
            y1, y2 = np.min(ys_min[j:j + 2]), np.max(ys_max[j:j + 2])
            gt_boxes.append([x1, y1, x2, y2])
            gt_class_ids.append(input_gt_class_ids[i])
    gt_boxes = np.reshape(np.array(gt_boxes), (-1, 4))
    gt_class_ids = np.reshape(np.array(gt_class_ids), (-1,))
    # 过滤高度太小的边框
    height = gt_boxes[:, 3] - gt_boxes[:, 1]
    width = gt_boxes[:, 2] - gt_boxes[:, 0]
    indices = np.where(np.logical_and(height >= 8, width >= 2))
    return gt_boxes[indices], gt_class_ids[indices]