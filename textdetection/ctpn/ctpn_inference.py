# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from textdetection.ctpn.ctpn_model import CTPN_Model
from textdetection.ctpn.ctpn_utils import gen_anchor, bbox_transfor_inv, clip_box, filter_bbox,nms, TextProposalConnectorOriented
from textdetection.ctpn.ctpn_utils import resize
from textdetection.ctpn import ctpn_params

class OcrDetCTPN():
    def __init__(self, model_path='./checkpoints/CTPN.pth'):
        self.model = CTPN_Model()
        self.use_gpu = torch.cuda.is_available()
        if self.use_gpu:
            self.model.cuda()
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(model_path, map_location=device)['model_state_dict'])
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.prob_thresh = 0.5

    def inference(self, image):
        image_sz = resize(image, height=ctpn_params.IMAGE_HEIGHT)
        # 宽高缩放比例(等比例缩放)
        rescale_fac = image.shape[0] / image_sz.shape[0]
        h, w = image_sz.shape[:2]
        # 减均值
        image_sz = image_sz.astype(np.float32) - ctpn_params.IMAGE_MEAN
        image_sz = torch.from_numpy(image_sz.transpose(2, 0, 1)).unsqueeze(0).float()

        if self.use_gpu:
            image_sz = image_sz.cuda()
        cls, regr = self.model(image_sz)
        cls_prob = F.softmax(cls, dim=-1).cpu().numpy()
        regr = regr.cpu().numpy()
        anchor = gen_anchor((int(h / 16), int(w / 16)), 16)
        bbox = bbox_transfor_inv(anchor, regr)
        bbox = clip_box(bbox, [h, w])

        fg = np.where(cls_prob[0, :, 1] > self.prob_thresh)[0]
        select_anchor = bbox[fg, :]
        select_score = cls_prob[0, fg, 1]
        select_anchor = select_anchor.astype(np.int32)
        keep_index = filter_bbox(select_anchor, 16)

        # nms
        select_anchor = select_anchor[keep_index]
        select_score = select_score[keep_index]
        select_score = np.reshape(select_score, (select_score.shape[0], 1))
        nmsbox = np.hstack((select_anchor, select_score))
        keep = nms(nmsbox, 0.3)
        select_anchor = select_anchor[keep]
        select_score = select_score[keep]

        # text line-
        textConn = TextProposalConnectorOriented()
        text = textConn.get_text_lines(select_anchor, select_score, [h, w])
        text = [np.hstack((res[:8]*rescale_fac, res[8])) for res in text]

        return text


if __name__ == '__main__':
    image_name = 'data/t2.png'
    image = cv2.imread(image_name)
    model_path = './checkpoints/CTPN.pth'
    ctpn_det = OcrDetCTPN(model_path)
    text, image = ctpn_det.inference(image)

    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()