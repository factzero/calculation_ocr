# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse
from ocr_det_rec import OcrDetRec
from PIL import Image, ImageDraw, ImageFont


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--image_name', default='./data/tx2.jpg', type=str, help='image for processing')
parser.add_argument('--det_model', default='./checkpoints/CTPN.pth', type=str, help='det model')
parser.add_argument('--rec_model', default='./checkpoints/CRNN.pth', type=str, help='rec model')
  
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)): # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype("font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


if __name__ == "__main__":
    opt = parser.parse_args()

    ocr = OcrDetRec(opt.det_model, opt.rec_model)
    image_name = opt.image_name
    image = cv2.imread(image_name)
    results = ocr.processing(image)
    for (k, v) in results.items():
        print(k, v[1])
        res = [int(j) for j in v[0]]
        # cv2.putText(image_framed, str(v[1]), (i[0], i[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.line(image, (res[0], res[1]), (res[2], res[3]), (0, 0, 255), 2)
        cv2.line(image, (res[0], res[1]), (res[4], res[5]), (0, 0, 255), 2)
        cv2.line(image, (res[6], res[7]), (res[2], res[3]), (0, 0, 255), 2)
        cv2.line(image, (res[4], res[5]), (res[6], res[7]), (0, 0, 255), 2)
        image = cv2ImgAddText(image, str(v[1]), res[0], res[1] - 10, (0, 0, 255), 20)
        
    # print(result)
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()