# -*- coding: utf-8 -*-
import cv2
import numpy as np
import argparse
from ocr_det_rec import OcrDetRec
from PIL import Image, ImageDraw, ImageFont


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--image_name', default='./data/tx2.jpg', type=str, help='image for processing')
 
  
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

    ocr = OcrDetRec('./checkpoints/CTPN.pth', './checkpoints/CRNN.pth')
    image_name = opt.image_name
    image = cv2.imread(image_name)
    results, image_framed = ocr.processing(image)
    for (k, v) in results.items():
        print(k, v[1])
        i = [int(j) for j in v[0]]
        # cv2.putText(image_framed, str(v[1]), (i[0], i[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        image_framed = cv2ImgAddText(image_framed, str(v[1]), i[0], i[1] - 10, (0, 0, 255), 20)
        
    # print(result)
    cv2.imshow('image_framed', image_framed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()