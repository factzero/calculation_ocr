
# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
from textdetection.ctpn.ctpn_dataset import vocDataset


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


if __name__ == '__main__':
    train_dataset = vocDataset("D:/04download/VOC2007")
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False)
	
    cv2.namedWindow('input_image', 0)
    for i_batch, (image, gtboxes, labels, index) in enumerate(train_dataloader):
        cv2.resizeWindow('input_image', 1000, 800)
        image = image.squeeze(0).numpy().transpose([1, 2, 0]).astype(np.uint8)
        gtboxes = gtboxes.squeeze(0).numpy()
        # for gtbox, label in zip(gtboxes, labels):
        #     cv2.line(image, (int(gtbox[0]), int(gtbox[1])), (int(gtbox[2]), int(gtbox[3])), (255, 0, 0), 8)
        #     cv2.line(image, (int(gtbox[2]), int(gtbox[3])), (int(gtbox[4]), int(gtbox[5])), (255, 0, 0), 8)
        #     cv2.line(image, (int(gtbox[4]), int(gtbox[5])), (int(gtbox[6]), int(gtbox[7])), (255, 0, 0), 8)
        #     cv2.line(image, (int(gtbox[6]), int(gtbox[7])), (int(gtbox[0]), int(gtbox[1])), (255, 0, 0), 8)
            # image = cv2ImgAddText(image, str(label[0]), gtbox[2], gtbox[3], (255, 0, 0), 80)
        # for gtbox, gtcls in zip(gtboxes, gtclss):
        image = np.ascontiguousarray(image)
        for gtbox in gtboxes:
            cv2.line(image, (int(gtbox[0]), int(gtbox[1])), (int(gtbox[2]), int(gtbox[1])), (0, 0, 255), 4)
            cv2.line(image, (int(gtbox[2]), int(gtbox[1])), (int(gtbox[2]), int(gtbox[3])), (0, 0, 255), 4)
            cv2.line(image, (int(gtbox[2]), int(gtbox[3])), (int(gtbox[0]), int(gtbox[3])), (0, 0, 255), 4)
            cv2.line(image, (int(gtbox[0]), int(gtbox[3])), (int(gtbox[0]), int(gtbox[1])), (0, 0, 255), 4)
        cv2.imshow("input_image", image)
        cv2.waitKey(0)
        print(image.shape)
        
 
