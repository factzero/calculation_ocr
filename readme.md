# 简介

+ 基于pytorch实现对文字检测和识别

# 一、文字检测

## 1、CTPN

### 1.1 原理

+ [ctpn论文](https://arxiv.org/abs/1609.03605)
+ [ctpn论文翻译](https://github.com/yizt/cv-papers/blob/master/CTPN.md)

### 1.2 关键点说明

a、骨干网络使用VGG

b、训练图片输入大小为720x720，等比例缩放，将图片长边缩放至720，短边补黑至720

c、每张图片训练128个anchor，正负样本比例1:1

d、

### 1.3 准备数据



### 1.4 训练



# 二、文字识别

+ CTC

 