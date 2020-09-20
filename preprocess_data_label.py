# -*- coding: utf-8 -*-
'''
360万中文训练集标签修改
'''

with open('./char_std_5990.txt', 'rb') as file:
	char_dict = {num : char.strip().decode('gbk','ignore') for num, char in enumerate(file.readlines())}

with open('D:/80dataset/ocr/DataSet/data_train.txt') as file:
	value_list = ['%s %s'%(segment_list.split(' ')[0], ''.join([char_dict[int(val)] for val in segment_list[:-1].split(' ')[1:]])) for segment_list in file.readlines()]

with open('D:/80dataset/ocr/DataSet/data_train.list', 'w', encoding='utf-8') as file:
	[ file.write(val+'\n') for val in value_list]