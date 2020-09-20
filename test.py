# -*- coding: utf-8 -*-
import pickle as pkl
import tools.utils as utils
# from alphabets_v1 import alphabet_v1


# with open('./textrecognition/alphabet_v1.pkl', 'wb') as file:
#     pkl.dump(list(alphabet_v1), file)

alphabet_list = pkl.load(open('./textrecognition/alphabet.pkl','rb'))
# alphabet = [ord(ch) for ch in alphabet_list]
# alphabet = ''.join([chr(uni) for uni in alphabet])
# converter = utils.strLabelConverter(alphabet)
print(len(alphabet_list))
# print(alphabet_list)

alphabet_v1_list = pkl.load(open('./textrecognition/alphabet_v1.pkl','rb'))
# alphabet_v1 = [ord(ch) for ch in alphabet_v1_list]
# alphabet_v1 = ''.join([chr(uni) for uni in alphabet_v1])
print(len(alphabet_v1_list))

cnt_num = 0
for val in alphabet_v1_list:
    print(val, ord(val))
    if val not in alphabet_list:
        cnt_num += 1
        print(val, ord(val))
print(cnt_num)

# with open('./char_std_5990.txt', 'rb') as file:
#     char_dict = {num : char.strip().decode('gbk','ignore') for num, char in enumerate(file.readlines())}
# for (k, v) in char_dict.items():
#     text, length = converter.encode(v)
#     print(k, v, text, length)