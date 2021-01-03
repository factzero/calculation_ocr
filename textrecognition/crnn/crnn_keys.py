import pickle as pkl


alphabet_list = pkl.load(open('./textrecognition/alphabet_v1.pkl','rb'))
alphabet = ''.join([chr(ord(ch)) for ch in alphabet_list])