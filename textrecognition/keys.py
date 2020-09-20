import pickle as pkl

# alphabet_list = pkl.load(open('./textrecognition/alphabet.pkl','rb'))
# alphabet = [ord(ch) for ch in alphabet_list]
alphabet_list = pkl.load(open('./textrecognition/alphabet_v1.pkl','rb'))
alphabet = [ord(ch) for ch in alphabet_list]