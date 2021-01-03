# -*- coding: utf-8 -*-
import argparse
import os
import torch
from angle_class import ANGCLS


parser = argparse.ArgumentParser(description='test')
parser.add_argument('--resume_net', default='./checkpoints/angle_class_best.pth', type=str, help='net')
parser.add_argument('--onnx_net', default='./checkpoints/angle_class.onnx', type=str, help='onnx net')


if __name__ == "__main__":
    opt = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = ANGCLS(nclass=4, pretrained=True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        model = model.cuda()
    if opt.resume_net !='' and os.path.exists(opt.resume_net):
        print('loading pretrained model from %s' % opt.resume_net)
        model.load_state_dict(torch.load(opt.resume_net))
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    onnx_save_path = opt.onnx_net
    example_tensor = torch.randn(1, 3, 224, 224, device='cuda')
    _ = torch.onnx.export(model,                     # model being run
                          example_tensor,            # model input (or a tuple for multiple inputs)
                          onnx_save_path,
                          verbose=False,             # store the trained parameter weights inside the model file
                          training=False,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output']
                        )

