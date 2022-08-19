import os
import torch

weight_path = "/home/yubin/EasyNLP-alibaba/pai-painter-base-zh/pytorch_model.bin"
sd = torch.load(weight_path, map_location="cpu")


weight_path1 = "/home/yubin/EasyNLP-alibaba/tmp/finetune_model_MUGE/pytorch_model.bin"
sd1 = torch.load(weight_path1, map_location="cpu")

print('finish!')