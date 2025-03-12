import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from segment_anything.utils.onnx import SamOnnxModel

from depth_anything.dpt import DepthAnything
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
from AsymKD.dpt import AsymKD_DepthAnything
import time


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    return trainable_params, non_trainable_params


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#checkpoint = "sam_vit_b_01ec64.pth"
checkpoint = "sam_vit_l_0b3195.pth"
model_type = "vit_l"
segment_anything = sam_model_registry[model_type](checkpoint=checkpoint).to(DEVICE).eval()
segment_anything_predictor = SamPredictor(segment_anything)
'''AsymKD model load'''
for child in segment_anything.children():
        ImageEncoderViT = child
        break
model = AsymKD_DepthAnything(ImageEncoderViT = ImageEncoderViT)
model.to(DEVICE)
model.eval()


transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
    ])

for layer in model.children():
     print(layer)



M = 1000000

# 모델의 학습 가능 및 학습되지 않는 파라미터 수 출력
trainable_params, non_trainable_params = count_parameters(model)
print(f'Trainable parameters: {trainable_params/M}')
print(f'Non-trainable parameters: {non_trainable_params/M}')
print(f'Total parameters: {(non_trainable_params+trainable_params)/M}')
# quit()
img1 = torch.rand(518, 518, 3).numpy()
img2 = torch.rand(518, 1722, 3).numpy()
arr = [img1]



for image in arr:
    #image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
    
    h, w = image.shape[:2]
    
    x_seg = torch.rand(4, 3, 1024, 1024).to(DEVICE)

    x_depth = torch.rand(4, 3, 518, 518).to(DEVICE)
    
    total_time = 0
    for i in range(5):
        with torch.no_grad():
            start = time.time()
            depth, _ = model(x_depth, x_seg)
            end = time.time()
            infer_time = end - start
            total_time += infer_time
    print(f'infer time {(total_time /5):.2f}s')
    print(depth.shape)
    print(depth)

    