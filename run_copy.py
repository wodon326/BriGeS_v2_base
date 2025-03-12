import argparse
import cv2
import numpy as np
import os
import torch
import torch.nn.functional as F
from torchvision.transforms import Compose
from tqdm import tqdm
import matplotlib.pyplot as plt

from AsymKD.dpt import AsymKD_DepthAnything

if __name__ == '__main__':
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    AsymKD = AsymKD_DepthAnything().to(DEVICE).eval()


    img_path = "assets/examples copy"

    if os.path.isfile(img_path):
        if img_path.endswith('txt'):
            with open(img_path, 'r') as f:
                filenames = f.read().splitlines()
        else:
            filenames = [img_path]
    else:
        filenames = os.listdir(img_path)
        filenames = [os.path.join(img_path, filename) for filename in filenames if not filename.startswith('.')]
        filenames.sort()


    for filename in tqdm(filenames):
        raw_image = cv2.imread(filename)
        
        
        with torch.no_grad():
            depth = AsymKD(raw_image)
    print(depth)
    print(depth.shape)
