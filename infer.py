import cv2
import torch
import os
import matplotlib
from segment_anything import sam_model_registry, SamPredictor
from AsymKD.dpt import AsymKD_DepthAnything_visualize
from depth_anything_for_evaluate.dpt import DepthAnything
from torchvision import transforms
from PIL import Image
import numpy as np
import os.path as osp
from glob import glob


def resize_to_nearest_divisible(img, divisor=14):
    # Get the original size of the image
    original_width, original_height = img.size

    # Calculate the new size that is divisible by the divisor
    new_width = (original_width // divisor) * divisor
    new_height = (original_height // divisor) * divisor

    # Apply the resize transform
    resize_transform = transforms.Resize((new_height, new_width))
    resized_img = resize_transform(img)

    return resized_img


def resize_img_map(img, target_width):
    # feature_map의 원본 크기
    original_width, original_height = img.size

    # 비율을 유지하면서 target_width에 맞는 새 높이 계산
    aspect_ratio = original_height / original_width
    target_height = int(target_width * aspect_ratio)
    resize_transform = transforms.Resize((target_height, target_width))
    resized_img = resize_transform(img)
    # 이미지 리사이즈
    return resized_img


device = 'cuda' if torch.cuda.is_available() else 'cpu'
segment_anything = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth").to(device)
segment_anything_predictor = SamPredictor(segment_anything)

for child in segment_anything.children():
    ImageEncoderViT = child
    break
model = AsymKD_DepthAnything_visualize(ImageEncoderViT=ImageEncoderViT).to(device)
restore_ckpt = "checkpoints_new_loss_001_smooth/46500_AsymKD_new_loss.pth"

encoder = "vitl"
depthanything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).to(device)


if restore_ckpt is not None:
    checkpoint = torch.load(restore_ckpt, map_location=device)
    model_state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in checkpoint.items():
        new_key = k.replace('module.', '')
        if new_key in model_state_dict:
            new_state_dict[new_key] = v

    model_state_dict.update(new_state_dict)
    model.load_state_dict(model_state_dict)

root = '/home/wodon326/data/AsymKD/diode_val'
image_path_arr = sorted( glob(osp.join(root,'*/*/*/*.png')) )
save_dir = './inference'
cmap = matplotlib.colormaps.get_cmap('Spectral_r')
os.makedirs(save_dir, exist_ok=True)
for image_path in image_path_arr:
    img = Image.open(image_path)
    file_name = image_path.split('/')[-1].split('.')[0]
    print(image_path)
    print(file_name)
    # img = resize_img_map(img, 1036)

    resized_img = resize_to_nearest_divisible(img)
    resized_img = np.array(resized_img)
    depth_image = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    depth_image = depth_image / 255.0 * 2.0 - 1.0
    depth_image = np.transpose(depth_image, (2, 0, 1))
    depth_image = torch.from_numpy(depth_image).unsqueeze(0).to(device)

    img = np.array(img)
    seg_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    seg_image = segment_anything_predictor.set_image(seg_image)
    seg_image = seg_image.to(device)


    depth, depth_feature_out, seg_resize_out, Cross_Attention_Features = None, None, None, None
    with torch.no_grad():
        depth, depth_feature_out, seg_resize_out, Cross_Attention_Features = model(depth_image.float(), seg_image.float())
        depth_depthanything = depthanything(depth_image.float())

    depth_map = depth



    # 시각화 및 저장
    depth_map = depth_map.squeeze()
    depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
    depth_map = depth_map.detach().cpu().numpy().astype(np.uint8)
    depth_map = (cmap(depth_map)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f'{file_name}_Ours.jpg'), depth_map)

    depthanything_depth_map = depth_depthanything
        
    
    # 시각화 및 저장
    depthanything_depth_map = depthanything_depth_map.squeeze()
    depthanything_depth_map = (depthanything_depth_map - depthanything_depth_map.min()) / (depthanything_depth_map.max() - depthanything_depth_map.min()) * 255.0
    depthanything_depth_map = depthanything_depth_map.detach().cpu().numpy().astype(np.uint8)
    depthanything_depth_map = (cmap(depthanything_depth_map)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f'{file_name}_Any.jpg'), depthanything_depth_map)
