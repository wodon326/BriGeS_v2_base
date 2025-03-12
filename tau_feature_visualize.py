import cv2
import torch
import os
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from AsymKD.dpt import AsymKD_DepthAnything_visualize
from depth_anything_for_evaluate.dpt import DepthAnything
from torchvision import transforms
from PIL import Image
import numpy as np

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


image_path = '/home/wodon326/data/AsymKD/diode_val/indoors/scene_00021/scan_00188/00021_00188_indoors_110_020.png'
img = Image.open(image_path)

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


depth_image_h, depth_image_w = depth_image.shape[-2:]
depth_patch_h, depth_patch_w = depth_image_h // 14, depth_image_w // 14
depth, depth_feature_out, seg_resize_out, Cross_Attention_Features = None, None, None, None
with torch.no_grad():
    depth, depth_feature_out, seg_resize_out, Cross_Attention_Features = model(depth_image.float(), seg_image.float())

depth_map = depth




save_dir = './tau_x_feature_visualize'
os.makedirs(save_dir, exist_ok=True)
# 시각화 및 저장
depth_map = depth_map.squeeze()
depth_map = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255.0
depth_map = depth_map.detach().cpu().numpy().astype(np.uint8)
depth_map = cv2.applyColorMap(depth_map, cv2.COLORMAP_INFERNO)
cv2.imwrite(os.path.join(save_dir, f'Ours_depth_map.jpg'), depth_map)


# 파일명 생성 및 저장
C = 768

softmax_tau1 = torch.load('./softmax_tau1_x.pt').permute(0, 2, 1).reshape((1, 768, depth_patch_h, depth_patch_w))
softmax_tau2 = torch.load('./softmax_tau2_x.pt').permute(0, 2, 1).reshape((1, 768, depth_patch_h, depth_patch_w))
softmax_tau_divide2 = torch.load('./softmax_tau_divide2_x.pt').permute(0, 2, 1).reshape((1, 768, depth_patch_h, depth_patch_w))

# 저장할 디렉토리 생성
sub_dir = 'tau2_feature_maps'
os.makedirs(f'{save_dir}/{sub_dir}', exist_ok=True)


# 모든 채널에 대해 feature map 저장
for channel in range(C):
    # 각 feature_map 추출
    tau1_map = softmax_tau1[0, channel, :, :].detach().cpu().numpy()
    tau2_map = softmax_tau2[0, channel, :, :].detach().cpu().numpy()
    tau_divide2_map = softmax_tau_divide2[0, channel, :, :].detach().cpu().numpy()
    
    # 세 가지 feature_map을 하나의 이미지로 결합
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # 각 subplot에 feature_map 시각화
    axs[0].imshow(tau1_map, cmap='viridis')
    axs[0].axis('off')
    axs[0].set_title('tau1_map')

    axs[1].imshow(tau2_map, cmap='viridis')
    axs[1].axis('off')
    axs[1].set_title('tau2_map')

    axs[2].imshow(tau_divide2_map, cmap='viridis')
    axs[2].axis('off')
    axs[2].set_title('tau_divide2_map')

    # 파일명 생성 및 저장
    save_path = os.path.join(save_dir, sub_dir, f'tau_feature_maps_{channel}.jpg')
    plt.savefig(save_path, format='jpg', bbox_inches='tight', pad_inches=0)
    plt.close()

quit()
# 저장할 디렉토리 생성
sub_dir = 'seg_feature_maps'
os.makedirs(f'{save_dir}/{sub_dir}', exist_ok=True)
# 모든 채널에 대해 feature map 저장
for index, features in enumerate(seg_resize_out):
    os.makedirs(f'{save_dir}/{sub_dir}/layer{index+1}', exist_ok=True)
    for channel in range(C):
        feature_map = features[0, channel, :, :].detach().cpu().numpy()
        
        # 시각화 및 저장
        plt.imshow(feature_map, cmap='viridis')
        plt.axis('off')  # 축 제거
        plt.gca().set_position([0, 0, 1, 1])  # 이미지만 남기기 위해 패딩 제거
        # 파일명 생성 및 저장
        save_path = os.path.join(save_dir, sub_dir,f'layer{index+1}', f'Seg_feature_map_channel_{channel}.jpg')
        plt.savefig(save_path, format='jpg')
        plt.close()
    break

# 저장할 디렉토리 생성
sub_dir = 'depth_feature_maps'
os.makedirs(f'{save_dir}/{sub_dir}', exist_ok=True)
# 모든 채널에 대해 feature map 저장
for index, features in enumerate(depth_feature_out):
    os.makedirs(f'{save_dir}/{sub_dir}/layer{index+1}', exist_ok=True)
    for channel in range(C):
        feature_map = features[0, channel, :, :].detach().cpu().numpy()
        
        # 시각화 및 저장
        plt.imshow(feature_map, cmap='viridis')
        plt.axis('off')  # 축 제거
        plt.gca().set_position([0, 0, 1, 1])  # 이미지만 남기기 위해 패딩 제거
        
        # 파일명 생성 및 저장
        save_path = os.path.join(save_dir, sub_dir,f'layer{index+1}', f'Depth_feature_map_channel_{channel}.jpg')
        plt.savefig(save_path, format='jpg')
        plt.close()
    break

# 저장할 디렉토리 생성
sub_dir = 'fuison_feature_maps'
os.makedirs(f'{save_dir}/{sub_dir}', exist_ok=True)
# 모든 채널에 대해 feature map 저장
for index, features in enumerate(Cross_Attention_Features):
    os.makedirs(f'{save_dir}/{sub_dir}/layer{index+1}', exist_ok=True)
    for channel in range(C):
        feature_map = features[0, channel, :, :].detach().cpu().numpy()
        
        # 시각화 및 저장
        plt.imshow(feature_map, cmap='viridis')
        plt.axis('off')  # 축 제거
        plt.gca().set_position([0, 0, 1, 1])  # 이미지만 남기기 위해 패딩 제거
        
        # 파일명 생성 및 저장
        save_path = os.path.join(save_dir, sub_dir,f'layer{index+1}', f'Fusion_feature_map_channel_{channel}.jpg')
        plt.savefig(save_path, format='jpg')
        plt.close()
    break


print(f'Feature maps saved to {save_dir}/')
