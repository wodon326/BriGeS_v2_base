import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from torchhub.facebookresearch_dinov2_main.dinov2.layers.block import CrossAttentionBlock,Block, CrossAttentionBlock_tau,Block_tau
from torchhub.facebookresearch_dinov2_main.dinov2.layers.mlp import Mlp
from AsymKD.blocks import FeatureFusionBlock, _make_scratch
from depth_anything.dpt import DepthAnything
from AsymKD.util.transform import Resize, NormalizeImage, PrepareForNet
from segment_anything import sam_model_registry, SamPredictor
from torchvision.transforms import Compose
import cv2
import numpy as np

def _make_fusion_block(features, use_bn, size = None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class AsymKD_DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=128, use_bn=False, out_channels=[96, 192, 384, 768], use_clstoken=False):
        super(AsymKD_DPTHead, self).__init__()
        
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
                
        depth = 1
        drop_path_rate=0.0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.Cross_Attention_blocks_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=in_channels,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=dpr[0],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
            ) for _ in range(4)])
        
        self.Blocks_layers = nn.ModuleList([
            Block(
                dim=in_channels,
                num_heads=16,
                mlp_ratio=4,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=dpr[0],
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
            ) for _ in range(4)])

        
        
        self.scratch = _make_scratch(
            out_channels,
            features,
            groups=1,
            expand=False,
        )

        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        head_features_1 = features
        head_features_2 = 32
        
        if nclass > 1:
            self.scratch.output_conv = nn.Sequential(
                nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_1, nclass, kernel_size=1, stride=1, padding=0),
            )
        else:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
            
            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
                nn.ReLU(True),
                nn.Identity(),
            )

            
    def forward(self, depth_intermediate_features, depth_patch_h, depth_patch_w, seg_intermediate_features, seg_patch_h, seg_patch_w ):
        depth_out = []
        for i, x in enumerate(depth_intermediate_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
        
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], depth_patch_h, depth_patch_w))
            depth_out.append(x)

        seg_feature = seg_intermediate_features
        seg_feature = F.interpolate(seg_feature, size=(depth_out[3].shape[2]*2, depth_out[3].shape[3]*2), mode='bilinear', align_corners=False)
        seg_feature = F.max_pool2d(seg_feature, kernel_size=2)
        batch_size, channels, height, width = seg_feature.shape
        reshaped_seg_feature = seg_feature.reshape(batch_size, channels, height * width).permute(0, 2, 1)


        '''Cross Attention 연산 코드'''
        Cross_Attention_Features = []
        return_Cross_Attention_Features = []
        for Cross_Attention_Blocks, Blocks,depth_feature in zip(self.Cross_Attention_blocks_layers, self.Blocks_layers, depth_out):
            batch_size, channels, height, width = depth_feature.shape
            reshaped_depth_feature = depth_feature.reshape(batch_size, channels, height * width).permute(0, 2, 1)
            feature = Cross_Attention_Blocks(reshaped_depth_feature, reshaped_seg_feature)
            feature = Blocks(feature)
            return_Cross_Attention_Features.append(feature)
            feature = feature.permute(0, 2, 1).reshape((feature.shape[0], feature.shape[-1], depth_patch_h, depth_patch_w))
            Cross_Attention_Features.append(feature)


        '''decoder에 넣기전 Select layer conv 연산'''
        conv_selected_layer = []
        for i, x in enumerate(Cross_Attention_Features):
            # print(f'seg size : {x.shape}')
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            conv_selected_layer.append(x)
        
  
        layer_1, layer_2, layer_3, layer_4 = conv_selected_layer
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        


        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        depth_out = self.scratch.output_conv1(path_1)
        depth_out = F.interpolate(depth_out, (int(depth_patch_h * 14), int(depth_patch_w * 14)), mode="bilinear", align_corners=True)
        depth_out = self.scratch.output_conv2(depth_out)
        
        return depth_out
            
    def forward_return_Cross_Attention_Features(self, depth_intermediate_features, depth_patch_h, depth_patch_w, seg_intermediate_features, seg_patch_h, seg_patch_w ):
        depth_out = []
        for i, x in enumerate(depth_intermediate_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x
        
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], depth_patch_h, depth_patch_w))
            depth_out.append(x)

        seg_feature = seg_intermediate_features
        seg_feature = F.interpolate(seg_feature, size=(depth_out[3].shape[2]*2, depth_out[3].shape[3]*2), mode='bilinear', align_corners=False)
        seg_feature = F.max_pool2d(seg_feature, kernel_size=2)
        batch_size, channels, height, width = seg_feature.shape
        reshaped_seg_feature = seg_feature.reshape(batch_size, channels, height * width).permute(0, 2, 1)


        '''Cross Attention 연산 코드'''
        Cross_Attention_Features = []
        return_Cross_Attention_Features = []
        for Cross_Attention_Blocks, Blocks,depth_feature in zip(self.Cross_Attention_blocks_layers, self.Blocks_layers, depth_out):
            batch_size, channels, height, width = depth_feature.shape
            reshaped_depth_feature = depth_feature.reshape(batch_size, channels, height * width).permute(0, 2, 1)
            feature = Cross_Attention_Blocks(reshaped_depth_feature, reshaped_seg_feature)
            feature = Blocks(feature)
            return_Cross_Attention_Features.append(feature)
            feature = feature.permute(0, 2, 1).reshape((feature.shape[0], feature.shape[-1], depth_patch_h, depth_patch_w))
            Cross_Attention_Features.append(feature)


        '''decoder에 넣기전 Select layer conv 연산'''
        conv_selected_layer = []
        for i, x in enumerate(Cross_Attention_Features):
            # print(f'seg size : {x.shape}')
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            conv_selected_layer.append(x)
        
  
        layer_1, layer_2, layer_3, layer_4 = conv_selected_layer
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        


        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        depth_out = self.scratch.output_conv1(path_1)
        depth_out = F.interpolate(depth_out, (int(depth_patch_h * 14), int(depth_patch_w * 14)), mode="bilinear", align_corners=True)
        depth_out = self.scratch.output_conv2(depth_out)
        
        return depth_out, return_Cross_Attention_Features

        
class AsymKD_DepthAnything(nn.Module):
    def __init__(self, ImageEncoderViT, features=128, out_channels=[96, 192, 384, 768], use_bn=False, use_clstoken=False, localhub=True):
        super(AsymKD_DepthAnything, self).__init__()
        
        
        encoder = 'vitb' # can also be 'vitb' or 'vitl'
        self.depth_anything = DepthAnything.from_pretrained('LiheYoung/depth_anything_{}14'.format(encoder)).eval()
        

        for param in self.depth_anything.parameters():
            param.requires_grad = False

        

        self.ImageEncoderViT = ImageEncoderViT

        for i, (name, param) in enumerate(self.ImageEncoderViT.named_parameters()):
            param.requires_grad = False

        dim = 768 #= self.pretrained.blocks[0].attn.qkv.in_features
        #dim = 1024

        self.depth_head = AsymKD_DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        
        for i, (name, param) in enumerate(self.depth_head.named_parameters()):
            if(name.split('.')[0]!='Cross_Attention_blocks_layers' and name.split('.')[0]!='Blocks_layers'):
                param.requires_grad = False
            # else:
            #     print(name)
        self.nomalize = NormalizeLayer()

    def forward(self, depth_image,seg_image):


        depth_image_h, depth_image_w = depth_image.shape[-2:]
        seg_image_h, seg_image_w = seg_image.shape[-2:]
        self.ImageEncoderViT.eval()
        self.depth_anything.eval()

        
        depth_intermediate_features = self.depth_anything(depth_image)
        
        seg_intermediate_features = self.ImageEncoderViT(seg_image)



        depth_patch_h, depth_patch_w = depth_image_h // 14, depth_image_w // 14
        seg_patch_h, seg_patch_w = seg_image_h // 16, seg_image_w // 16


        depth = self.depth_head(depth_intermediate_features, depth_patch_h, depth_patch_w, seg_intermediate_features, seg_patch_h, seg_patch_w )
        depth = F.interpolate(depth, size=(depth_patch_h*14, depth_patch_w*14), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        depth = self.nomalize(depth) if self.training else depth


        return depth
    
    def forward(self, depth_image,seg_image):


        depth_image_h, depth_image_w = depth_image.shape[-2:]
        seg_image_h, seg_image_w = seg_image.shape[-2:]
        self.ImageEncoderViT.eval()
        self.depth_anything.eval()

        
        depth_intermediate_features = self.depth_anything(depth_image)
        
        seg_intermediate_features = self.ImageEncoderViT(seg_image)



        depth_patch_h, depth_patch_w = depth_image_h // 14, depth_image_w // 14
        seg_patch_h, seg_patch_w = seg_image_h // 16, seg_image_w // 16


        depth = self.depth_head(depth_intermediate_features, depth_patch_h, depth_patch_w, seg_intermediate_features, seg_patch_h, seg_patch_w )
        depth = F.interpolate(depth, size=(depth_patch_h*14, depth_patch_w*14), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        depth = self.nomalize(depth) if self.training else depth


        return depth

        
    def forward_with_depth_features(self, depth_intermediate_features, depth_image, seg_image):


        depth_image_h, depth_image_w = depth_image.shape[-2:]
        seg_image_h, seg_image_w = seg_image.shape[-2:]
        self.ImageEncoderViT.eval()

        
        # depth_intermediate_features = self.depth_anything(depth_image)
        


        seg_intermediate_features = self.ImageEncoderViT(seg_image)



        depth_patch_h, depth_patch_w = depth_image_h // 14, depth_image_w // 14
        seg_patch_h, seg_patch_w = seg_image_h // 16, seg_image_w // 16


        depth, Cross_Attention_Features = self.depth_head.forward_return_Cross_Attention_Features(depth_intermediate_features, depth_patch_h, depth_patch_w, seg_intermediate_features, seg_patch_h, seg_patch_w )
        depth = F.interpolate(depth, size=(depth_patch_h*14, depth_patch_w*14), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        depth = self.nomalize(depth) if self.training else depth


        return depth, Cross_Attention_Features
        
    def forward_with_encoder_features(self, depth_image, seg_image):


        depth_image_h, depth_image_w = depth_image.shape[-2:]
        seg_image_h, seg_image_w = seg_image.shape[-2:]
        self.ImageEncoderViT.eval()

        
        depth_intermediate_features = self.depth_anything(depth_image)
        


        seg_intermediate_features = self.ImageEncoderViT(seg_image)



        return depth_intermediate_features, seg_intermediate_features

    
    def load_ckpt(
        self,
        ckpt: str,
        device: torch.device
    ):
        assert ckpt.endswith('.pth'), 'Please provide the path to the checkpoint file.'
        
        ckpt = torch.load(ckpt, map_location=device)
        # ckpt = ckpt['model_state_dict']
        model_state_dict = self.state_dict()
        new_state_dict = {}
        for k, v in ckpt.items():
            # 키 매핑 규칙을 정의
            new_key = k.replace('module.', '')  # 'module.'를 제거
            if new_key in model_state_dict:
                new_state_dict[new_key] = v

        model_state_dict.update(new_state_dict)
        self.load_state_dict(model_state_dict)
    
        return new_state_dict
    

class NormalizeLayer(nn.Module):
    def __init__(self):
        super(NormalizeLayer, self).__init__()
    
    def forward(self, x):
        min_val = x.amin(dim=(1, 2, 3), keepdim=True)  # 각 배치별 최소값
        max_val = x.amax(dim=(1, 2, 3), keepdim=True)  # 각 배치별 최대값
        x = (x - min_val) / (max_val - min_val + 1e-6)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--encoder",
        default="vits",
        type=str,
        choices=["vits", "vitb", "vitl"],
    )
    args = parser.parse_args()
    
    model = DepthAnything.from_pretrained("LiheYoung/depth_anything_{:}14".format(args.encoder))
    
    print(model)
    