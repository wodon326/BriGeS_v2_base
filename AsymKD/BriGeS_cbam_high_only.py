import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from torchhub.facebookresearch_dinov2_main.dinov2.layers.block import CrossAttentionBlock,Block, CrossAttentionBlock_tau,Block_tau
from torchhub.facebookresearch_dinov2_main.dinov2.layers.mlp import Mlp
from AsymKD.blocks import FeatureFusionBlock, _make_scratch
from depth_anything.dpt import DepthAnything
from .cbam import SpatialAttentionExtractor, ChannelAttentionEnhancement
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

        self.Adapter_sams = SpatialAttentionExtractor()
        self.Adapter_cams = ChannelAttentionEnhancement(features)
        self.Adapter_conv =nn.Sequential(
            # nn.Conv2d(features*3, features, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1),
        )

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

    def unfreeze_BriGeS_cbam(self):
        for param in self.Adapter_sams.parameters():
            param.requires_grad = True
        for param in self.Adapter_cams.parameters():
            param.requires_grad = True
        for param in self.Adapter_conv.parameters():
            param.requires_grad = True
            
    def forward(self, depth_intermediate_features, depth_patch_h, depth_patch_w, seg_intermediate_features, seg_patch_h, seg_patch_w ):
        depth_out = []
        for i, x in enumerate(depth_intermediate_features):
        
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], depth_patch_h, depth_patch_w))
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            depth_out.append(x)

        layer_1, layer_2, layer_3, layer_4 = depth_out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        cbam_attn = self.Adapter_cams(path_1) * path_1
        cbam_attn = self.Adapter_sams(cbam_attn) * cbam_attn
        # attn으로 인해 disparity가 아닌 depth로 예측하게됨
        # -1 곱해서 disparity로 예측하게 수정
        # high_freq_feat = path_1 * cbam_attn
        # low_freq_feat =  path_1 * (1-cbam_attn) * -1
        # cbam_feat = torch.cat([high_freq_feat, low_freq_feat, path_1], dim=1)
        cbam_feat = cbam_attn + path_1
        path_1 = self.Adapter_conv(cbam_feat)
        
        depth_out = self.scratch.output_conv1(path_1)
        depth_out = F.interpolate(depth_out, (int(depth_patch_h * 14), int(depth_patch_w * 14)), mode="bilinear", align_corners=True)
        depth_out = self.scratch.output_conv2(depth_out)
        
        return depth_out

        
class BriGeS_cbam_high_only(nn.Module):
    def __init__(self, ImageEncoderViT, features=128, out_channels=[96, 192, 384, 768], use_bn=False, use_clstoken=False, localhub=True):
        super(BriGeS_cbam_high_only, self).__init__()
        
        print('BriGeS_cbam_high_only')
        
        encoder = 'vitb' # can also be 'vitb' or 'vitl'
        if localhub:
            self.pretrained = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
        

        self.ImageEncoderViT = ImageEncoderViT


        dim = 768 #= self.pretrained.blocks[0].attn.qkv.in_features
        #dim = 1024

        
        depth = 1
        drop_path_rate=0.0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.Cross_Attention_blocks_layers = nn.ModuleList([
            CrossAttentionBlock(
                dim=dim,
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
                dim=dim,
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


        self.depth_head = AsymKD_DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        
        self.nomalize = NormalizeLayer()

    def forward(self, depth_image,seg_image):


        depth_image_h, depth_image_w = depth_image.shape[-2:]
        seg_image_h, seg_image_w = seg_image.shape[-2:]

        depth_patch_h, depth_patch_w = depth_image_h // 14, depth_image_w // 14
        seg_patch_h, seg_patch_w = seg_image_h // 16, seg_image_w // 16
        self.ImageEncoderViT.eval()
        self.pretrained.eval()

        
        depth_intermediate_features = self.pretrained.get_intermediate_layers(depth_image, 4, return_class_token=False)
        
        seg_intermediate_features = self.ImageEncoderViT(seg_image)


        seg_feature = seg_intermediate_features
        seg_feature = F.interpolate(seg_feature, size=(depth_patch_h*2, depth_patch_w*2), mode='bilinear', align_corners=False)
        seg_feature = F.max_pool2d(seg_feature, kernel_size=2)
        batch_size, channels, height, width = seg_feature.shape
        reshaped_seg_feature = seg_feature.reshape(batch_size, channels, height * width).permute(0, 2, 1)


        '''Cross Attention 연산 코드'''
        Cross_Attention_Features = []
        for Cross_Attention_Blocks, Blocks,depth_feature in zip(self.Cross_Attention_blocks_layers, self.Blocks_layers, depth_intermediate_features):
            feature = Cross_Attention_Blocks(depth_feature, reshaped_seg_feature)
            feature = Blocks(feature)
            Cross_Attention_Features.append(feature)




        depth = self.depth_head(Cross_Attention_Features, depth_patch_h, depth_patch_w, seg_intermediate_features, seg_patch_h, seg_patch_w )
        depth = F.interpolate(depth, size=(depth_patch_h*14, depth_patch_w*14), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        depth = self.nomalize(depth) if self.training else depth


        return depth

    def freeze_BriGeS_cbam_style(self):
        for param in self.pretrained.parameters():
            param.requires_grad = False

        for i, (name, param) in enumerate(self.ImageEncoderViT.named_parameters()):
            param.requires_grad = False

        for param in self.depth_head.parameters():
            param.requires_grad = False

        self.depth_head.unfreeze_BriGeS_cbam()
    
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
    