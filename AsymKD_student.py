from torchhub.facebookresearch_dinov2_main.vision_transformer import vit_tiny

import torch
import torch.nn as nn
from AsymKD.blocks import FeatureFusionBlock, _make_scratch
import torch.nn.functional as F

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


class AsymKD_Student_DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(AsymKD_Student_DPTHead, self).__init__()
        
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.nclass = nclass
        self.use_clstoken = use_clstoken
        
        '''input output channel 변경 필요'''
        #in_channels = 1024
        selected_feature_channels = 256 #
        selected_out_channels=[64, 128, 256, 256]
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=selected_feature_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in selected_out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=selected_out_channels[0],
                out_channels=selected_out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=selected_out_channels[1],
                out_channels=selected_out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=selected_out_channels[3],
                out_channels=selected_out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])


        features = 64
        #double_out_channels=[512, 1024, 2048, 2048]
        self.scratch = _make_scratch(
            selected_out_channels,
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

            
    def forward(self, intermediate_layers, patch_h, patch_w):
        
        '''decoder에 넣기전 Select layer conv 연산'''
        conv_selected_layer = []
        for i, x in enumerate(intermediate_layers):
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
        depth_out = F.interpolate(depth_out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        depth_out = self.scratch.output_conv2(depth_out)
        
        return depth_out

class AsymKD_Student_Encoder(nn.Module):
    def __init__(self, in_channels = 256, use_clstoken=True):
        super(AsymKD_Student_Encoder,self).__init__()
        self.AsymKD_student = vit_tiny()
        channel_num = 4
        self.use_clstoken = use_clstoken
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(channel_num):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
                
        
    def forward(self, x, patch_h, patch_w):
        student_feature = self.AsymKD_student.get_intermediate_layers(x, 4, return_class_token=True)
        out = []
        for i, x in enumerate(student_feature):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            
            out.append(x)

        return out

class AsymKD_Student(nn.Module):
    def __init__(self, encoder, decoder):
        super(AsymKD_Student,self).__init__()
        
        
        self.AsymKD_student_encoder = encoder

        self.AsymKD_student_dpthead = decoder
                
        
    def forward(self, x):
        depth_image_h, depth_image_w = x.shape[-2:]
        patch_h, patch_w = depth_image_h // 14, depth_image_w // 14

        out = self.AsymKD_student_encoder(x, patch_h, patch_w)

        depth = self.AsymKD_student_dpthead(out, patch_h, patch_w)
        depth = F.interpolate(depth, size=(patch_h*14, patch_w*14), mode="bilinear", align_corners=True)
        depth = F.relu(depth)

        return depth
