import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from huggingface_hub import PyTorchModelHubMixin, hf_hub_download

from torchhub.facebookresearch_dinov2_main.dinov2.layers import NestedTensorBlock as Block
from torchhub.facebookresearch_dinov2_main.dinov2.layers.mlp import Mlp
from .cbam import SpatialAttentionExtractor, ChannelAttentionEnhancement


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output


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


class DPTHead(nn.Module):
    def __init__(self, nclass, in_channels, features=256, use_bn=False, out_channels=[256, 512, 1024, 1024], use_clstoken=False):
        super(DPTHead, self).__init__()
        
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
        
        
        self.Adapter_sams = nn.ModuleList([
            SpatialAttentionExtractor(),
            SpatialAttentionExtractor(),
            nn.Identity(),
            nn.Identity()
        ])
        

        self.Adapter_cams = nn.ModuleList([
            ChannelAttentionEnhancement(out_channels[0]),
            ChannelAttentionEnhancement(out_channels[1]),
            nn.Identity(),
            nn.Identity()
            ])
        
        self.Adapter_conv = nn.ModuleList([
            nn.Sequential(
            nn.Conv2d(out_channels[0]*3, out_channels[0]*3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channels[0]*3, out_channels[0]*3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels[0]*3, out_channels[0], kernel_size=1, stride=1, padding=0)
        ),
         nn.Sequential(
            nn.Conv2d(out_channels[1]*3, out_channels[1]*3, kernel_size=1, stride=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(out_channels[1]*3, out_channels[1]*3, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channels[1]*3, out_channels[1], kernel_size=1, stride=1, padding=0)
        ),
        nn.Identity(),
        nn.Identity()
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
            
    def forward(self, out_features, patch_h, patch_w):
        out = []
        for i, x in enumerate(out_features):
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        

        adapter_feature = []
        for feat, sam, cam, conv in zip(out, self.Adapter_sams,self.Adapter_cams, self.Adapter_conv):
            if not isinstance(conv, nn.Identity):
                cbam_attn = cam(feat) * feat
                cbam_attn = sam(cbam_attn)
                high_freq_feat = feat * cbam_attn
                context_feat = feat * (1-cbam_attn)
                feat = conv(torch.cat((high_freq_feat, feat, context_feat), dim=1))

            adapter_feature.append(feat)

        
        layer_1, layer_2, layer_3, layer_4 = adapter_feature

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        
        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        
        out = self.scratch.output_conv1(path_1)
        out = F.interpolate(out, (int(patch_h * 14), int(patch_w * 14)), mode="bilinear", align_corners=True)
        out = self.scratch.output_conv2(out)
        
        return out, adapter_feature
    
    def unfreeze_kd_style(self):
        for param in self.Adapter_sams.parameters():
            param.requires_grad = True
        for param in self.Adapter_cams.parameters():
            param.requires_grad = True
        for param in self.Adapter_conv.parameters():
            param.requires_grad = True
        
        
class kd_naive_depth_latent4_split_adapter_cbam(nn.Module):
    def __init__(self, encoder='vitb', features=128, out_channels=[96, 192, 384, 768], use_bn=False, use_clstoken=False, localhub=True):
        super(kd_naive_depth_latent4_split_adapter_cbam, self).__init__()
        
        print('kd_naive_depth_latent4_split_adapter_cbam')
        assert encoder in ['vits', 'vitb', 'vitl']
        
        # in case the Internet connection is not stable, please load the DINOv2 locally
        if localhub:
            self.pretrained = torch.hub.load('torchhub/facebookresearch_dinov2_main', 'dinov2_{:}14'.format(encoder), source='local', pretrained=False)
        else:
            self.pretrained = torch.hub.load('facebookresearch/dinov2', 'dinov2_{:}14'.format(encoder))
        
        dim = self.pretrained.blocks[0].attn.qkv.in_features
        self.split_dim = dim // 4
        
        depth = 1
        drop_path_rate=0.0
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
            
        self.residual_self_layers3 = nn.ModuleList([
            nn.Sequential(
                Block(
                    dim=dim//4,
                    num_heads=6,
                    mlp_ratio=4,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=dpr[0],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    ffn_layer=Mlp,
                    init_values=None,
                ),
                Block(
                    dim=dim//4,
                    num_heads=6,
                    mlp_ratio=4,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=dpr[0],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    ffn_layer=Mlp,
                    init_values=None,
                ),
                Block(
                    dim=dim//4,
                    num_heads=6,
                    mlp_ratio=4,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=dpr[0],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    ffn_layer=Mlp,
                    init_values=None,
                ),
            ) for _ in range(4)])
            
        self.residual_self_layers4 = nn.ModuleList([
            nn.Sequential(
                Block(
                    dim=dim//4,
                    num_heads=6,
                    mlp_ratio=4,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=dpr[0],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    ffn_layer=Mlp,
                    init_values=None,
                ),
                Block(
                    dim=dim//4,
                    num_heads=6,
                    mlp_ratio=4,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=dpr[0],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    ffn_layer=Mlp,
                    init_values=None,
                ),
                Block(
                    dim=dim//4,
                    num_heads=6,
                    mlp_ratio=4,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=dpr[0],
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    act_layer=nn.GELU,
                    ffn_layer=Mlp,
                    init_values=None,
                )
            ) for _ in range(4)])

        self.residual_modules = [None, None, self.residual_self_layers3,self.residual_self_layers4]

        self.depth_head = DPTHead(1, dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        self.nomalize = NormalizeLayer()
        
    def forward(self, x):
        h, w = x.shape[-2:]
        
        features = self.pretrained.get_intermediate_layers(x, 4, return_class_token=False)
        
        residual_features = []
        for feat, residual_self_layers  in zip(features, self.residual_modules):
            if(residual_self_layers == None):
                residual_features.append(feat)
                continue
            layer_feat = []
            for i, residual_self in enumerate(residual_self_layers):
                split_self = feat[:,:,i*self.split_dim:(i+1)*self.split_dim]
                split_self = residual_self(split_self)
                layer_feat.append(split_self)
            redisual_feat = torch.cat(layer_feat, dim=2)
            sum_feat = feat + redisual_feat
            residual_features.append(sum_feat)

        patch_h, patch_w = h // 14, w // 14

        depth, cbam_adapter_feature = self.depth_head(residual_features, patch_h, patch_w)
        depth = F.interpolate(depth, size=(h, w), mode="bilinear", align_corners=True)
        depth = F.relu(depth)
        depth = self.nomalize(depth) if self.training else depth

        if self.training:
            return depth, features, residual_features, cbam_adapter_feature

        return depth
    
    def freeze_kd_naive_dpt_latent4_split_adapter_cbam_with_kd_style(self):
        
        for i, (name, param) in enumerate(self.pretrained.named_parameters()):
            param.requires_grad = False

        for i, (name, param) in enumerate(self.depth_head.named_parameters()):
            param.requires_grad = False

        self.depth_head.unfreeze_kd_style()
        

    
    
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
    