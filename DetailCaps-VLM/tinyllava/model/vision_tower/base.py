import os

import torch
import torch.nn as nn
from torch.nn import MultiheadAttention, LayerNorm
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models.resnet import ResNet50_Weights
from torchvision.models.resnet import ResNet101_Weights
from torchvision.ops import DeformConv2d
import numpy as np
import math
import random
from transformers import PreTrainedModel
from timm.models.helpers import named_apply
from functools import partial
from torch.nn.functional import interpolate
# from tinyllava.utils.data_utils import get_value_from_kwargs
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
def get_value_from_kwargs(kwargs, name):
    if name in kwargs:
        return kwargs.pop(name)
    else:
        return None
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, input_en, input_de, descriptors):
        return input_de
class ACA(nn.Module):
    def __init__(self, c_in, c_feat, c_atten):
        super(ACA, self).__init__()
        self.c_feat = c_feat
        self.c_atten = c_atten
        self.conv_feat = nn.Conv2d(c_in, c_feat, kernel_size=1)
        self.conv_atten = nn.Conv2d(c_in, c_atten, kernel_size=1)
    def forward(self, input: torch.Tensor):
        b, c, h, w = input.size()
        feat = self.conv_feat(input).view(b, self.c_feat, -1)  # feature map
        atten = self.conv_atten(input).view(b, self.c_atten, -1)  # attention map
        atten = F.softmax(atten, dim=-1)
        descriptors = torch.bmm(feat, atten.permute(0, 2, 1))  # (c_feat, c_atten)
        return descriptors
class SDM(nn.Module):
    def __init__(self, c_atten, c_de):
        super(SDM, self).__init__()
        self.c_atten = c_atten
        self.conv_de = nn.Conv2d(c_de, c_atten, kernel_size=1)
        self.out_conv = nn.Conv2d(c_de, c_de, kernel_size=1)

    def forward(self, descriptors: torch.Tensor, input_de: torch.Tensor):
        b, c, h, w = input_de.size()
        atten_vectors = F.softmax(self.conv_de(input_de), dim=1)
        output = descriptors.matmul(atten_vectors.view(b, self.c_atten, -1)).view(b, -1, h, w)
        return self.out_conv(output)
class Engery(nn.Module):
    def __init__(self, c_en, c_de):
        super(Engery, self).__init__()
        self.c_en = c_en
        self.c_de = c_de

    def forward(self, input_en: torch.Tensor, input_de: torch.Tensor, gate_map):
        b, c, h, w = input_de.size()
        input_en = input_en.view(b, self.c_en, -1)
        # Channel Resampling
        energy = input_de.view(b, self.c_de, -1).matmul(input_en.transpose(-1, -2))
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(
            energy) - energy  # Prevent loss divergence during training
        channel_attention_map = torch.softmax(energy_new, dim=-1)
        input_en = channel_attention_map.matmul(input_en).view(b, -1, h, w)  # channel_attention_feat
        # Spatial Gating
        gate_map = torch.sigmoid(gate_map)
        input_en = input_en.mul(gate_map)
        return input_en
class HCFG(nn.Module):
    def __init__(self, fpn_dim=256, c_atten=256, norm_layer=None, ):
        super(HCFG, self).__init__()
        self.sdm = SDM(c_atten, fpn_dim)
        self.engery = Engery(fpn_dim, fpn_dim)
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            # nn.Conv2d(3 * fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            # DepthwiseConv(3 * fpn_dim, fpn_dim),
            norm_layer(fpn_dim),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.alpha = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

    def forward(self, input_en: torch.Tensor, input_de: torch.Tensor, global_descripitors: torch.Tensor):
        feat_global = self.sdm(global_descripitors, input_de)
        feat_local = self.gamma * self.engery(input_en, input_de, feat_global) + input_en
        # add fusion
        return self.conv_fusion(input_de + self.alpha * feat_global + self.beta * feat_local)
        # concat fusion
        # return self.conv_fusion(torch.cat((input_de, self.beta * feat_global, self.gamma * feat_local), dim=1))
class MSCORCH(nn.Module):
    def __init__(self, out_channels, norm_layer=None, fpn_inchannels=[256, 512, 1024],  # 修改输入通道
                 fpn_dim=256, up_kwargs=None):
        super(MSCORCH, self).__init__()
        # 确保up_kwargs至少是空字典
        self._up_kwargs = up_kwargs if up_kwargs is not None else {}
        
        # FPN lateral convolutions (for f2 and f3)
        self.fpn_lateral = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, fpn_dim, 1, bias=False),  # f2
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(512, fpn_dim, 1, bias=False),  # f3
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True)
            )
        ])
        
        # FPN output convolutions
        self.fpn_out = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1, bias=False),
                norm_layer(fpn_dim),
                nn.ReLU(inplace=True)
            )
        ])

        # 处理f4层
        self.e4conv = nn.Sequential(
            nn.Conv2d(1024, fpn_dim, 3, padding=1, bias=False),
            norm_layer(fpn_dim),
            nn.ReLU()
        )

        # 调整ACA模块
        self.aca_d4 = ACA(fpn_dim, fpn_dim, fpn_dim//4)  # 处理f4层
        self.aca_d3 = ACA(fpn_dim, fpn_dim, fpn_dim//4)  # 处理f3层
        self.aca_d2 = ACA(fpn_dim, fpn_dim, fpn_dim//4)  # 处理f2层

        # 调整HCFG模块数量
        self.hcfg_d4 = nn.ModuleList([HCFG(fpn_dim, fpn_dim//4, norm_layer) for _ in range(2)])
        self.hcfg_d3 = nn.ModuleList([HCFG(fpn_dim, fpn_dim//4, norm_layer) for _ in range(2)])
        self.hcfg_d2 = nn.ModuleList([HCFG(fpn_dim, fpn_dim//4, norm_layer), Identity()])

        # 最终输出卷积
        self.conv5 = nn.Sequential(
            nn.Conv2d(3*fpn_dim, 512, 3, padding=1, bias=False),
            norm_layer(512),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(512, out_channels, 1)
        )

    def forward(self, f2, f3, f4):  # 输入改为三个层
        # 处理f4层
        feat = self.e4conv(f4)
        descriptors = [None, None, self.aca_d4(feat)]  # 初始化描述符

        # 特征融合流程
        e1_size = f2.size()[2:]
        outputs = [interpolate(feat, e1_size, **self._up_kwargs)]

        # 从深层到浅层处理（f3->f2）
        for i in reversed(range(2)):  # 处理两个层（索引1和0）
            # 获取当前层特征
            if i == 1:
                feat_i = self.fpn_lateral[1](f3)  # 处理f3
            else:
                feat_i = self.fpn_lateral[0](f2)  # 处理f2

            # 上采样深层特征
            feat_up = interpolate(feat, feat_i.shape[2:], **self._up_kwargs)
            
            # 更新描述符
            if i == 1:
                descriptors[1] = self.aca_d3(feat)
            else:
                descriptors[0] = self.aca_d2(feat)

            # 特征融合
            feat_up = self.hcfg_d4[i](feat_i, feat_up, descriptors[2])
            feat_up = self.hcfg_d3[i](feat_i, feat_up, descriptors[1])
            feat = self.hcfg_d2[i](feat_i, feat_up, descriptors[0])

            # 保存上采样结果
            outputs.append(interpolate(self.fpn_out[i](feat), e1_size, **self._up_kwargs))

        # 最终输出
        outputs = torch.cat(outputs, dim=1)
        return self.conv5(outputs)
class SSA(nn.Module):
    def __init__(self, in_channels=1024, reduction_ratio=8):
        super().__init__()
        # 空间增强分支 (轻量化设计)
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels), # 深度可分离卷积
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels//reduction_ratio, 1), # 压缩通道
            nn.GELU(),
            nn.Conv2d(in_channels//reduction_ratio, in_channels, 1)  # 恢复通道
        )
        
        # 通道注意力(CAM)
        self.local_att = nn.Sequential(
            nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(1,in_channels//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction_ratio, in_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(1,in_channels),
        )
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels//reduction_ratio, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(1,in_channels//reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels//reduction_ratio,in_channels, kernel_size=1, stride=1, padding=0),
            nn.GroupNorm(1,in_channels),
        )
        self.sigmoid = nn.Sigmoid()
       
        
        # 残差归一化
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        # 输入形状: (32, 728, 1152)
        B, N, C = x.shape
        
        # 空间增强分支 ----------------------------------
        # 假设输入序列可还原为 28x26 网格 (28*26=728)
        h, w = 24, 24
        x_spatial = x.view(B, h, w, C).permute(0, 3, 1, 2) # [B, C, H, W]
        spatial_feat = self.spatial_conv(x_spatial)         # 空间增强
        spatial_feat = spatial_feat.permute(0, 2, 3, 1).view(B, N, C)
        
        # 通道注意力分支 ---------------------------------
        xl = self.local_att(x_spatial)
        xg = self.global_att(x_spatial)
        xlg = xl + xg
        wei = self.sigmoid(xlg)
        
        # 特征融合 --------------------------------------
        out = self.norm(spatial_feat + x) # 残差连接 + 空间增强
        out = out * wei.reshape(B,576,-1)# 通道注意力加权
        
        return out
class DSVAB(nn.Module):
    def __init__(self, d_model=2560, n_heads=8, n_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        
        # 线性变换层
        self.linear_rsclip = nn.Linear(1024, d_model)   # CLIP特征维度->d_model
        self.linear_rn101 = nn.Linear(512, d_model)   # ResNet特征维度->d_model
        
        # 多头注意力模块
        self.self_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.cross_attn = MultiheadAttention(d_model, n_heads, dropout=dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 1024),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(1024, d_model)
        )
        
        # 层归一化
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        
        # 多层堆叠支持
        self.n_layers = n_layers
        self.dropout = nn.Dropout(dropout)

    def forward(self, F_rsclip, F_rn101):
        """输入:
        F_rsclip: [m, batch_size, 768] 
        F_rn101: [n, batch_size, 2048]
        """
        # Step 1: 线性变换
        F_rsclip = self.linear_rsclip(F_rsclip)  # [m, B, d]
        F_rn101 = self.linear_rn101(F_rn101)    # [n, B, d]
        print(F_rsclip.shape,F_rn101.shape)
        for _ in range(self.n_layers):
            # Step 2: 自注意力
            attn_rsclip, _ = self.self_attn(
                F_rsclip, F_rsclip, F_rsclip
            )
            attn_rsclip = self.dropout(attn_rsclip)
            
            # Step 3: 交叉注意力
            cross_attn, _ = self.cross_attn(
                F_rsclip, F_rn101, F_rn101
            )
            cross_attn = self.dropout(cross_attn)
            
            #step 4:第一次残差连接
            F_im = self.norm1(F_rsclip+attn_rsclip+cross_attn)
            
            
            # Step 5: 前馈网络
            ffn_out = self.ffn(F_im)
            ffn_out = self.dropout(ffn_out)
            F_fusion = self.norm3(F_im + ffn_out)
            
            # 更新特征用于下一层
            F_rn101 = F_fusion
            F_rsclip = F_fusion
            
        return F_fusion
class VisionTower(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self._vision_tower = None
        self._image_processor = None
        self.config = cfg
        self.resnet50 = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1).to('cuda')
        self.feature_extractor_c2 = torch.nn.Sequential(*list(self.resnet50.children())[:5]).to('cuda')
        self.feature_extractor_c3 = torch.nn.Sequential(*list(self.resnet50.children())[:6]).to('cuda')
        self.feature_extractor_c4 = torch.nn.Sequential(*list(self.resnet50.children())[:7]).to('cuda')
        self.MSCORCH = MSCORCH(out_channels=512, norm_layer=lambda channels: nn.GroupNorm(num_groups=32, num_channels=channels)).to('cuda')  # 对齐到f3的尺寸
        self.SSA = SSA().to(device = 'cuda:0')
        self.DSVAB = DSVAB().to(device = 'cuda:0')
        
    

    def load_model(self, vision_tower_name, **kwargs):
        self._load_model(vision_tower_name, **kwargs)
        self._vision_tower.requires_grad_(False)


    def load_model_MSCORCH(self,**kwargs):
        pretrained_MSCORCH_path = get_value_from_kwargs(kwargs, 'pretrained_MSCORCH_path')
        MSCORCH_weights = torch.load(os.path.join(pretrained_MSCORCH_path, 'pytorch_model.bin'), map_location='cpu')
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        self.MSCORCH.load_state_dict(MSCORCH_weights)
    def load_model_SSA(self,**kwargs):
        pretrained_SSA_path = get_value_from_kwargs(kwargs, 'pretrained_SSA_path')
        SSA_weights = torch.load(os.path.join(pretrained_SSA_path, 'pytorch_model.bin'), map_location='cpu')
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        self.SSA.load_state_dict(SSA_weights)
    def load_model_DSVAB(self,**kwargs):
        pretrained_DSVAB_path = get_value_from_kwargs(kwargs, 'pretrained_DSVAB_path')
        DSVAB_weights = torch.load(os.path.join(pretrained_DSVAB_path, 'pytorch_model.bin'), map_location='cpu')
        def get_w(weights, keyword):
            return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
        self.DSVAB.load_state_dict(DSVAB_weights)
        
    def _load_model(self, vision_tower_name, **kwargs):
        pretrained_vision_tower_path = get_value_from_kwargs(kwargs, 'pretrained_vision_tower_path')
        if isinstance(self._vision_tower, PreTrainedModel): # hf model
            if pretrained_vision_tower_path is not None:
                vision_tower_name = pretrained_vision_tower_path
            self._vision_tower = self._vision_tower.from_pretrained(vision_tower_name, **kwargs)      
        else: # nn.Module
            if pretrained_vision_tower_path is not None:
                vision_tower_weights = torch.load(os.path.join(pretrained_vision_tower_path, 'pytorch_model.bin'), map_location='cpu')
                def get_w(weights, keyword):
                    return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
                self._vision_tower.load_state_dict(vision_tower_weights)

        print("Loading vision tower from ", vision_tower_name)
        


    def forward(self, x, **kwargs):
        image_features = self._vision_tower(x, output_hidden_states=True)
        image_features = image_features.hidden_states[kwargs.get('vision_feature_layer', -2)]

        if kwargs.get('vision_feature_select_strategy', 'patch') == 'patch':
            image_features = image_features[:, 1:]
        elif kwargs.get('vision_feature_select_strategy', 'patch') == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f"Unexpected select feature: {kwargs.get('vision_feature_select_strategy')}")
        images_res = x.to(torch.float16)
        images_features_res_c2 = self.feature_extractor_c2(images_res).to(torch.float16)
        images_features_res_c3 = self.feature_extractor_c3(images_res).to(torch.float16)
        images_features_res_c4 = self.feature_extractor_c4(images_res).to(torch.float16)
        global_features = self.SSA(image_features)
        concepts = self.MSCORCH(images_features_res_c2, images_features_res_c3, images_features_res_c4)
        global_features = global_features.transpose(0,1)
        concepts = concepts.view(-1,84*84,512)
        concepts = concepts.transpose(0,1)
        fused_feature = self.DSVAB(global_features,concepts)
        fused_feature = fused_feature.transpose(0,1)
        return fused_feature
        

    
    @property
    def vision_tower(self):
        return self._vision_tower
        
    @vision_tower.setter
    def vision_tower(self, vision_tower):
        self._vision_tower = vision_tower
        
    
