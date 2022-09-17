from os import replace
#from turtle import forward
import torch
import torch.nn as nn
from pcdet.ops.adapool.adaPool import AdaPool2d
from pcdet.ops.softpool.SoftPool import SoftPool2d

class HeightCompression(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor = batch_dict['encoded_spconv_tensor']
        spatial_features = encoded_spconv_tensor.dense()
        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict


class HeightCompressionV1(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        self.conv_4x_1 = nn.Sequential(
            nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        # self.conv_4x_2 = nn.Sequential(
        #     nn.Conv2d(128*4, 128, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(128),
        #     nn.ReLU()
        # )
        self.deconv_8x = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 2, stride=2, bias=False),
            nn.BatchNorm2d(256, eps=1e-3, momentum=0.01),
            nn.ReLU()
            )
        # self.conv_fusion = SERES_block(256, 256)
        # self.conv_out = nn.Sequential(
        #     nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU()
        # )

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor_8x = batch_dict['encoded_spconv_tensor_8x']
        spatial_features_8x = encoded_spconv_tensor_8x.dense()
        N, C, D, H, W = spatial_features_8x.shape
        spatial_features_8x = spatial_features_8x.view(N, C * D, H, W)

        encoded_spconv_tensor_4x = batch_dict['encoded_spconv_tensor_4x']
        spatial_features_4x = encoded_spconv_tensor_4x.dense()
        N, C, D, H, W = spatial_features_4x.shape
        spatial_features_4x = spatial_features_4x.view(N, C * D, H, W)
        spatial_features_4x = self.conv_4x_1(spatial_features_4x)
        # spatial_features_4x = space2depth(spatial_features_4x, 2)
        # spatial_features_4x = self.conv_4x_2(spatial_features_4x)

        spatial_features_8x_4x = self.deconv_8x(spatial_features_8x)

        # fusion_features = torch.cat([spatial_features_4x, spatial_features_8x_4x], dim=1)
        fusion_features = spatial_features_4x + spatial_features_8x_4x

        # fusion_features = self.conv_fusion(fusion_features)
        # fusion_features = self.conv_out(fusion_features)
        batch_dict['spatial_features'] = fusion_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict

class HeightCompressionV2(nn.Module):
    def __init__(self, model_cfg, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.num_bev_features = self.model_cfg.NUM_BEV_FEATURES
        # self.pool = AdaPool2d(kernel_size=2, stride=2, beta=(188, 188))
        self.pool = SoftPool2d(kernel_size=2, stride=2)
        self.se_conv = SERES_block(576, 576)
        

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        Returns:
            batch_dict:
                spatial_features:

        """
        encoded_spconv_tensor_8x = batch_dict['encoded_spconv_tensor_8x']
        spatial_features_8x = encoded_spconv_tensor_8x.dense()
        N, C, D, H, W = spatial_features_8x.shape
        spatial_features_8x = spatial_features_8x.view(N, C * D, H, W)

        encoded_spconv_tensor_4x = batch_dict['encoded_spconv_tensor_4x']
        spatial_features_4x = encoded_spconv_tensor_4x.dense()
        N, C, D, H, W = spatial_features_4x.shape
        spatial_features_4x = spatial_features_4x.view(N, C * D, H, W)
        spatial_features_4x = self.pool(spatial_features_4x)

        spatial_features = torch.cat([spatial_features_8x, spatial_features_4x], dim=1)
        spatial_features = self.se_conv(spatial_features)
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = batch_dict['encoded_spconv_tensor_stride']
        return batch_dict

class SERES_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.se = SE_block(out_channel, reduction=16)
    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = self.se(x)
        out += identity
        out = self.relu(out)
        return out

class SE_block(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.weight = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel//reduction, channel, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.maxpool(x).view(b, c)
        y = self.weight(y).view(b, c, 1, 1)
        y = torch.mul(x, y)
        return y

def space2depth(x, down_scale):
    b, c, h, w = x.size()
    s2d_x = torch.nn.functional.unfold(x, down_scale, stride=down_scale)
    return s2d_x.view(b, c*down_scale*down_scale, h // down_scale, w // down_scale)