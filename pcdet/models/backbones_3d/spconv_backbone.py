from audioop import bias
from functools import partial
# from turtle import forward

# import spconv.pytorch as spconv
# from spconv.pytorch import functional as Fsp

import spconv
from spconv import functional as Fsp

import torch.nn as nn
import torch
from pcdet.ops.pointnet2.pointnet2_batch.pointnet2_utils import three_nn, three_interpolate
import numpy as np
from typing import List
from functools import reduce 


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        ### spconv 2.1
        # out = self.conv1(x)
        # out = out.replace_feature(self.bn1(out.features))
        # out = out.replace_feature(self.relu(out.features))

        # out = self.conv2(out)
        # out = out.replace_feature(self.bn2(out.features))

        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out = out.replace_feature(out.features + identity.features)
        # out = out.replace_feature(self.relu(out.features))

        return out

class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x, coors, batch_size):
        out = []
        for i in range(batch_size):
            cur_batch_index = coors[:, 0] == i
            cur_batch_features = x[cur_batch_index].permute(1, 0).contiguous().unsqueeze(0)
            b, c, _ = cur_batch_features.size()
            y = self.avg_pool(cur_batch_features).view(b, c)
            y = self.fc(y).view(b,c,1)
            se_out = cur_batch_features*y.expand_as(cur_batch_features)
            out.append(se_out.squeeze(0).permute(1, 0))
        out = torch.cat(out, dim=0)
        return out

class SparseSEBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseSEBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.se = SELayer3D(planes, 16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)
        out.features = self.se(out.features, out.indices, out.batch_size)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        ###  spconv 2.1
        # out = self.conv1(x)
        # out = out.replace_feature(self.bn1(out.features))
        # out = out.replace_feature(self.relu(out.features))

        # out = self.conv2(out)
        # out = out.replace_feature(self.bn2(out.features))
        # se_features = self.se(out.features, out.indices, out.batch_size)
        # out = out.replace_feature(se_features)
        # if self.downsample is not None:
        #     identity = self.downsample(x)

        # out = out.replace_feature(out.features + identity.features)
        # out = out.replace_feature(self.relu(out.features))

        return out

class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }
        
    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelResBackBone4x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(64),
            nn.ReLU(),
        )
        self.num_point_features = 64
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
        }


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv3)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 4
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
            }
        })

        return batch_dict

class VoxelSEResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseSEBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseSEBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseSEBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseSEBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseSEBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseSEBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseSEBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseSEBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelSEResBackBone8xV1(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        #self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.sparse_shape = grid_size[::-1] 

        block = post_act_block

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
            ##1504
            SparseSEBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseSEBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            ###752
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseSEBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseSEBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),

            ###376
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseSEBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseSEBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        ###stage 1
        ### 376
        self.p1_stage1 = SparseSEBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3')

        ### fuse 376->188
        self.p1_stage1_t12 = block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv')
        self.p1_stage1_t11 = block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res3', conv_type='subm')

        ### stage2
        ### 376 188
        self.p1_stage2 = SparseSEBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3')
        self.p2_stage2 = SparseSEBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4')

        # ### v1
        self.p1_conv = block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res3', conv_type='subm')
        self.p2_conv = block(128, 128, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res4', conv_type='subm')
        self.p1t2 = block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv6', conv_type='spconv')  
        self.p2t2 = block(128, 128, 3, norm_fn=norm_fn, indice_key='res4', conv_type='subm')
        
        ### v2
        # self.p1t2 = block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv6', conv_type='spconv')  
        # self.p2t2 = block(128, 128, 3, norm_fn=norm_fn, indice_key='res4', conv_type='subm')
        # self.p1_conv = SparseSEBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4')
        # self.p2_conv = SparseSEBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4')


        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv9'),
            norm_fn(128),
            nn.ReLU(),
        )

        ### se out
        # self.out = SparseSEBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res5')
        self.num_point_features = 128

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        p1 = self.input_conv(input_sp_tensor)
        p1 = self.p1_stage1(p1)

        p2 = self.p1_stage1_t12(p1)
        p1 = self.p1_stage1_t11(p1)
        p1 = self.p1_stage2(p1)
        p2 = self.p2_stage2(p2)

        # ### v1
        p1_out = self.p1_conv(p1)
        p2_out = self.p2_conv(p2)
        ### add
        hr_out = sparse_add(self.p1t2(p1_out), self.p2t2(p2_out))
        out = self.conv_out(hr_out)

        # ### v2
        # p1_out = self.p1t2(p1)
        # p2_out = self.p2t2(p2)
        # ### add
        # hr_out = sparse_add(self.p1_conv(p1_out), self.p2_conv(p2_out))
        # out = self.conv_out(hr_out)
        # out = self.out(out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        return batch_dict

class VoxelSEResBackBone8xV2(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        #self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.sparse_shape = grid_size[::-1] 

        block = post_act_block

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
            ##1504
            SparseSEBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseSEBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            ###752
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseSEBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseSEBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),

            ###376
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseSEBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseSEBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        ###stage 1
        ### 376
        self.p1_stage1 = SparseSEBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3')

        ### fuse 376->188
        self.p1_stage1_t12 = block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv')
        self.p1_stage1_t11 = block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res3', conv_type='subm')

        ### stage2
        ### 376 188
        self.p1_stage2 = SparseSEBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3')
        self.p2_stage2 = SparseSEBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4')

        # ### fuse
        self.p1_stage2_t11 = block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res3', conv_type='subm')
        self.p2_stage2_t22 = block(128, 128, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res4', conv_type='subm')
        self.p2_stage2_t23 = block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv5', conv_type='spconv')
        
        # ### stage3 376, 188, 94
        self.p1_stage3 = SparseSEBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3')
        self.p2_stage3 = SparseSEBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4')
        self.p3_stage3 = SparseSEBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5')

        ### fuse
        
        self.p1_conv = block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res3', conv_type='subm')
        self.p2_conv = block(128, 128, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res4', conv_type='subm')
        self.p3_conv = block(256, 256, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res5', conv_type='subm')
        
        self.p1t1 = block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res3', conv_type='subm')
        self.p2t1 = block(128, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')
        self.p3t1 = spconv.SparseSequential( 
            block(256, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv5', conv_type='inverseconv'),
            SparseSEBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            block(128, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='inverseconv')
            )
        # self.p1t2 = block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv6', conv_type='spconv')
            
        # self.p2t2 = block(128, 128, 3, norm_fn=norm_fn, indice_key='res4', conv_type='subm')
        # self.p3t3 = block(256, 256, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res5', conv_type='subm')
        
        
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv9'),
            norm_fn(64),
            nn.ReLU(),
        )
        self.num_point_features = 64

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        p1 = self.input_conv(input_sp_tensor)
        p1 = self.p1_stage1(p1)

        p2 = self.p1_stage1_t12(p1)
        p1 = self.p1_stage1_t11(p1)
        

        p1 = self.p1_stage2(p1)
        p2 = self.p2_stage2(p2)

        p3 = self.p2_stage2_t23(p2)
        p1 = self.p1_stage2_t11(p1)
        p2 = self.p2_stage2_t22(p2)

        p1 = self.p1_stage3(p1)
        p2 = self.p2_stage3(p2)
        p3 = self.p3_stage3(p3)

        p1_out = self.p1_conv(p1)
        p2_out = self.p2_conv(p2)
        p3_out = self.p3_conv(p3)


        hr_out = sparse_add(self.p1t1(p1_out), self.p2t1(p2_out), self.p3t1(p3_out))

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(hr_out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 4
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        return batch_dict

class VoxelSEResBackBone8xV3(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        # self.sparse_shape = grid_size[::-1]
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.conv_out3 = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down3'),
            norm_fn(64),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        
        out = self.conv_out(x_conv4)
        out3 = self.conv_out3(x_conv3)
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_sf4': out3,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelResBackBone8xV1(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.conv_out_3 = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down3'),
            norm_fn(64),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        out_3 = self.conv_out_3(x_conv3)

        batch_dict.update({
            'encoded_spconv_tensor_8x': out,
            'encoded_spconv_tensor_4x': out_3,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict

class VoxelResBackBone8xV2(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        
        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict

class VoxelResBackBone8xV3(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        #self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.sparse_shape = grid_size[::-1] 

        block = post_act_block

        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
            ##1504
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            ###752
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),

            ###376
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        ###stage 1
        ### 376
        self.p1_stage1 = SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3')

        ### fuse 376->188
        self.p1_stage1_t12 = block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='spconv')
        self.p1_stage1_t11 = block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res3', conv_type='subm')

        ### stage2
        ### 376 188
        self.p1_stage2 = SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3')
        self.p2_stage2 = SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4')

        # ### fuse
        self.p1_stage2_t11 = block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res3', conv_type='subm')
        self.p2_stage2_t22 = block(128, 128, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res4', conv_type='subm')
        # self.p2_stage2_t23 = block(128, 256, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv5', conv_type='spconv')
        
        # ### stage3 376, 188, 94
        self.p1_stage3 = SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3')
        self.p2_stage3 = SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4')
        # self.p3_stage3 = SparseBasicBlock(256, 256, norm_fn=norm_fn, indice_key='res5')

        ### fuse
        
        self.p1_conv = block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res3', conv_type='subm')
        self.p2_conv = block(128, 128, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res4', conv_type='subm')
        # self.p3_conv = block(256, 256, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res5', conv_type='subm')
        
        self.p1t1 = block(64, 64, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res3', conv_type='subm')
        self.p2t1 = block(128, 64, 3, norm_fn=norm_fn, indice_key='spconv4', conv_type='inverseconv')
        # self.p3t1 = spconv.SparseSequential( 
        #     block(256, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv5', conv_type='inverseconv'),
        #     SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        #     block(128, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv4', conv_type='inverseconv')
        #     )
        # self.p1t2 = block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv6', conv_type='spconv')
            
        # self.p2t2 = block(128, 128, 3, norm_fn=norm_fn, indice_key='res4', conv_type='subm')
        # self.p3t3 = block(256, 256, 3, norm_fn=norm_fn, stride=1, padding=1, indice_key='res5', conv_type='subm')
        
        
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 64, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv9'),
            norm_fn(64),
            nn.ReLU(),
        )
        self.num_point_features = 64

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        p1 = self.input_conv(input_sp_tensor)
        p1 = self.p1_stage1(p1)

        p2 = self.p1_stage1_t12(p1)
        p1 = self.p1_stage1_t11(p1)
        

        p1 = self.p1_stage2(p1)
        p2 = self.p2_stage2(p2)

        # p3 = self.p2_stage2_t23(p2)
        p1 = self.p1_stage2_t11(p1)
        p2 = self.p2_stage2_t22(p2)

        p1 = self.p1_stage3(p1)
        p2 = self.p2_stage3(p2)
        # p3 = self.p3_stage3(p3)

        p1_out = self.p1_conv(p1)
        p2_out = self.p2_conv(p2)
        # p3_out = self.p3_conv(p3)


        # hr_out = sparse_add(self.p1t1(p1_out), self.p2t1(p2_out), self.p3t1(p3_out))
        hr_out = sparse_add(self.p1t1(p1_out), self.p2t1(p2_out))

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(hr_out)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 4
        })
        # batch_dict.update({
        #     'multi_scale_3d_features': {
        #         'x_conv1': x_conv1,
        #         'x_conv2': x_conv2,
        #         'x_conv3': x_conv3,
        #         'x_conv4': x_conv4,
        #     }
        # })

        return batch_dict

class VoxelBackBone8xV1(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.bn_input = nn.BatchNorm1d(304)
        self.fc = torch.nn.Linear(304, 128, bias=False)
        self.bn = nn.BatchNorm1d(128)

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        encode_features = []
        x = self.conv_input(input_sp_tensor)
        x_conv1 = self.conv1(x)
        encode_features.append(x_conv1)
        x_conv2 = self.conv2(x_conv1)
        encode_features.append(x_conv2)
        x_conv3 = self.conv3(x_conv2)
        encode_features.append(x_conv3)
        x_conv4 = self.conv4(x_conv3)
        encode_features.append(x_conv4)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        out_voxel_size = np.array(self.voxel_size) * 8.
        out_voxel_size[-1] = out_voxel_size[-1] * 2
        offset = self.point_cloud_range[:3]
        vx_feature, points_mean = tensor2points(out, offset=offset, voxel_size=out_voxel_size)
        
        multi_scales_feature = []
        for i in range(len(encode_features)):
            voxel_size = np.array(self.voxel_size)*(2**i)
            vx_feat, vx_nxyz = tensor2points(encode_features[i], offset=offset, voxel_size=voxel_size)
            interpolate_feature = nearest_neighbor_interpolate(points_mean, vx_nxyz, vx_feat, batch_size)
            multi_scales_feature.append(interpolate_feature)
        multi_scales_feature = torch.cat(multi_scales_feature, dim=1)
        concatenate_feature = torch.cat((multi_scales_feature, out.features), dim=1)
        feature_with_semantic = torch.nn.functional.relu(self.bn(self.fc(self.bn_input(concatenate_feature))))
        output_sp_tensor = spconv.SparseConvTensor(feature_with_semantic, out.indices.int(), out.spatial_shape,
                                                   batch_size)
        # spatial_features = output_sp_tensor.dense()
        # N, C, D, H, W = spatial_features.shape
        # spatial_features = spatial_features.view(N, C * D, H, W)

        batch_dict.update({
            'encoded_spconv_tensor': output_sp_tensor,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict

def tensor2points(tensor, offset=(0., -40., -3.), voxel_size=(.05, .05, .1)):
    indices = tensor.indices.float()
    offset = torch.Tensor(offset).to(indices.device)
    voxel_size = torch.Tensor(voxel_size).to(indices.device)
    indices[:, 1:] = indices[:, [3, 2, 1]] * voxel_size + offset + .5 * voxel_size
    return tensor.features, indices

def nearest_neighbor_interpolate(unknown, known, known_feats, batch_size):
    """
    :param pts: (n, 4) tensor of the bxyz positions of the unknown features
    :param ctr: (m, 4) tensor of the bxyz positions of the known features
    :param ctr_feats: (m, C) tensor of features to be propigated
    :return:
        new_features: (n, C) tensor of the features of the unknown features
    """
    interpolated_feats_list = []
    for i in range(batch_size):
        cur_unknown = unknown[unknown[:, 0]==i]
        cur_unknown = cur_unknown.unsqueeze(0)
        cur_known = known[known[:, 0]==i]
        cur_known = cur_known.unsqueeze(0)
        cur_known_feats = known_feats[known[:, 0]==i]
        cur_known_feats = cur_known_feats.transpose(1, 0).unsqueeze(0)
        dist, idx = three_nn(cur_unknown, cur_known)
        dist_recip = 1.0 / (dist + 1e-8)
        norm = torch.sum(dist_recip, dim=1, keepdim=True)
        weight = dist_recip / norm
        interpolated_feats = three_interpolate(cur_known_feats.contiguous(), idx.contiguous(), weight.contiguous())
        interpolated_feats_list.append(interpolated_feats.squeeze(0).transpose(1, 0))
    interpolated_feats = torch.cat(interpolated_feats_list)
    return interpolated_feats

def sparse_add(*tens: spconv.SparseConvTensor):
    """reuse torch.sparse. the internal is sort + unique 
    """
    max_num_indices = 0
    max_num_indices_idx = 0
    ten_ths: List[torch.Tensor] = []
    first = tens[0]
    res_shape = [first.batch_size, *first.spatial_shape, first.features.shape[1]]

    for i, ten in enumerate(tens):
        assert ten.spatial_shape == tens[0].spatial_shape
        assert ten.batch_size == tens[0].batch_size
        assert ten.features.shape[1] == tens[0].features.shape[1]
        if max_num_indices < ten.features.shape[0]:
            max_num_indices_idx = i
            max_num_indices = ten.features.shape[0]
        ten_ths.append(torch.sparse_coo_tensor(ten.indices.T, ten.features, res_shape, requires_grad=True))
    
    c_th = reduce(lambda x, y: x + y, ten_ths).coalesce()
    c_th_inds = c_th.indices().T.contiguous().int()
    c_th_values = c_th.values()
    assert c_th_values.is_contiguous()

    res = spconv.SparseConvTensor(c_th_values, c_th_inds, first.spatial_shape, first.batch_size)
    if c_th_values.shape[0] == max_num_indices:
        res.indice_dict = tens[max_num_indices_idx].indice_dict
    # res.benchmark_record = first.benchmark_record
    # res._timer = first._timer 
    # res.thrust_allocator = first.thrust_allocator
    return res

def sparse_cat(*tens: spconv.SparseConvTensor):
    """reuse torch.sparse. the internal is sort + unique 
    """
    first = tens[0]
    for i in range(1, len(tens)):
        first.features = torch.cat([first.features, tens[i].features], dim=1)
    return first 