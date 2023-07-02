# ------------------------------------------
# CSWin Transformer
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# written By Xiaoyi Dong
# ------------------------------------------


import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from einops.layers.torch import Rearrange
import torch.utils.checkpoint as checkpoint
import numpy as np
from einops import rearrange, einsum
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .875, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'cswinmlp_224': _cfg(),
    'cswinmlp_384': _cfg(
        crop_pct=1.0
    ),
}


class ScaleLayer(nn.Module):
    def __init__(self, alpha=0.2, learnable=True, dim=1):
        super().__init__()
        self.alpha = alpha
        self.learnable = learnable
        self.dim = dim
        if self.learnable:
            self.scale = nn.Parameter(torch.ones(dim) * self.alpha)
        else:
            self.scale = self.alpha

    def forward(self, x):
        if self.learnable:
            y = self.scale[None, None, :]*x
        else:
            y = self.scale*x
        return  y

    def __repr__(self):
        return f"ScaleLayer(alpha={self.alpha}, learnable={self.learnable}, dim={self.dim})"


class CenterNorm(nn.Module):
    r""" CenterNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.scale = normalized_shape/(normalized_shape-1.0)
    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        x = self.scale * (x - u)
        x = self.weight[None, None, :] * x + self.bias[None, None, :]
        return x

    def __repr__(self):
        return "CenterNorm()"
    

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MixingAttention(nn.Module):
    def __init__(self, dim, resolution, idx, num_heads=8, split_size=7, dim_out=None, d=2, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.num_heads = num_heads
        self.resolution = resolution
        self.split_size = split_size
        assert self.resolution % self.split_size == 0
        self.d = d
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print ("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.x_windows = self.resolution // H_sp
        self.y_windows = self.resolution // W_sp

        self.compress = nn.Linear(dim, num_heads * d)
        self.generate = nn.Linear(H_sp * W_sp * d, (H_sp * W_sp) ** 2)
        self.activation = nn.Softmax(dim=-2)

        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, x):
        """
        x: B N C
        """
        H_sp, W_sp = self.H_sp, self.W_sp
        weights = rearrange(self.compress(x), "b (n1 h n2 w) (m d) -> b (n1 n2 m) (h w d)", 
                            n1=self.x_windows, h=H_sp, n2=self.y_windows, w=W_sp, m=self.num_heads)
        weights = rearrange(self.generate(weights), "b N (h1 w1 h2 w2) -> b N (h1 w1) (h2 w2)",
                            h1=H_sp, w1=W_sp, h2=H_sp, w2=W_sp)
        weights = self.activation(weights)
        x = rearrange(x, "b (n1 h1 n2 w1) (m c) -> b (n1 n2 m) c (h1 w1)",
                      n1=self.x_windows, h1=H_sp, n2=self.y_windows, w1=W_sp, m=self.num_heads)
        x = torch.matmul(x, weights)
        x = rearrange(x, "b (n1 n2 m) d (h2 w2) -> b (n1 h2 n2 w2) (m d)", n1=self.x_windows, n2=self.y_windows, h2=H_sp, w2=W_sp)

        return x        # B N C


class CSWinMLPLayer(nn.Module):

    def __init__(self, dim, reso, d, num_heads,
                 split_size=7, mlp_ratio=4., qkv_bias=False, 
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=CenterNorm,
                 num_layers=12, last_stage=False):
        super().__init__()
        self.dim = dim
        self.d = d
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        if last_stage:
            self.attns = nn.ModuleList([
                MixingAttention(
                    dim, resolution=self.patches_resolution, idx = -1,
                    split_size=split_size, d=d, dim_out=dim, num_heads=num_heads,
                    attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                MixingAttention(
                    dim//2, resolution=self.patches_resolution, idx = i,
                    split_size=split_size, d=d, dim_out=dim//2, num_heads=num_heads,
                    attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)
        self.alpha1 = ScaleLayer(dim=dim, alpha=1.0/num_layers)  #**alpha_cfg)
        self.alpha2 = ScaleLayer(dim=dim, alpha=1.0/num_layers)  #**alpha_cfg)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, N, C = x.shape
        assert N == H * W, "flatten img_tokens has wrong size"
        
        attended_x = self.norm1(x)
        if self.branch_num == 2:
            x1 = self.attns[0](attended_x[:,:,:C//2])
            x2 = self.attns[1](attended_x[:,:,C//2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](attended_x)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(self.alpha1(attened_x))
        x = x + self.drop_path(self.alpha2(self.mlp(self.norm2(x))))

        return x


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=CenterNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        
        return x


class CSWinMLPTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=96, depth=[2,2,6,2], split_size = [3,5,7],
                 d=2, num_heads=12, mlp_ratio=4., qkv_bias=True, drop_rate=0., attn_drop_rate=0.,
                 drop_path=0., hybrid_backbone=None, norm_layer=CenterNorm, use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads=num_heads

        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h = img_size//4, w = img_size//4),
            CenterNorm(embed_dim)
        )

        curr_dim = embed_dim
        dpr = [x.item() for x in torch.linspace(0, drop_path, np.sum(depth))]  # stochastic depth decay rule
        self.stage1 = nn.ModuleList([
            CSWinMLPLayer(
                dim=curr_dim, num_heads=heads[0], reso=img_size//4, mlp_ratio=mlp_ratio, d=d,
                qkv_bias=qkv_bias, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer, num_layers=depth[0])
            for i in range(depth[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        self.stage2 = nn.ModuleList(
            [CSWinMLPLayer(
                dim=curr_dim, num_heads=heads[1], reso=img_size//8, mlp_ratio=mlp_ratio, d=d,
                qkv_bias=qkv_bias, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1])+i], norm_layer=norm_layer, num_layers=depth[1])
            for i in range(depth[1])])
        
        self.merge2 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinMLPLayer(
                dim=curr_dim, num_heads=heads[2], reso=img_size//16, mlp_ratio=mlp_ratio, d=d,
                qkv_bias=qkv_bias, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2])+i], norm_layer=norm_layer, num_layers=depth[2])
            for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)
        
        self.merge3 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim*2
        self.stage4 = nn.ModuleList(
            [CSWinMLPLayer(
                dim=curr_dim, num_heads=heads[3], reso=img_size//32, mlp_ratio=mlp_ratio, d=d,
                qkv_bias=qkv_bias, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1])+i], norm_layer=norm_layer, last_stage=True, num_layers=depth[-1])
            for i in range(depth[-1])])
       
        self.norm = norm_layer(curr_dim)
        # Classifier head
        self.head = nn.Linear(curr_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.head.weight, std=0.02)
        self.apply(self._spectral_init)

    def _spectral_init(self, m):
        if isinstance(m, nn.Linear):
            # torch.nn.init.orthogonal_(m.weight, gain=1)
            torch.nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            u, s, v = torch.svd(m.weight)
            m.weight.data = m.weight.data / s[0]

        elif isinstance(m, (nn.Conv2d)):
            # torch.nn.init.orthogonal_(m.weight, gain=1)
            torch.nn.init.xavier_normal_(m.weight)
            weight = torch.reshape(m.weight.data, (m.weight.data.shape[0], -1))
            u, s, v = torch.svd(weight)
            m.weight.data = m.weight.data / s[0]

        elif isinstance(m, (nn.LayerNorm, CenterNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable

    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes, global_pool=''):
        if self.num_classes != num_classes:
            print ('reset head to', num_classes)
            self.num_classes = num_classes
            self.head = nn.Linear(self.out_dim, num_classes) if num_classes > 0 else nn.Identity()
            self.head = self.head.cuda()
            trunc_normal_(self.head.weight, std=.02)
            if self.head.bias is not None:
                nn.init.constant_(self.head.bias, 0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.stage1_conv_embed(x)
        for blk in self.stage1:
            if self.use_chk:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        for pre, blocks in zip([self.merge1, self.merge2, self.merge3], 
                               [self.stage2, self.stage3, self.stage4]):
            x = pre(x)
            for blk in blocks:
                if self.use_chk:
                    x = checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
        x = self.norm(x)
        return torch.mean(x, dim=1)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

### 224 models

@register_model
def cswinmlp_tiny_224(pretrained=False, **kwargs):
    model = CSWinMLPTransformer(patch_size=4, embed_dim=64, depth=[2,2,6,2], d=2,
        split_size=[1,2,7,7], num_heads=[2,4,8,16], mlp_ratio=4.)
    model.default_cfg = default_cfgs['cswinmlp_224']
    return model

@register_model
def cswinmlp_small_224(pretrained=False, **kwargs):
    model = CSWinMLPTransformer(patch_size=4, embed_dim=64, depth=[2,4,8,2], d=2,
        split_size=[1,2,7,7], num_heads=[2,4,8,16], mlp_ratio=4.)
    model.default_cfg = default_cfgs['cswinmlp_224']
    return model

@register_model
def cswinmlp_base_224(pretrained=False, **kwargs):
    model = CSWinMLPTransformer(patch_size=4, embed_dim=96, depth=[2,4,8,2], d=4,
        split_size=[1,2,7,7], num_heads=[4,8,16,32], mlp_ratio=4.)
    model.default_cfg = default_cfgs['cswinmlp_224']
    return model

@register_model
def cswinmlp_large_224(pretrained=False, **kwargs):
    model = CSWinMLPTransformer(patch_size=4, embed_dim=144, depth=[2,4,10,2], d=8,
        split_size=[1,2,7,7], num_heads=[6,12,24,24], mlp_ratio=4.)
    model.default_cfg = default_cfgs['cswinmlp_224']
    return model

### 384 models

@register_model
def CSWinMLP_96_24322_base_384(pretrained=False, **kwargs):
    model = CSWinMLPTransformer(patch_size=4, embed_dim=96, depth=[2,4,32,2], d=8,
        split_size=[1,2,12,12], num_heads=[4,4,4,8], mlp_ratio=4.)
    model.default_cfg = default_cfgs['cswinmlp_384']
    return model

@register_model
def CSWinMLP_144_24322_large_384(pretrained=False, **kwargs):
    model = CSWinMLPTransformer(patch_size=4, embed_dim=144, depth=[2,4,32,2], d=8,
        split_size=[4,2,12,12], num_heads=[4,4,8,8], mlp_ratio=4.)
    model.default_cfg = default_cfgs['cswinmlp_384']
    return model
