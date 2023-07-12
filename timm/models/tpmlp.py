import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.utils.checkpoint as checkpoint
import numpy as np
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import einsum
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        **kwargs
    }


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


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, overlap=False):
        super().__init__()
        if overlap:
            padding = (patch_size - 1) // 2
            stride = (patch_size + 1) // 2
        else:
            padding = 0
            stride = patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, padding=padding, stride=stride)

    def forward(self, x):
        x = self.proj(x)  # B, C, H, W
        return x


class Downsample(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, in_embed_dim, out_embed_dim, patch_size, overlap=False):
        super().__init__()
        if overlap:
            assert patch_size==2
            self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=3, padding=1, stride=2)
        else:
            self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)  # B, C, H, W
        x = x.permute(0, 2, 3, 1)
        return x
    
    
class MixingAttention(nn.Module):
    def __init__(self, dim, resolution, idx, num_heads=8, split_size=2, dim_out=None, d=2, d_i=32, attn_drop=0., proj_drop=0.):
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
        L = H_sp * W_sp
        self.H_sp = H_sp
        self.W_sp = W_sp
        self.x_windows = self.resolution // H_sp
        self.y_windows = self.resolution // W_sp

        self.proj_in = nn.Linear(dim, num_heads * d)
        self.full = nn.Linear(num_heads * L * d, L * d)
        self.proj_out = nn.Linear(num_heads * d, dim)

    def forward(self, x):
        """
        x: B H W C
        """
        H_sp, W_sp = self.H_sp, self.W_sp
        x = self.proj_in(x)
        x = rearrange(x, "b (n1 h) (n2 w) (m d) -> (b n1 n2) m (h w d)", 
                      n1=self.x_windows, h=H_sp, n2=self.y_windows, w=W_sp, m=self.num_heads)
        w = rearrange(self.full.weight, "d2 (m d1) -> m d2 d1", m=self.num_heads)
        x = einsum("b m d, m f d -> b m f", x, w) + self.full.bias
        x = self.proj_out(rearrange(x, "(b n1 n2) m (h w d) -> b (n1 h) (n2 w) (m d)",
                                    n1=self.x_windows, h=H_sp, n2=self.y_windows, w=W_sp, m=self.num_heads))
        return x


class MixC(nn.Module):
    def __init__(self, resolution, dim, n=2):
        super().__init__()
        self.rearrange1 = Rearrange("b (h n1) (w n2) c -> (b h w) c (n1 n2)", n1=n, n2=n)
        self.weight = nn.Parameter(torch.randn(dim, n * n, n * n))
        self.bias = nn.Parameter(torch.zeros(dim, n * n))
        self.rearrange2 = Rearrange("(b h w) c (n1 n2) -> b (h n1) (w n2) c", h=resolution//n, w=resolution//n, n1=n, n2=n)
    def forward(self, x):
        x = self.rearrange1(x)
        x = torch.einsum("b m d, m d D -> b m D", x, self.weight) + self.bias
        return self.rearrange2(x)


class TpMLPBlock(nn.Module):
    def __init__(self, dim, resolution=32, num_heads=8, reduced_dim=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., split_size=2):
        super().__init__()
        self.resolution = resolution
        self.num_heads = num_heads
        self.mix_h = MixingAttention(dim, resolution // 2, idx=0, split_size=split_size, num_heads=self.num_heads, d=reduced_dim)
        self.mix_w = MixingAttention(dim, resolution // 2, idx=1, split_size=split_size, num_heads=self.num_heads, d=reduced_dim)
        self.mlp_c = MixC(resolution, dim)
        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.tp1 = Rearrange("b (h n1) (w n2) d -> (b n1 n2) h w d", n1=2, n2=2)
        self.tp2 = Rearrange("(b n1 n2) h w d -> b (h n1) (w n2) d", n1=2, n2=2)

    def forward(self, x):
        B, H, W, C = x.shape
        
        u = self.tp1(x)
        h = self.tp2(self.mix_h(u))
        w = self.tp2(self.mix_w(u))
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class VisionBlock(nn.Module):

    def __init__(self, dim, resolution, num_heads, reduced_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=TpMLPBlock, split_size=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, resolution=resolution, num_heads=num_heads, reduced_dim=reduced_dim, qkv_bias=qkv_bias, qk_scale=None,
                           attn_drop=attn_drop, split_size=split_size)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x))) / self.skip_lam
        x = x + self.drop_path(self.mlp(self.norm2(x))) / self.skip_lam
        return x


def basic_blocks(dim, index, layers, resolution, num_heads, reduced_dim, mlp_ratio=3., qkv_bias=False, qk_scale=None, \
                 attn_drop=0, drop_path_rate=0., skip_lam=1.0, mlp_fn=TpMLPBlock, split_size=2, **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(VisionBlock(dim, resolution, num_heads, reduced_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, \
                                  attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn, split_size=split_size))

    blocks = nn.Sequential(*blocks)

    return blocks


class VisionModel(nn.Module):

    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=None,
                 transitions=None, resolutions=None, num_heads=None, reduced_dims=None, mlp_ratios=None, skip_lam=1.0,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, mlp_fn=TpMLPBlock, overlap=False, split_size=2, **kwargs):

        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dims[0], overlap=overlap)

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, resolutions[i], num_heads[i], reduced_dims[i],
                                 mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate,
                                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn, split_size=split_size)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size, overlap=overlap))

        self.network = nn.ModuleList(network)

        self.norm = norm_layer(embed_dims[-1])

        # Classifier head
        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        # B,C,H,W-> B,H,W,C
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        return x

    def forward(self, x):
        x = self.forward_embeddings(x)
        # B, H, W, C -> B, N, C
        x = self.forward_tokens(x)
        x = self.norm(x)
        return self.head(x.mean(1))
    
    
default_cfgs = {
    'TpMLP_S': _cfg(crop_pct=0.9),
    'TpMLP_M': _cfg(crop_pct=0.9),
    'TpMLP_L': _cfg(crop_pct=0.875),
}


@register_model
def tpmlp_s(pretrained=False, **kwargs):
    layers = [4, 3, 8, 3]
    transitions = [True, False, False, False]
    resolutions = [32, 16, 16, 16]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [192, 384, 384, 384]
    reduced_dims = [4, 4, 4, 4]
    split_size = 4
    model = VisionModel(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, reduced_dims=reduced_dims, mlp_ratios=mlp_ratios,
                        mlp_fn=TpMLPBlock, split_size=split_size, **kwargs)
    model.default_cfg = default_cfgs['TpMLP_S']
    return model


@register_model
def tpmlp_m(pretrained=False, **kwargs):
    layers = [4, 3, 14, 3]
    transitions = [False, True, False, False]
    resolutions = [32, 32, 16, 16]
    num_heads = [8, 8, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [256, 256, 512, 512]
    reduced_dims = [4, 4, 4, 4]
    split_size = 4
    model = VisionModel(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, reduced_dims=reduced_dims, mlp_ratios=mlp_ratios,
                        mlp_fn=TpMLPBlock, split_size=split_size, **kwargs)
    model.default_cfg = default_cfgs['TpMLP_M']
    return model


@register_model
def tpmlp_l(pretrained=False, **kwargs):
    layers = [8, 8, 16, 4]
    transitions = [True, False, False, False]
    resolutions = [32, 16, 16, 16]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [256, 512, 512, 512]
    reduced_dims = [8, 8, 8, 8]
    split_size = 2
    model = VisionModel(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, reduced_dims=reduced_dims, mlp_ratios=mlp_ratios,
                        mlp_fn=TpMLPBlock, split_size=split_size, **kwargs)
    model.default_cfg = default_cfgs['TpMLP_L']
    return model