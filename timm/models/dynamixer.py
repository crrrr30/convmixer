import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
import torch.utils.checkpoint as checkpoint
import numpy as np
from einops import rearrange, einsum
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
allow_ops_in_compiled_graph()

try:
    from fvcore.nn.jit_handles import get_shape, conv_flop_count
except ImportError:
    has_fvcore = False


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
    
    
class DynaMixerOp(nn.Module):
    def __init__(self, dim, seq_len, num_head, reduced_dim=2):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.num_head = num_head
        self.reduced_dim = reduced_dim
        self.out = nn.Linear(dim, dim)
        self.compress = nn.Linear(dim, num_head * reduced_dim)
        self.generate = nn.Linear(seq_len * reduced_dim, seq_len * seq_len)
        self.activation = nn.Softmax(dim=-2)

    def forward(self, x):
        B, L, C = x.shape
        weights = self.compress(x).reshape(B, L, self.num_head, self.reduced_dim).permute(0, 2, 1, 3).reshape(B, self.num_head, -1)
        weights = self.generate(weights).reshape(B, self.num_head, L, L)
        weights = self.activation(weights)
        x = x.reshape(B, L, self.num_head, C//self.num_head).permute(0, 2, 3, 1)
        x = torch.matmul(x, weights)
        x = x.permute(0, 3, 1, 2).reshape(B, L, C)
        x = self.out(x)
        return x


class DynaMixerBlock(nn.Module):
    def __init__(self, dim, resolution=32, num_head=8, reduced_dim=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.resolution = resolution
        self.num_head = num_head
        self.mix_h = DynaMixerOp(dim, resolution, self.num_head, reduced_dim=reduced_dim)
        self.mix_w = DynaMixerOp(dim, resolution, self.num_head, reduced_dim=reduced_dim)
        self.mlp_c = nn.Linear(dim, dim, bias=qkv_bias)
        self.reweight = Mlp(dim, dim // 4, dim * 3)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, H, W, C = x.shape
        h = self.mix_h(x.permute(0, 2, 1, 3).reshape(-1, H, C)).reshape(B, W, H, C).permute(0, 2, 1, 3)
        w = self.mix_w(x.reshape(-1, W, C)).reshape(B, H, W, C)
        c = self.mlp_c(x)

        a = (h + w + c).permute(0, 3, 1, 2).flatten(2).mean(2)
        a = self.reweight(a).reshape(B, C, 3).permute(2, 0, 1).softmax(dim=0).unsqueeze(2).unsqueeze(2)

        x = h * a[0] + w * a[1] + c * a[2]

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class VisionBlock(nn.Module):

    def __init__(self, dim, resolution, num_head, reduced_dim, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, skip_lam=1.0, mlp_fn=DynaMixerBlock):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = mlp_fn(dim, resolution=resolution, num_head=num_head, reduced_dim=reduced_dim, qkv_bias=qkv_bias, qk_scale=None,
                           attn_drop=attn_drop)

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


def basic_blocks(dim, index, layers, resolution, num_head, reduced_dim, mlp_ratio=3., qkv_bias=False, qk_scale=None, \
                 attn_drop=0, drop_path_rate=0., skip_lam=1.0, mlp_fn=DynaMixerBlock, **kwargs):
    blocks = []

    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(VisionBlock(dim, resolution, num_head, reduced_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, \
                                  attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))

    blocks = nn.Sequential(*blocks)

    return blocks


class VisionModel(nn.Module):

    def __init__(self, layers, img_size=224, patch_size=4, in_chans=3, num_classes=1000, embed_dims=None,
                 transitions=None, resolutions=None, num_heads=None, reduced_dims=None, mlp_ratios=None, skip_lam=1.0,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, mlp_fn=DynaMixerBlock, overlap=False, **kwargs):

        super().__init__()
        self.num_classes = num_classes

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dims[0], overlap=overlap)

        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, resolutions[i], num_heads[i], reduced_dims[i],
                                 mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate,
                                 drop_path_rate=drop_path_rate, norm_layer=norm_layer, skip_lam=skip_lam, mlp_fn=mlp_fn)
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
    'DynaMixer_S': _cfg(crop_pct=0.9),
    'DynaMixer_M': _cfg(crop_pct=0.9),
    'DynaMixer_L': _cfg(crop_pct=0.875),
}


@register_model
def dynamixer_s(pretrained=False, **kwargs):
    layers = [4, 3, 8, 3]
    transitions = [True, False, False, False]
    resolutions = [32, 16, 16, 16]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [192, 384, 384, 384]
    reduced_dims = [2, 2, 2, 2]
    model = VisionModel(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, reduced_dims=reduced_dims, mlp_ratios=mlp_ratios,
                        mlp_fn=DynaMixerBlock, **kwargs)
    model.default_cfg = default_cfgs['DynaMixer_S']
    return model


@register_model
def dynamixer_m(pretrained=False, **kwargs):
    layers = [4, 3, 14, 3]
    transitions = [False, True, False, False]
    resolutions = [32, 32, 16, 16]
    num_heads = [8, 8, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [256, 256, 512, 512]
    reduced_dims = [2, 2, 2, 2]
    model = VisionModel(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, reduced_dims=reduced_dims, mlp_ratios=mlp_ratios,
                        mlp_fn=DynaMixerBlock, **kwargs)
    model.default_cfg = default_cfgs['DynaMixer_M']
    return model


@register_model
def dynamixer_l(pretrained=False, **kwargs):
    layers = [8, 8, 16, 4]
    transitions = [True, False, False, False]
    resolutions = [32, 16, 16, 16]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    embed_dims = [256, 512, 512, 512]
    reduced_dims = [8, 8, 8, 8]
    model = VisionModel(layers, embed_dims=embed_dims, patch_size=7, transitions=transitions,
                        resolutions=resolutions, num_heads=num_heads, reduced_dims=reduced_dims, mlp_ratios=mlp_ratios,
                        mlp_fn=DynaMixerBlock, **kwargs)
    model.default_cfg = default_cfgs['DynaMixer_L']
    return model