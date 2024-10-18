# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from collections import OrderedDict
import math
import requests
from io import BytesIO
from functools import partial
from PIL import Image
from typing import Callable, Optional, Sequence, Tuple, List
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from flash_attn import flash_attn_func

def reconstruct_matrix(windows):
    temp =[]
    for col in windows:
        temp.append(torch.cat((col),dim=3))
    all_img = torch.cat(temp,dim=2)
    return all_img


def sliding_window(matrix, window_size, stride):
    b,c,height, width = matrix.shape
    window_rows = (height - window_size[0]) // stride + 1
    window_cols = (width - window_size[1]) // stride + 1
    windows = []
    for i in range(window_rows):
        windows_col = []
        for j in range(window_cols):
            window = matrix[:,:, i*stride:i*stride+window_size[0],  j*stride:j*stride+window_size[1]]
            windows_col.append(window)
        windows.append(windows_col)
    return windows

def get_resized_pos_vit(abs_pos):
    if not hasattr(get_resized_pos_vit, "resized_pos"):
        get_resized_pos_vit.resized_pos = F.interpolate(
            abs_pos.float().reshape(1, 16, 16, -1).permute(0, 3, 1, 2),
            size=(32, 32),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=abs_pos.dtype)
    
    return get_resized_pos_vit.resized_pos


def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: M
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype

    if src_size != tgt_size:
        return F.interpolate(
            abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
            size=(tgt_size, tgt_size),
            mode="bicubic",
            align_corners=False,
        ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)
    else:
        return abs_pos

# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """
    def __init__(
            self,
            grid_size,
            embed_dim,
            num_heads,
            kv_dim=None,
            norm_layer=nn.LayerNorm
    ):
        super().__init__()
        self.num_queries = grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(embed_dim, grid_size)).float()
        ).requires_grad_(False)

        self.query = nn.Parameter(torch.zeros(self.num_queries, embed_dim))
        trunc_normal_(self.query, std=.02)

        if kv_dim is not None and kv_dim != embed_dim:
            self.kv_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        else:
            self.kv_proj = nn.Identity()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ln_q = norm_layer(embed_dim)
        self.ln_kv = norm_layer(embed_dim)
        
        # self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):

        pos_embed = get_abs_pos(self.pos_embed, x.size(1))

        x = self.kv_proj(x)
        x = self.ln_kv(x).permute(1, 0, 2)

        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(
            self._repeat(q, N) + self.pos_embed.unsqueeze(1),
            x + pos_embed.unsqueeze(1),
            x,
            attn_mask=attn_mask)[0]
        return out.permute(1, 0, 2)

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)



class Lora_Adapter(nn.Module):
    def __init__(self,
                 d_model=None,
                 out_feat=None,
                 r=16,
                 dropout=0.05):
        super().__init__()
        self.d_model = d_model
        self.out_feat = out_feat
        self.r = r

        self.lora_scale = nn.Parameter(torch.ones(1))


        self.lora_a = nn.Linear(self.d_model, self.r,bias=False)
        self.lora_b = nn.Linear(self.r, self.out_feat,bias=False)

        self.lora_dropout =  nn.Dropout(p=dropout)

        with torch.no_grad():
            nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_b.weight)

    def forward(self, x ):
        #residual = x if residual is None else residual

        x = self.lora_dropout(x)
        down = self.lora_a(x)
        up = self.lora_b(down)

        up = up * self.lora_scale
        output = up

        return output


class VisualAttention(nn.Module):
    """self-attention layer class.

    Self-attention layer takes input with size [s, b, h]
    and returns output of the same size.
    """

    def __init__(self, embed_dim, num_heads,
                 bias=True, kdim=None, vdim=None,lora_repeat_num=4):
        super(VisualAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads

        # Per attention head and per partition values.
        assert embed_dim % num_heads == 0
        self.head_size = embed_dim // num_heads
        self.num_heads = num_heads
        self.hidden_size_per_partition = embed_dim

        # Strided linear layer.
        assert self._qkv_same_embed_dim, 'Only Support SelfAttention Currently'
        self.in_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.in_proj_lora = []
        for _ in range(lora_repeat_num):
            self.in_proj_lora.append(Lora_Adapter(d_model=embed_dim,out_feat=3 * embed_dim))
        self.in_proj_lora = nn.ModuleList(self.in_proj_lora)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj_lora = []
        for _ in range(lora_repeat_num):
            self.out_proj_lora.append(Lora_Adapter(d_model=embed_dim,out_feat=embed_dim))
        self.out_proj_lora = nn.ModuleList(self.out_proj_lora)
        self.norm_factor = math.sqrt(self.head_size)

    def forward(self, query, key, value, attn_mask = None,idx = None):
        qkv = self.in_proj(query)  # (B, T, 3*C)
        qkv = qkv.unflatten(dim=2, sizes=(self.num_heads, 3*self.head_size))  # (B, T, nh, 3*hs)
        
        q, k, v = qkv.split(self.head_size, dim=-1)  # (B, T, nh, hs)
        attn_res = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)  # (B, T, nh, hs)
        attn_res = attn_res.flatten(2, 3)  # (B, T, C)

        # q, k, v = qkv.transpose(1, 2).split(self.head_size, dim=-1)  # (B, nh, T, hs)
        # attn_res = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)  # (B, nh, T, hs)
        # attn_res = attn_res.transpose(1, 2).flatten(2, 3)  # (B, T, C)

        output = self.out_proj(attn_res)
        return output


class VisualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            is_cross_attention: bool = False,
            lora_repeat_num = 4,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.attn = VisualAttention(d_model, n_head,lora_repeat_num = lora_repeat_num)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.mlp_lora = []
        for _ in range(lora_repeat_num):
            self.mlp_lora.append(Lora_Adapter(d_model=d_model,out_feat=d_model,r=32))
        self.mlp_lora = nn.ModuleList(self.mlp_lora)


    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            idx = None
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(q_x, k_x, v_x, attn_mask=attn_mask,idx=idx)

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
            idx = None
    ):
        # k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        # v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None

        x = q_x + self.attention(q_x=self.ln_1(q_x))
        # residual = x 
        x = x + self.mlp(self.ln_2(x))

        
        # if idx != None:
        #     x += self.mlp_lora[idx](residual)
        return x


class TransformerBlock(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = nn.LayerNorm,
            lora_repeat_num=4
    ):
        super().__init__()
        self.width = width
        self.layers = layers

        self.resblocks = nn.ModuleList([
            VisualAttentionBlock(
                width, heads, mlp_ratio, act_layer=act_layer, norm_layer=norm_layer,lora_repeat_num=lora_repeat_num)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def get_cast_device(self) -> torch.device:
        return self.resblocks[0].mlp.c_fc.weight.device

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,idx=None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask,idx=idx)
        return x


class VisionTransformer(nn.Module):

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            n_queries: int = 256,
            output_dim: int = 512,
            lora_repeat_num: int = 4,
            **kwargs
    ):
        super().__init__()
        image_height, image_width = self.image_size = (image_size, image_size)
        patch_height, patch_width = self.patch_size = (patch_size, patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.output_dim = output_dim

        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose([
            transforms.Resize(
                (image_size, image_size),
                interpolation=InterpolationMode.BICUBIC
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.positional_embedding = nn.Parameter(scale * torch.randn(256, width))

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        act_layer = nn.GELU

        self.ln_pre = norm_layer(width)
        self.transformer = TransformerBlock(
            width,
            layers,
            heads,
            mlp_ratio,
            act_layer=act_layer,
            norm_layer=norm_layer,
            lora_repeat_num=lora_repeat_num
        )

        self.attn_pool = Resampler(
            grid_size=int(math.sqrt(n_queries)),
            embed_dim=output_dim,
            num_heads=output_dim // 128,
            kv_dim=width,
            norm_layer=norm_layer,
        )
        self.ln_post = norm_layer(output_dim)
        self.proj = nn.Parameter((output_dim** -0.5) * torch.randn(output_dim, output_dim))

    def forward(self, x: torch.Tensor,idx=None):
        x = x.to(
            dtype=self.transformer.get_cast_dtype(),
            device=self.transformer.get_cast_device(),
        )
        with torch.no_grad():
            # to patches
            x = self.conv1(x)  # shape = [*, width, grid, grid]
            x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
            x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

            x = x + get_resized_pos_vit(self.positional_embedding)

            x = self.ln_pre(x)

            # x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.transformer(x,idx=idx)
            # x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.attn_pool(x)
        x = self.ln_post(x)
        x = x @ self.proj
        return x


if __name__ == "__main__":
    pass
    visual = VisionTransformer(
        image_size= 896,
        patch_size= 14,
        width=1664,
        layers = 48,
        heads= 16,
        mlp_ratio =  4.9231,
        output_dim= 4096)

    img = torch.randn(1,3,896,896)


    from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType

    # Define LoRA Config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["in_proj","out_proj","c_fc","c_proj"],
        lora_dropout=0.05,
        bias="none",    
    )
    # prepare int-8 model for training
    model = visual

    # add LoRA adaptor
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    print(model)
    print(visual)

