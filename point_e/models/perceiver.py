#定义了一个简单的Perceiver模型，主要由多个交叉注意力块组成。
import math
from typing import Optional

import torch
import torch.nn as nn

from .checkpoint import checkpoint
from .transformer import MLP, init_linear

#__init__ 方法初始化交叉注意力层。
#forward 方法执行交叉注意力操作。首先对查询（q）和键值对（kv）进行线性变换，然后通过QKVMultiheadCrossAttention实现多头交叉注意力，最后应用线性变换
class MultiheadCrossAttention(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_data: int,
        width: int,
        heads: int,
        init_scale: float,
        data_width: Optional[int] = None,
    ):
        super().__init__()
        self.n_data = n_data
        self.width = width
        self.heads = heads
        self.data_width = width if data_width is None else data_width
        self.c_q = nn.Linear(width, width, device=device, dtype=dtype)
        self.c_kv = nn.Linear(self.data_width, width * 2, device=device, dtype=dtype)
        self.c_proj = nn.Linear(width, width, device=device, dtype=dtype)
        self.attention = QKVMultiheadCrossAttention(
            device=device, dtype=dtype, heads=heads, n_data=n_data
        )
        init_linear(self.c_q, init_scale)
        init_linear(self.c_kv, init_scale)
        init_linear(self.c_proj, init_scale)

    def forward(self, x, data):
        x = self.c_q(x)
        data = self.c_kv(data)
        x = checkpoint(self.attention, (x, data), (), True)
        x = self.c_proj(x)
        return x

#__init__ 方法初始化多头交叉注意力层。
#forward 方法执行多头交叉注意力操作。首先将查询（q）和键值对（kv）按照头数进行分割，然后计算注意力权重并应用到值（v）上。
class QKVMultiheadCrossAttention(nn.Module):
    def __init__(self, *, device: torch.device, dtype: torch.dtype, heads: int, n_data: int):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.heads = heads
        self.n_data = n_data

    def forward(self, q, kv):
        _, n_ctx, _ = q.shape
        bs, n_data, width = kv.shape
        attn_ch = width // self.heads // 2
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        q = q.view(bs, n_ctx, self.heads, -1)
        kv = kv.view(bs, n_data, self.heads, -1)
        k, v = torch.split(kv, attn_ch, dim=-1)
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)

#__init__ 方法初始化一个残差交叉注意力块。该块包含了一个交叉注意力层、LayerNorm层、MLP层以及另一个LayerNorm层。
#forward 方法执行残差块的前向传播操作，首先对输入应用交叉注意力层，然后应用LayerNorm层、MLP层、另一个LayerNorm层，并将结果与输入相加。
class ResidualCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_data: int,
        width: int,
        heads: int,
        data_width: Optional[int] = None,
        init_scale: float = 1.0,
    ):
        super().__init__()

        if data_width is None:
            data_width = width

        self.attn = MultiheadCrossAttention(
            device=device,
            dtype=dtype,
            n_data=n_data,
            width=width,
            heads=heads,
            data_width=data_width,
            init_scale=init_scale,
        )
        self.ln_1 = nn.LayerNorm(width, device=device, dtype=dtype)
        self.ln_2 = nn.LayerNorm(data_width, device=device, dtype=dtype)
        self.mlp = MLP(device=device, dtype=dtype, width=width, init_scale=init_scale)
        self.ln_3 = nn.LayerNorm(width, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_2(data))
        x = x + self.mlp(self.ln_3(x))
        return x

#__init__ 方法初始化一个简单的Perceiver模型，仅包含交叉注意力块。
#forward 方法执行Perceiver模型的前向传播，通过多个残差交叉注意力块对输入进行处理。
class SimplePerceiver(nn.Module):
    """
    Only does cross attention
    """

    def __init__(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
        n_data: int,
        width: int,
        layers: int,
        heads: int,
        init_scale: float = 0.25,
        data_width: Optional[int] = None,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualCrossAttentionBlock(
                    device=device,
                    dtype=dtype,
                    n_data=n_data,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                    data_width=data_width,
                )
                for _ in range(layers)
            ]
        )

    def forward(self, x: torch.Tensor, data: torch.Tensor):
        for block in self.resblocks:
            x = block(x, data)
        return x
