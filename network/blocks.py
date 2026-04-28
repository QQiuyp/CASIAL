"""Building blocks used by the CASIAL encoder and decoder."""

from __future__ import annotations

import torch
import torch.nn as nn
from einops import rearrange


def make_group_norm(num_channels: int, group: int = 32) -> nn.GroupNorm:
    num_groups = min(group, num_channels)
    while num_channels % num_groups != 0 and num_groups > 1:
        num_groups -= 1
    return nn.GroupNorm(num_groups, num_channels, eps=1e-6)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)


class Downsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 2, 1, padding_mode="replicate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, 4, 2, 1)
        self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1, padding_mode="replicate")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.upsample(x))


class ResnetBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, group: int = 32) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.norm1 = make_group_norm(in_channels, group)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode="replicate")
        self.norm2 = make_group_norm(out_channels, group)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode="replicate")
        self.act = Swish()
        if self.in_channels != self.out_channels:
            self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = self.conv2(self.act(self.norm2(h)))
        if self.in_channels != self.out_channels:
            x = self.nin_shortcut(x)
        return x + h


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels: int, group: int = 32) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.norm = make_group_norm(in_channels, group)
        self.q = nn.Conv2d(in_channels, in_channels, 1, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1, 1)
        self.proj_out = nn.Conv2d(in_channels, in_channels, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)

        _, c, height, _width = q.shape
        q = rearrange(q, "b c h w -> b (h w) c")
        k = rearrange(k, "b c h w -> b c (h w)")
        attn = torch.einsum("bij,bjk->bik", q, k) * (int(c) ** -0.5)
        attn = torch.softmax(attn, dim=2)

        v = rearrange(v, "b c h w -> b c (h w)")
        attn = rearrange(attn, "b i j -> b j i")
        h = torch.einsum("bij,bjk->bik", v, attn)
        h = rearrange(h, "b c (h w) -> b c h w", h=height)
        return x + self.proj_out(h)


class IdentitySpatialAttention(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


def make_spatial_attention(in_channels: int, group: int = 32, enabled: bool = True) -> nn.Module:
    if enabled:
        return SpatialSelfAttention(in_channels, group=group)
    return IdentitySpatialAttention()


class MidExFeature(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 96,
        f_channels: int = 8,
        group: int = 32,
        output_attr_name: str = "conv_out",
        use_spatial_attention: bool = True,
    ) -> None:
        super().__init__()
        self.output_attr_name = output_attr_name
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, 1, 1, padding_mode="replicate")
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(base_channels, base_channels, group=group)
        self.mid.attn_1 = make_spatial_attention(base_channels, group=group, enabled=use_spatial_attention)
        self.mid.block_2 = ResnetBlock(base_channels, base_channels, group=group)
        self.norm_out = make_group_norm(base_channels, group)
        setattr(
            self,
            output_attr_name,
            nn.Conv2d(base_channels, f_channels, 3, 1, 1, padding_mode="replicate"),
        )
        self.act = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.act(self.norm_out(h))
        return getattr(self, self.output_attr_name)(h)


class ExFeature(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 96,
        ch_mult=None,
        num_res_blocks: int = 2,
        f_channels: int = 64,
        group: int = 32,
        use_spatial_attention: bool = True,
    ) -> None:
        super().__init__()
        if ch_mult is None:
            ch_mult = [1, 2, 4]

        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = int(num_res_blocks)
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, 1, 1, padding_mode="replicate")

        in_ch_mult = [1] + list(ch_mult)
        self.down = nn.ModuleList()
        for level in range(self.num_resolutions):
            block = nn.ModuleList()
            block_in_channels = base_channels * in_ch_mult[level]
            block_out_channels = base_channels * ch_mult[level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in_channels, block_out_channels, group=group))
                block_in_channels = block_out_channels
            down = nn.Module()
            down.block = block
            if level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in_channels)
            self.down.append(down)

        block_in_channels = base_channels * ch_mult[-1]
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in_channels, block_in_channels, group=group)
        self.mid.attn_1 = make_spatial_attention(block_in_channels, group=group, enabled=use_spatial_attention)
        self.mid.block_2 = ResnetBlock(block_in_channels, block_in_channels, group=group)
        self.norm_out = make_group_norm(block_in_channels, group)
        self.conv_out = nn.Conv2d(block_in_channels, f_channels, 3, 1, 1, padding_mode="replicate")
        self.act = Swish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hs = [self.conv_in(x)]
        for level in range(self.num_resolutions):
            for block in self.down[level].block:
                hs.append(block(hs[-1]))
            if level != self.num_resolutions - 1:
                hs.append(self.down[level].downsample(hs[-1]))

        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)
        h = self.act(self.norm_out(h))
        return self.conv_out(h)


class ReFeature(nn.Module):
    def __init__(
        self,
        out_channels: int = 256,
        base_channels: int = 96,
        ch_mult=None,
        num_res_blocks: int = 2,
        f_channels: int = 4,
        group: int = 32,
        output_attr_name: str = "conv_out",
        use_spatial_attention: bool = True,
    ) -> None:
        super().__init__()
        if ch_mult is None:
            ch_mult = [1, 2, 4]

        self.output_attr_name = output_attr_name
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = int(num_res_blocks)
        block_in_channels = base_channels * ch_mult[-1]
        self.conv_in = nn.Conv2d(f_channels, block_in_channels, 3, 1, 1, padding_mode="replicate")

        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(block_in_channels, block_in_channels, group=group)
        self.mid.attn_1 = make_spatial_attention(block_in_channels, group=group, enabled=use_spatial_attention)
        self.mid.block_2 = ResnetBlock(block_in_channels, block_in_channels, group=group)

        self.up = nn.ModuleList()
        for level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            block_out_channels = base_channels * ch_mult[level]
            for _ in range(self.num_res_blocks):
                block.append(ResnetBlock(block_in_channels, block_out_channels, group=group))
                block_in_channels = block_out_channels
            up = nn.Module()
            up.block = block
            if level != 0:
                up.upsample = Upsample(block_in_channels)
            self.up.insert(0, up)

        self.norm_out = make_group_norm(block_in_channels, group)
        setattr(
            self,
            output_attr_name,
            nn.Conv2d(block_in_channels, out_channels, 3, 1, 1, padding_mode="replicate"),
        )
        self.act = Swish()

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        for level in reversed(range(self.num_resolutions)):
            for block in self.up[level].block:
                h = block(h)
            if level != 0:
                h = self.up[level].upsample(h)

        h = self.act(self.norm_out(h))
        return getattr(self, self.output_attr_name)(h)
