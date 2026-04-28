"""JND attenuation used by the final CASIAL checkpoint."""

from __future__ import annotations

import torch
import torch.nn as nn


class JND(nn.Module):
    """Just-noticeable-difference attenuation map.

    Inputs are RGB tensors in ``[0, 1]``. The module returns
    ``cover + heatmap * (watermarked - cover)``.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        groups = self.in_channels

        kernel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).unsqueeze(0).unsqueeze(0)
        kernel_y = torch.tensor(
            [[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]]
        ).unsqueeze(0).unsqueeze(0)
        kernel_lum = torch.tensor(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 2.0, 2.0, 2.0, 1.0],
                [1.0, 2.0, 0.0, 2.0, 1.0],
                [1.0, 2.0, 2.0, 2.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0],
            ]
        ).unsqueeze(0).unsqueeze(0)

        self.conv_x = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=groups)
        self.conv_y = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False, groups=groups)
        self.conv_lum = nn.Conv2d(3, 3, kernel_size=5, padding=2, bias=False, groups=groups)
        self.conv_x.weight = nn.Parameter(kernel_x.repeat(groups, 1, 1, 1), requires_grad=False)
        self.conv_y.weight = nn.Parameter(kernel_y.repeat(groups, 1, 1, 1), requires_grad=False)
        self.conv_lum.weight = nn.Parameter(kernel_lum.repeat(groups, 1, 1, 1), requires_grad=False)

    def jnd_la(self, x: torch.Tensor, alpha: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
        la = self.conv_lum(x) / 32.0
        low = la <= 127
        la = la.clone()
        la[low] = 17 * (1 - torch.sqrt(la[low] / 127 + eps))
        la[~low] = 3 / 128 * (la[~low] - 127) + 3
        return float(alpha) * la

    def jnd_cm(self, x: torch.Tensor, beta: float = 0.117) -> torch.Tensor:
        grad_x = self.conv_x(x)
        grad_y = self.conv_y(x)
        cm = torch.sqrt(grad_x**2 + grad_y**2)
        cm = 16 * cm**2.4 / (cm**2 + 26**2)
        return float(beta) * cm

    def heatmaps(self, imgs: torch.Tensor, clc: float = 0.3) -> torch.Tensor:
        imgs = imgs.clamp(0.0, 1.0) * 255.0
        if self.in_channels == 1:
            rgb = torch.tensor([0.299, 0.587, 0.114], device=imgs.device, dtype=imgs.dtype)
            imgs = rgb[0] * imgs[:, 0:1] + rgb[1] * imgs[:, 1:2] + rgb[2] * imgs[:, 2:3]
        la = self.jnd_la(imgs)
        cm = self.jnd_cm(imgs)
        hmaps = torch.clamp_min(la + cm - float(clc) * torch.minimum(la, cm), 0.0)
        if self.out_channels == 3 and self.in_channels == 1:
            hmaps = hmaps.repeat(1, 3, 1, 1)
        elif self.out_channels == 1 and self.in_channels == 3:
            hmaps = torch.sum(hmaps / 3.0, dim=1, keepdim=True)
        return hmaps / 255.0

    def forward(self, cover: torch.Tensor, watermarked: torch.Tensor) -> torch.Tensor:
        hmaps = self.heatmaps(cover)
        return cover + hmaps * (watermarked - cover)

