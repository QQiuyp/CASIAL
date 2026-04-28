"""Shear distortion."""

from __future__ import annotations

import torch.nn as nn
import kornia.augmentation as K


class Shear(nn.Module):
    def __init__(self, shear_range=(-60.0, 60.0)) -> None:
        super().__init__()
        self.affine = K.RandomAffine(degrees=(0.0, 0.0), shear=shear_range, p=1.0)

    def forward(self, image_and_cover):
        image, _cover = image_and_cover
        image = (image + 1.0) / 2.0
        return self.affine(image) * 2.0 - 1.0
