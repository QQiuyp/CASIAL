"""Rotation distortion."""

from __future__ import annotations

import torch.nn as nn
import kornia.augmentation as K


class Rotate(nn.Module):
    def __init__(self, rotation_range=(-180.0, 180.0)) -> None:
        super().__init__()
        self.affine = K.RandomAffine(degrees=rotation_range, shear=(0.0, 0.0), p=1.0)

    def forward(self, image_and_cover):
        image, _cover = image_and_cover
        image = (image + 1.0) / 2.0
        return self.affine(image) * 2.0 - 1.0
