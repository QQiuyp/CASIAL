import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, prob=0.4):
        super().__init__()
        self.prob = float(prob)

    def forward(self, images_clean):
        noised_image, clean_image = images_clean
        keep_mask = (torch.rand_like(noised_image[:, :1, :, :]) < self.prob).to(noised_image.dtype)
        keep_mask = keep_mask.expand_as(noised_image)
        return noised_image * keep_mask + clean_image * (1 - keep_mask)
