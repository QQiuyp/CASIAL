import random

import torch.nn as nn


class Combined(nn.Module):
    def __init__(self, layers=None):
        super().__init__()
        self.layers = nn.ModuleList(layers or [])
        self.last_index = None

    def forward(self, image_and_cover):
        if len(self.layers) == 0:
            raise RuntimeError("Combined noise layer requires at least one child layer.")
        index = random.randrange(len(self.layers))
        self.last_index = index
        return self.layers[index](image_and_cover)
