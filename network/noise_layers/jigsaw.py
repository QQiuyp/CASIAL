import torch
import torch.nn as nn
import kornia.augmentation as K


class Jigsaw(torch.nn.Module):
    def __init__(self, grid=(64.0, 64.0)):
        super().__init__()

        self.aug = K.RandomJigsaw(grid, p = 1.0) 
        
        
    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        return self.aug(image)