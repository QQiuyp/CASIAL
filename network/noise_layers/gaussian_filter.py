    
import torch
import torch.nn as nn
import kornia.augmentation as K

class GF(torch.nn.Module):
    def __init__(self, sigma=2.0, kernel=7):
        super().__init__()
        self.aug = K.RandomGaussianBlur((kernel, kernel), (sigma, sigma),  p=1.0)
        
    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        image = (image + 1) / 2
        image = self.aug(image)
        image = image * 2 - 1        
        return image.clamp(-1, 1)