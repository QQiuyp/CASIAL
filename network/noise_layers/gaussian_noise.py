import numpy as np
import torch
import torch.nn as nn
import kornia.augmentation as K

# class GN(nn.Module):
#     def __init__(self, var=0.05, mean=0):
#         super(GN, self).__init__()
#         self.var = var
#         self.mean = mean
 
#     def gaussian_noise(self, image, mean, var):
#         noise = torch.Tensor(np.random.normal(mean, var ** 0.5, image.shape)).to(image.device)
#         out = image + noise
#         return out

#     def forward(self, images_clean):
#         image, clean_image = images_clean
#         no_image = self.gaussian_noise(image, self.mean, self.var).clip(-1,1)
#         return no_image

class GN(torch.nn.Module):
    """Gaussian Noise attack"""
    def __init__(self, std = 0.04):
        super().__init__()
        self.aug = K.RandomGaussianNoise(mean=0., std=std, p=1.)

    def forward(self, image_and_cover):
        image, cover_image = image_and_cover
        image = (image + 1) / 2
        image = self.aug(image)
        image = image * 2 - 1
        return image.clamp(-1, 1)


