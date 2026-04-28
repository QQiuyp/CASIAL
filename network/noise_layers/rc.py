import torch.nn as nn
import kornia.augmentation as K 
import torch
import kornia


class RC(nn.Module):
    def __init__(self, min_crop_size=13, max_crop_size=128, output_size=(128, 128)):
        super(RC, self).__init__()
        assert output_size[0] == output_size[1]
        self.min_crop_size = int(min_crop_size)
        self.max_crop_size = int(max_crop_size)
        self.output_size = tuple(output_size)
        self.resize = kornia.geometry.transform.Resize(self.output_size)

    def forward(self, image_and_cover):
        image, _ = image_and_cover         
        B, C, H, W = image.shape
        s = int(torch.randint(self.min_crop_size, self.max_crop_size + 1, (1,), device=image.device).item())
        s_eff = int(min(s, H, W))
        crop = K.RandomCrop(size=(s_eff, s_eff), p=1.0, keepdim=False, same_on_batch=False)
        cropped = crop(image)
        return self.resize(cropped)
