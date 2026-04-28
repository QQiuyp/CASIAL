"""CASIAL encoder.

The encoder builds a cover-conditioned message feature with CAS and fuses it
with image features to generate the watermarked image.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ExFeature, MidExFeature, ReFeature


class Encoder(nn.Module):
    """Encode a binary message into a cover image."""

    def __init__(
        self,
        message_length: int = 64,
        message_branch_version: str = "casial",
        remove_spatial_attention: bool = False,
    ) -> None:
        super().__init__()
        if message_branch_version != "casial":
            raise ValueError("The public CASIAL encoder supports message_branch_version='casial'.")

        self.message_length = int(message_length)
        self.message_branch_version = "casial"
        self.remove_spatial_attention = bool(remove_spatial_attention)
        use_spatial_attention = not self.remove_spatial_attention

        self.mid_channel = 256
        self.ex_image = ExFeature(f_channels=self.mid_channel, use_spatial_attention=use_spatial_attention)
        self.fmix = MidExFeature(
            in_channels=self.mid_channel * 2,
            f_channels=self.mid_channel,
            use_spatial_attention=use_spatial_attention,
        )
        self.re_image = ReFeature(
            f_channels=self.mid_channel,
            out_channels=self.mid_channel,
            output_attr_name="conv_out",
            use_spatial_attention=use_spatial_attention,
        )
        self.final_im_mix = nn.Conv2d(self.mid_channel + 3, 3, 3, 1, 1, padding_mode="replicate")

        self.message0 = MidExFeature(
            in_channels=self.mid_channel,
            f_channels=self.mid_channel,
            use_spatial_attention=use_spatial_attention,
        )
        self.message1 = MidExFeature(
            in_channels=self.mid_channel,
            f_channels=self.mid_channel,
            use_spatial_attention=use_spatial_attention,
        )
        self.bit_mix_64 = MidExFeature(
            in_channels=self.mid_channel * self.message_length,
            f_channels=self.mid_channel,
            use_spatial_attention=use_spatial_attention,
        )

    @staticmethod
    def _flatten_selected_message(message: torch.Tensor, msg0: torch.Tensor, msg1: torch.Tensor) -> torch.Tensor:
        batch_size, message_length = message.shape
        mask = message.view(batch_size, message_length, 1, 1, 1).float()
        msg0 = msg0.unsqueeze(1).expand(-1, message_length, -1, -1, -1)
        msg1 = msg1.unsqueeze(1).expand(-1, message_length, -1, -1, -1)
        selected = (1.0 - mask) * msg0 + mask * msg1
        return selected.flatten(1, 2)

    def _build_message_features(self, image_features: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        detached_features = image_features.detach()
        msg0 = self.message0(detached_features)
        msg1 = self.message1(detached_features)
        selected = self._flatten_selected_message(message, msg0, msg1)
        return self.bit_mix_64(selected)

    def forward(self, image: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        image_features = self.ex_image(image)
        message_features = self._build_message_features(image_features, message)
        fused = self.fmix(torch.cat([image_features, message_features], dim=1))
        watermarked_features = self.re_image(fused)
        return self.final_im_mix(torch.cat([watermarked_features, image], dim=1))
