"""CASIAL decoder."""

from __future__ import annotations

import torch
import torch.nn as nn

from .blocks import ExFeature, MidExFeature, ReFeature, Swish, make_group_norm


class Decoder(nn.Module):
    """Recover the embedded message from a watermarked or distorted image."""

    def __init__(
        self,
        message_length: int = 64,
        decoder_version: str = "casial_decoder",
        remove_spatial_attention: bool = False,
    ) -> None:
        super().__init__()
        if decoder_version != "casial_decoder":
            raise ValueError("The public CASIAL decoder supports decoder_version='casial_decoder'.")

        self.message_length = int(message_length)
        self.decoder_version = "casial_decoder"
        self.remove_spatial_attention = bool(remove_spatial_attention)
        use_spatial_attention = not self.remove_spatial_attention

        self.head_channels = 256
        self.ex_image = ExFeature(f_channels=self.head_channels, use_spatial_attention=use_spatial_attention)
        self.mid = MidExFeature(
            in_channels=self.head_channels,
            f_channels=self.head_channels,
            output_attr_name="conv_out30",
            use_spatial_attention=use_spatial_attention,
        )
        self.re_image = ReFeature(
            f_channels=self.head_channels,
            out_channels=self.head_channels,
            output_attr_name="conv_out_256",
            use_spatial_attention=use_spatial_attention,
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm_out_256 = make_group_norm(self.head_channels)
        self.act = Swish()
        self.message_layer_64 = nn.Linear(self.head_channels, self.message_length)

    def forward(self, image_with_wm: torch.Tensor) -> torch.Tensor:
        batch_size = image_with_wm.shape[0]
        x = self.ex_image(image_with_wm)
        x = self.mid(x)
        x = self.re_image(x)
        x = self.pool(x).view(batch_size, self.head_channels)
        x = self.act(self.norm_out_256(x))
        return self.message_layer_64(x)
