"""Final CASIAL model wrapper."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .jnd import JND
from .metrics import m11_to_unit, unit_to_m11
from .noise import apply_noise, instantiate

from network.Decoder import Decoder  # noqa: E402
from network.Encoder import Encoder  # noqa: E402

FINAL_ENCODER_VERSION = "casial"
FINAL_DECODER_VERSION = "casial_decoder"


class CASIAL(nn.Module):
    """Final CASIAL encoder-noise-decoder with additive JND attenuation."""

    def __init__(
        self,
        noise_layers,
        message_length: int = 64,
        watermark_alpha: float = 1.0,
        scaling_w: float = 1.0,
        use_jnd: bool = True,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        self.message_length = int(message_length)
        self.watermark_alpha = float(watermark_alpha)
        self.register_buffer("videoseal_scaling_w", torch.tensor(float(scaling_w), dtype=torch.float32))
        self.encoder = Encoder(message_length=self.message_length, message_branch_version=FINAL_ENCODER_VERSION)
        self.decoder = Decoder(message_length=self.message_length, decoder_version=FINAL_DECODER_VERSION)
        self.noise = instantiate(noise_layers, device or torch.device("cpu"), allow_unsafe=True)
        self.attenuation = JND(in_channels=1, out_channels=1) if use_jnd else None

    @property
    def scaling_w(self) -> float:
        return float(self.videoseal_scaling_w.detach().float().item())

    def encode(self, cover_m11: torch.Tensor, bits: torch.Tensor) -> torch.Tensor:
        raw_m11 = self.encoder(cover_m11, bits.float()).clamp(-1.0, 1.0)
        if self.watermark_alpha != 1.0:
            raw_m11 = (cover_m11 + (raw_m11 - cover_m11) * self.watermark_alpha).clamp(-1.0, 1.0)
        cover_01 = m11_to_unit(cover_m11)
        raw_01 = m11_to_unit(raw_m11)
        blended_01 = (cover_01 + self.scaling_w * (raw_01 - cover_01)).clamp(0.0, 1.0)
        if self.attenuation is not None:
            blended_01 = self.attenuation(cover_01, blended_01).clamp(0.0, 1.0)
        return unit_to_m11(blended_01)

    def decode(self, attacked_m11: torch.Tensor) -> torch.Tensor:
        return self.decoder(attacked_m11)

    def forward(self, cover_m11: torch.Tensor, bits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encode(cover_m11, bits)
        noised = apply_noise(self.noise, encoded, cover_m11)
        decoded = self.decode(noised)
        return encoded, noised, decoded


def load_model_state(model: CASIAL, checkpoint: str, strict: bool = False) -> Dict[str, Any]:
    state = torch.load(checkpoint, map_location="cpu")
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    ignored_noise_keys = sorted(k for k in state if k.startswith("noise."))
    if ignored_noise_keys:
        state = {k: v for k, v in state.items() if not k.startswith("noise.")}
    result = model.load_state_dict(state, strict=strict)
    return {
        "missing": list(result.missing_keys),
        "unexpected": list(result.unexpected_keys),
        "ignored_noise_keys": ignored_noise_keys,
    }
