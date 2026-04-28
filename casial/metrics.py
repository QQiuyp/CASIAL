"""Metric helpers shared by CASIAL train/test entrypoints."""

from __future__ import annotations

import torch


def m11_to_unit(x: torch.Tensor) -> torch.Tensor:
    return ((x + 1.0) * 0.5).clamp(0.0, 1.0)


def unit_to_m11(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0.0, 1.0) * 2.0 - 1.0).clamp(-1.0, 1.0)


def iclr_psnr_per_sample(x_m11: torch.Tensor, y_m11: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mse = (x_m11.detach().float() - y_m11.detach().float()).flatten(1).pow(2).mean(dim=1)
    return 10.0 * torch.log10(torch.tensor(4.0, device=mse.device, dtype=mse.dtype) / mse.clamp_min(eps))


def bits_from_logits(logits: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    return (logits.detach().float() >= float(threshold)).to(torch.float32)


def bit_accuracy(pred_bits: torch.Tensor, target_bits: torch.Tensor) -> tuple[float, float]:
    pred = pred_bits.detach().float().round().clamp(0, 1)
    target = target_bits.detach().float().round().clamp(0, 1)
    if pred.shape != target.shape:
        pred = pred.reshape(target.shape)
    acc = float((pred == target).sum().item()) / float(max(1, target.numel()))
    return acc, 1.0 - acc


def deterministic_messages(indices: torch.Tensor, message_length: int, seed: int, device: torch.device) -> torch.Tensor:
    rows = []
    for index in indices.detach().cpu().tolist():
        gen = torch.Generator(device="cpu").manual_seed(int(seed) + int(index) * 1000003)
        rows.append(torch.randint(0, 2, (int(message_length),), generator=gen, dtype=torch.float32))
    return torch.stack(rows, dim=0).to(device)
