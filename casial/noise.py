"""Final train/test noise definitions and adapters."""

from __future__ import annotations

from typing import Any, Dict, List

import torch
import torch.nn as nn

from network.noise_layers import instantiate_noise_layers, noise_config_to_name  # noqa: E402


def to_iclr_noise(entry: Dict[str, Any]) -> Dict[str, Any]:
    name = str(entry["name"]).lower()
    if name == "rc":
        return {
            "name": "rc",
            "min_crop_size": int(entry["min_crop_size"]),
            "max_crop_size": int(entry["max_crop_size"]),
            "output_size": list(entry.get("output_size", [128, 128])),
        }
    if name == "erase":
        return {"name": "erase", "scale": (float(entry["scale_min"]), float(entry["scale_max"]))}
    if name == "jigsaw":
        grid = int(entry["grid_min"])
        return {"name": "jigsaw", "grid": (grid, grid)}
    if name == "elastic":
        return {"name": "elastic", "alpha": (float(entry["alpha_min"]), float(entry["alpha_max"]))}
    if name == "rotate":
        return {
            "name": "rotate",
            "rotation_range": (float(entry["rotation_min"]), float(entry["rotation_max"])),
        }
    if name == "shear":
        return {
            "name": "shear",
            "shear_range": (float(entry["shear_min"]), float(entry["shear_max"])),
        }
    if name in {"hue", "bright", "contrast", "saturation"}:
        return {
            "name": name,
            "min_factor": float(entry["factor_min"]),
            "max_factor": float(entry["factor_max"]),
        }
    if name == "mf":
        return {"name": "mf", "kernel": int(entry["kernel_min"])}
    if name == "gf":
        return {"name": "gf", "sigma": float(entry["sigma_min"]), "kernel": int(entry.get("kernel", 7))}
    if name == "dropout":
        return {"name": "dropout", "prob": float(entry["prob_min"])}
    if name == "sp":
        return {"name": "sp", "amount": float(entry["amount_min"])}
    if name == "gn":
        return {"name": "gn", "std": float(entry["std_min"])}
    if name == "kjpeg":
        return {"name": "kjpeg", "Q": int(entry["Q_min"])}
    if name == "jpegtest":
        return {"name": "jpegtest", "Q": int(entry["Q_min"]), "path": str(entry.get("path", "tmp_noise/casial_jpegtest"))}
    raise KeyError(f"Unsupported CASIAL noise entry: {entry}")


def build_train_noise(train_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [{"name": "combined", "layers": [to_iclr_noise(item) for item in train_entries]}]


def instantiate(configs: List[Dict[str, Any]], device: torch.device, allow_unsafe: bool = True) -> nn.Module:
    modules, _ = instantiate_noise_layers(configs, allow_unsafe=allow_unsafe)
    if len(modules) != 1:
        return nn.Sequential(*modules).to(device)
    return modules[0].to(device)


def apply_noise(noise_module: nn.Module, watermarked: torch.Tensor, cover: torch.Tensor) -> torch.Tensor:
    out = noise_module((watermarked.float(), cover.float()))
    if isinstance(out, (tuple, list)):
        if len(out) != 1:
            raise RuntimeError(f"Expected one noise output, got {len(out)}")
        out = out[0]
    return out.clamp(-1.0, 1.0)


def display_name(config: Dict[str, Any]) -> str:
    return noise_config_to_name(config)
