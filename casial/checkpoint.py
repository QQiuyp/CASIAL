"""Checkpoint helpers for CASIAL training."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import torch


def save_latest(path: Path, model, optimizer, epoch: int, global_step: int, config: Dict[str, Any], best: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "config": config,
        "best": best,
    }, path)


def save_weights(path: Path, model) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_latest(path: Path, model, optimizer=None) -> Dict[str, Any]:
    payload = torch.load(path, map_location="cpu")
    state = payload.get("model_state", payload)
    model.load_state_dict(state, strict=False)
    if optimizer is not None and "optimizer_state" in payload:
        optimizer.load_state_dict(payload["optimizer_state"])
    return payload

