"""Noise layer registry used by CASIAL."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

from .color import Bright, Contrast, Hue, Saturation
from .combined import Combined
from .drop import Dropout
from .elastic import Elastic
from .erase import Erase
from .gaussian_filter import GF
from .gaussian_noise import GN
from .identity import Identity
from .jigsaw import Jigsaw
from .jpeg import JpegTest, KJpeg
from .middle_filter import MF
from .rc import RC
from .rotate import Rotate
from .salt_pepper_noise import SP
from .shear import Shear


def _normalize_key(name: str) -> str:
    return name.replace("_", "").replace("-", "").lower()


_NOISE_REGISTRY: Dict[str, Dict[str, Any]] = {}


def _register(display_name: str, cls, gpu_safe: bool, aliases: Iterable[str]) -> None:
    spec = {"display_name": display_name, "cls": cls, "gpu_safe": bool(gpu_safe)}
    for alias in {display_name, *aliases}:
        _NOISE_REGISTRY[_normalize_key(alias)] = spec


_register("Identity", Identity, True, aliases=["identity"])
_register("GN", GN, True, aliases=["gaussiannoise", "gaussian_noise", "gn"])
_register("MF", MF, True, aliases=["medianfilter", "middlefilter", "middle_filter", "mf"])
_register("GF", GF, True, aliases=["gaussianfilter", "gaussian_filter", "gf"])
_register("SP", SP, True, aliases=["saltpepper", "salt_pepper_noise", "sp"])
_register("Dropout", Dropout, True, aliases=["drop", "dropout"])
_register("Rotate", Rotate, True, aliases=["rotate", "rotation"])
_register("Shear", Shear, True, aliases=["shear"])
_register("RC", RC, True, aliases=["randomcrop", "cropresize", "rc"])
_register("Erase", Erase, True, aliases=["erase"])
_register("Elastic", Elastic, True, aliases=["elastic"])
_register("Jigsaw", Jigsaw, True, aliases=["jigsaw"])
_register("Combined", Combined, True, aliases=["combined"])
_register("Bright", Bright, True, aliases=["bright", "brightness"])
_register("Contrast", Contrast, True, aliases=["contrast"])
_register("Saturation", Saturation, True, aliases=["saturation", "sat"])
_register("Hue", Hue, True, aliases=["hue"])
_register("KJpeg", KJpeg, True, aliases=["kjpeg"])
_register("JpegTest", JpegTest, False, aliases=["jpegtest", "jpegtst"])


def get_noise_spec(name: str) -> Dict[str, Any]:
    key = _normalize_key(name)
    if key not in _NOISE_REGISTRY:
        raise KeyError(f"Unknown noise layer: {name}")
    return _NOISE_REGISTRY[key]


def get_noise_display_name(name: str) -> str:
    return get_noise_spec(name)["display_name"]


def _normalize_value(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_normalize_value(item) for item in value)
    if isinstance(value, dict):
        return {key: _normalize_value(item) for key, item in value.items()}
    return value


def normalize_noise_config_entry(config_entry: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(config_entry, dict):
        raise TypeError(f"Noise config must be a dict, got {type(config_entry)!r}")
    if "name" not in config_entry:
        raise KeyError(f"Noise config missing 'name': {config_entry}")

    normalized = {"name": _normalize_key(str(config_entry["name"]))}
    for key, value in config_entry.items():
        if key == "name":
            continue
        if key == "layers":
            normalized["layers"] = [normalize_noise_config_entry(item) for item in value]
        else:
            normalized[key] = _normalize_value(value)
    return normalized


def _instantiate_noise_entry(config_entry: Dict[str, Any], allow_unsafe: bool):
    name = get_noise_display_name(config_entry["name"])
    if name == "Combined":
        layers = [_instantiate_noise_entry(item, allow_unsafe=allow_unsafe) for item in config_entry["layers"]]
        module = Combined(layers=layers)
        module.is_gpu_safe = all(getattr(layer, "is_gpu_safe", True) for layer in layers)
        return module

    spec = get_noise_spec(name)
    kwargs = {key: value for key, value in config_entry.items() if key != "name"}
    module = spec["cls"](**kwargs)
    module.is_gpu_safe = bool(spec["gpu_safe"])
    if not allow_unsafe and not module.is_gpu_safe:
        raise ValueError(f"Noise layer {name} is not safe for this execution path.")
    return module


def instantiate_noise_layers(config_layers: List[Dict[str, Any]], allow_unsafe: bool = False):
    normalized = [normalize_noise_config_entry(item) for item in config_layers]
    modules = [_instantiate_noise_entry(item, allow_unsafe=allow_unsafe) for item in normalized]
    return modules, normalized


def _format_value(value: Any) -> str:
    if isinstance(value, tuple):
        return f"({','.join(_format_value(item) for item in value)})"
    return repr(value)


def noise_config_to_name(config_entry: Dict[str, Any]) -> str:
    normalized = normalize_noise_config_entry(config_entry)
    name = get_noise_display_name(normalized["name"])
    if name == "Combined":
        inner = ",".join(noise_config_to_name(item) for item in normalized["layers"])
        return f"Combined([{inner}])"

    kwargs = [(key, value) for key, value in normalized.items() if key != "name"]
    if not kwargs:
        return f"{name}()"
    args = ",".join(f"{key}={_format_value(value)}" for key, value in kwargs)
    return f"{name}({args})"


__all__ = [
    "instantiate_noise_layers",
    "noise_config_to_name",
    "normalize_noise_config_entry",
]
