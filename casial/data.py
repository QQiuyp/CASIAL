"""Dataset utilities for CASIAL scripts."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class FlatImageDataset(Dataset):
    def __init__(self, root: str, image_size: int = 128) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset path does not exist: {self.root}")
        self.files = [
            p for p in sorted(self.root.rglob("*"))
            if p.is_file() and p.suffix.lower() in IMG_EXTS
        ]
        if not self.files:
            raise RuntimeError(f"No images found under {self.root}")
        self.transform = transforms.Compose(
            [
                transforms.Resize((int(image_size), int(image_size)), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Lambda(lambda x: x * 2.0 - 1.0),
            ]
        )

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        image = Image.open(self.files[index]).convert("RGB")
        return self.transform(image), int(index), str(self.files[index])


def subset_dataset(dataset: Dataset, ratio: float, seed: int) -> Dataset:
    ratio = float(ratio)
    if ratio >= 1.0:
        return dataset
    count = max(1, int(len(dataset) * ratio))
    rng = random.Random(int(seed))
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    return Subset(dataset, indices[:count])


def build_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
    max_images: int = 0,
    seed: int = 1234,
    drop_last: bool = False,
) -> DataLoader:
    if int(max_images) > 0:
        dataset = Subset(dataset, list(range(min(int(max_images), len(dataset)))))
    gen = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(num_workers),
        pin_memory=True,
        drop_last=bool(drop_last),
        generator=gen,
    )
