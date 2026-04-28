#!/usr/bin/env python3
"""Train the final CASIAL framework."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from casial.checkpoint import load_latest, save_latest, save_weights
from casial.data import FlatImageDataset, build_loader, subset_dataset
from casial.eval import evaluate
from casial.metrics import bit_accuracy, bits_from_logits, deterministic_messages, iclr_psnr_per_sample
from casial.model import CASIAL, load_model_state
from casial.noise import build_train_noise


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CASIAL final training")
    parser.add_argument("--config", default="configs/train.json")
    parser.add_argument("--output", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--init-checkpoint", default="")
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config.get("seed", 1234)))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    output = Path(args.output or config.get("output_dir", "runs/casial_final")).resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "checkpoints").mkdir(exist_ok=True)
    with (output / "effective_config.json").open("w") as f:
        json.dump(config, f, indent=2)

    train_noise = build_train_noise(config["train_noise"])
    model = CASIAL(
        noise_layers=train_noise,
        message_length=int(config["message_length"]),
        watermark_alpha=float(config["watermark_alpha"]),
        scaling_w=float(config["scaling_w"]),
        use_jnd=bool(config["jnd_1_1"]),
        device=device,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(config["lr"]))

    start_epoch = 0
    global_step = 0
    best = {"avg_acc": -1.0, "psnr": -1.0, "epoch": -1}
    if args.resume:
        payload = load_latest(Path(args.resume), model, optimizer)
        start_epoch = int(payload.get("epoch", -1)) + 1
        global_step = int(payload.get("global_step", 0))
        best = dict(payload.get("best", best))
    elif args.init_checkpoint:
        report = load_model_state(model, args.init_checkpoint, strict=False)
        print(f"Initialized from {args.init_checkpoint}: {report}")

    train_root = Path(config["dataset_path"]) / "train"
    test_root = Path(config["dataset_path"]) / "test"
    train_set = subset_dataset(
        FlatImageDataset(str(train_root), image_size=int(config["image_size"])),
        float(config["train_subset_ratio"]),
        seed=int(config.get("seed", 1234)),
    )
    val_set = FlatImageDataset(str(test_root), image_size=int(config["image_size"]))
    train_loader = build_loader(
        train_set,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        shuffle=True,
        seed=int(config.get("seed", 1234)),
        drop_last=True,
    )
    val_loader = build_loader(
        val_set,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        shuffle=False,
        max_images=int(config.get("validation_max_images", 0)),
        seed=int(config.get("seed", 1234)),
    )

    train_log = output / "train_log.txt"
    val_log = output / "val_log.txt"
    epoch_number = int(config["epoch_number"])
    max_steps = int(args.max_steps)
    for epoch in range(start_epoch, epoch_number):
        model.train()
        t0 = time.time()
        acc_sum = psnr_sum = loss_sum = 0.0
        seen = 0
        for covers, indices, _paths in train_loader:
            covers = covers.to(device)
            bits = torch.randint(0, 2, (covers.size(0), int(config["message_length"])), device=device).float()
            encoded, _noised, decoded = model(covers, bits)
            msg_loss = F.mse_loss(decoded, bits)
            img_loss = F.mse_loss(encoded, covers)
            loss = float(config["encoder_weight"]) * img_loss + float(config["decoder_weight"]) * msg_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            pred = bits_from_logits(decoded)
            acc, _ = bit_accuracy(pred, bits)
            psnr = float(iclr_psnr_per_sample(encoded.detach(), covers).mean().item())
            batch = covers.size(0)
            acc_sum += acc * batch
            psnr_sum += psnr * batch
            loss_sum += float(loss.detach().item()) * batch
            seen += batch
            global_step += 1
            if max_steps and global_step >= max_steps:
                break

        train_line = (
            f"Epoch {epoch} (train): {time.time() - t0:.1f}s,"
            f"acc={acc_sum / max(1, seen):.6f},"
            f"psnr={psnr_sum / max(1, seen):.6f},"
            f"loss={loss_sum / max(1, seen):.6f},"
            f"global_step={global_step},\n"
        )
        print(train_line, end="")
        with train_log.open("a") as f:
            f.write(train_line)

        if (epoch + 1) % int(config["validate_every"]) == 0 or max_steps:
            raw_rows, paper_rows, agg = evaluate(
                model,
                val_loader,
                device=device,
                message_length=int(config["message_length"]),
                seed=int(config.get("seed", 1234)),
                tmp_dir=str(output / "tmp_noise"),
            )
            val_line = (
                f"Epoch {epoch} (val): clean_acc={agg['clean_acc']:.6f},"
                f"psnr={agg['clean_psnr']:.6f},avg_acc={agg['avg_acc']:.6f},\n"
            )
            print(val_line, end="")
            with val_log.open("a") as f:
                f.write(val_line)
            if (agg["avg_acc"], agg["clean_psnr"]) > (float(best["avg_acc"]), float(best["psnr"])):
                best = {"avg_acc": float(agg["avg_acc"]), "psnr": float(agg["clean_psnr"]), "epoch": int(epoch)}
                save_weights(output / "checkpoints" / "best.pt", model)

        save_latest(output / "checkpoints" / "latest.pt", model, optimizer, epoch, global_step, config, best)
        if max_steps and global_step >= max_steps:
            break

    save_weights(output / "checkpoints" / "final.pt", model)
    print(f"Saved final weights: {output / 'checkpoints' / 'final.pt'}")


if __name__ == "__main__":
    main()

