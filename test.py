#!/usr/bin/env python3
"""Evaluate CASIAL with the final-table protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from casial.data import FlatImageDataset, build_loader
from casial.eval import evaluate, latex_row, write_csv
from casial.model import CASIAL, load_model_state
from casial.noise import build_train_noise


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("CASIAL final-table evaluation")
    parser.add_argument("--config", default="configs/test.json")
    parser.add_argument("--checkpoint", default="checkpoints/300.pt")
    parser.add_argument("--output", default="outputs/final_table")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "effective_config.json").open("w") as f:
        json.dump({**config, "checkpoint": args.checkpoint}, f, indent=2)

    identity_noise = [{"name": "combined", "layers": [{"name": "identity"}]}]
    model = CASIAL(
        noise_layers=identity_noise,
        message_length=int(config["message_length"]),
        watermark_alpha=float(config["watermark_alpha"]),
        scaling_w=float(config["scaling_w"]),
        use_jnd=bool(config["jnd_1_1"]),
        device=device,
    ).to(device)
    report = load_model_state(model, args.checkpoint, strict=False)
    print(f"Loaded {args.checkpoint}: {report}")

    data_root = args.data_root or config["data_root"]
    dataset = FlatImageDataset(data_root, image_size=int(config["image_size"]))
    loader = build_loader(
        dataset,
        batch_size=int(config["batch_size"]),
        num_workers=int(config["num_workers"]),
        shuffle=False,
        max_images=int(args.max_images),
        seed=int(config.get("seed", 1234)),
    )
    raw_rows, paper_rows, agg = evaluate(
        model,
        loader,
        device=device,
        message_length=int(config["message_length"]),
        seed=int(config.get("seed", 1234)),
        tmp_dir=str(output / "tmp_noise"),
    )
    write_csv(output / "df_raw.csv", raw_rows)
    write_csv(output / "df_paper.csv", paper_rows)
    write_csv(output / "df_agg.csv", [agg])
    row = latex_row(str(config.get("method_name", "CASIAL")), paper_rows)
    (output / "latex_row.tex").write_text(row + "\n")
    with (output / "val_log.txt").open("w") as f:
        f.write(f"clean_acc={agg['clean_acc']:.6f},psnr={agg['clean_psnr']:.6f},avg_acc={agg['avg_acc']:.6f},\n")
        f.write(row + "\n")
    print(row)
    print(f"Saved outputs to {output}")


if __name__ == "__main__":
    main()
