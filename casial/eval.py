"""Evaluation helpers for the final table protocol."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from .metrics import bit_accuracy, bits_from_logits, deterministic_messages, iclr_psnr_per_sample
from .noise import apply_noise, instantiate, to_iclr_noise


def final_table_cases() -> List[Dict[str, Any]]:
    cases: List[Dict[str, Any]] = []

    def add(raw_id, paper_id, family, config):
        cases.append({"raw_id": raw_id, "paper_id": paper_id, "family": family, "config": config})

    add("JpegTest(Q=50)", "JpegTest(Q=50)", "JPEG", {"name": "jpegtest", "Q_min": 50, "Q_max": 50})
    add("MF(kernel=7)", "MF(kernel=7)", "MF", {"name": "mf", "kernel_min": 7, "kernel_max": 7})
    add("GF(sigma=2.0)", "GF(sigma=2.0)", "GF", {"name": "gf", "sigma_min": 2.0, "sigma_max": 2.0, "kernel": 7})
    add("Dropout(prob=0.5)", "Dropout(prob=0.5)", "Dropout", {"name": "dropout", "prob_min": 0.5, "prob_max": 0.5})
    add("SP(amount=0.1)", "SP(amount=0.1)", "SP", {"name": "sp", "amount_min": 0.1, "amount_max": 0.1})
    add("GN(std=0.04)", "GN(std=0.04)", "GN", {"name": "gn", "std_min": 0.04, "std_max": 0.04})
    add("Erase(scale=0.8)", "Erase(scale=0.8)", "Erase", {"name": "erase", "scale_min": 0.8, "scale_max": 0.8})
    add("RC(area=0.2,side=57)", "RC(area=0.2,side=57)", "RC", {"name": "rc", "min_crop_size": 57, "max_crop_size": 57, "output_size": [128, 128]})
    for val in (-60.0, 60.0):
        add(f"Shear({val:+g})", "Shear(abs=60)", "Shear", {"name": "shear", "shear_min": val, "shear_max": val})
    for val in (-45.0, 45.0):
        add(f"Rotate({val:+g})", "Rotate(abs=45)", "Rotate", {"name": "rotate", "rotation_min": val, "rotation_max": val})
    add("Elastic(alpha=2.0)", "Elastic(alpha=2.0)", "Elastic", {"name": "elastic", "alpha_min": 2.0, "alpha_max": 2.0})
    add("Jigsaw(grid=8)", "Jigsaw(grid=8)", "Jigsaw", {"name": "jigsaw", "grid_min": 8, "grid_max": 8})
    for val in (-0.1, 0.1):
        add(f"Hue({val:+g})", "Hue(abs=0.1)", "Hue", {"name": "hue", "factor_min": val, "factor_max": val})
    for name, family in (("bright", "Bright"), ("contrast", "Contrast"), ("saturation", "Saturation")):
        for val in (1.5, 0.2):
            add(f"{family}(factor={val})", f"{family}(avg=1.5,0.2)", family, {"name": name, "factor_min": val, "factor_max": val})
    return cases


def evaluate(model, loader, device: torch.device, message_length: int, seed: int, tmp_dir: str) -> tuple[list[dict], list[dict], dict]:
    model.eval()
    cases = final_table_cases()
    meters: Dict[str, Dict[str, float]] = defaultdict(lambda: {"correct": 0.0, "total": 0.0, "psnr": 0.0, "final_psnr": 0.0, "count": 0.0})
    clean = {"correct": 0.0, "total": 0.0, "psnr": 0.0, "count": 0.0}
    modules = [(case, instantiate([to_iclr_noise(case["config"])], device, allow_unsafe=True)) for case in cases]

    with torch.no_grad():
        for covers, indices, _paths in loader:
            covers = covers.to(device)
            bits = deterministic_messages(indices, message_length, seed, device)
            encoded = model.encode(covers, bits)
            clean_pred = bits_from_logits(model.decode(encoded))
            clean_acc, _ = bit_accuracy(clean_pred, bits)
            nbits = bits.numel()
            clean["correct"] += clean_acc * nbits
            clean["total"] += nbits
            clean["psnr"] += float(iclr_psnr_per_sample(encoded, covers).mean().item()) * covers.size(0)
            clean["count"] += covers.size(0)
            for case, module in modules:
                attacked = apply_noise(module, encoded, covers)
                pred = bits_from_logits(model.decode(attacked))
                acc, _ = bit_accuracy(pred, bits)
                key = case["raw_id"]
                meters[key]["correct"] += acc * nbits
                meters[key]["total"] += nbits
                meters[key]["psnr"] += float(iclr_psnr_per_sample(encoded, covers).mean().item()) * covers.size(0)
                meters[key]["final_psnr"] += float(iclr_psnr_per_sample(attacked, covers).mean().item()) * covers.size(0)
                meters[key]["count"] += covers.size(0)

    raw_rows = []
    for case in cases:
        m = meters[case["raw_id"]]
        acc = m["correct"] / max(1.0, m["total"])
        count = max(1.0, m["count"])
        raw_rows.append({
            "paper_id": case["paper_id"],
            "raw_id": case["raw_id"],
            "family": case["family"],
            "acc": acc,
            "bit_error": 1.0 - acc,
            "psnr": m["psnr"] / count,
            "final_psnr": m["final_psnr"] / count,
            "count": int(m["count"]),
        })

    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in raw_rows:
        grouped[row["paper_id"]].append(row)
    paper_rows = [{
        "paper_id": "Clean",
        "family": "Clean",
        "acc": clean["correct"] / max(1.0, clean["total"]),
        "bit_error": 1.0 - clean["correct"] / max(1.0, clean["total"]),
        "psnr": clean["psnr"] / max(1.0, clean["count"]),
        "final_psnr": clean["psnr"] / max(1.0, clean["count"]),
        "count": int(clean["count"]),
    }]
    for paper_id, rows in grouped.items():
        paper_rows.append({
            "paper_id": paper_id,
            "family": rows[0]["family"],
            "acc": sum(r["acc"] for r in rows) / len(rows),
            "bit_error": sum(r["bit_error"] for r in rows) / len(rows),
            "psnr": sum(r["psnr"] for r in rows) / len(rows),
            "final_psnr": sum(r["final_psnr"] for r in rows) / len(rows),
            "count": sum(r["count"] for r in rows),
        })
    agg = {
        "clean_acc": paper_rows[0]["acc"],
        "clean_psnr": paper_rows[0]["psnr"],
        "avg_acc": sum(r["acc"] for r in paper_rows[1:]) / max(1, len(paper_rows) - 1),
    }
    return raw_rows, paper_rows, agg


def write_csv(path: Path, rows: List[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def latex_row(method: str, paper_rows: List[dict]) -> str:
    by_id = {row["paper_id"]: row for row in paper_rows}
    order = [
        "JpegTest(Q=50)", "MF(kernel=7)", "GF(sigma=2.0)", "Dropout(prob=0.5)",
        "SP(amount=0.1)", "GN(std=0.04)", "Erase(scale=0.8)", "RC(area=0.2,side=57)",
        "Shear(abs=60)", "Rotate(abs=45)", "Elastic(alpha=2.0)", "Jigsaw(grid=8)",
        "Hue(abs=0.1)", "Bright(avg=1.5,0.2)", "Contrast(avg=1.5,0.2)",
        "Saturation(avg=1.5,0.2)",
    ]
    psnr = by_id["Clean"]["psnr"]
    vals = [by_id[item]["acc"] * 100.0 for item in order]
    avg = sum(vals) / len(vals)
    return f"{method} & {psnr:.2f} & " + " & ".join(f"{v:.2f}" for v in vals) + f" & {avg:.2f} \\\\"
