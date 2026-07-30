#!/usr/bin/env python3
"""Render the reader-facing figures from the compact evidence JSON."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA = json.loads((ROOT / "results.json").read_text())
OUT = ROOT / "images"
OUT.mkdir(exist_ok=True)
NFES = np.array([1, 2, 4, 8])
COLORS = {"naive": "#4b5563", "euler": "#2563eb", "midpoint": "#dc2626"}


def series(group: dict, metric: str) -> np.ndarray:
    return np.array([group[str(n)][metric] for n in NFES])


def finish(name: str) -> None:
    plt.tight_layout()
    plt.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close()


def main() -> None:
    primary = DATA.get("primary", DATA["short_8000"])
    naive = DATA["naive"]

    plt.figure(figsize=(7.2, 4.3))
    for label, group in [
        ("Naive Euler", naive),
        ("PDD—Euler target", primary["euler"]),
        ("PDD—midpoint target", primary["midpoint"]),
    ]:
        key = "naive" if label.startswith("Naive") else label.split("—")[1].split()[0].lower()
        plt.plot(NFES, series(group, "kid_x1000"), "o-", lw=2.5, ms=6, label=label, color=COLORS[key])
    plt.xlabel("Sampling NFE (lower is faster)")
    plt.ylabel("KID × 1,000 (lower is better)")
    plt.xticks(NFES)
    plt.title("Few-step sample fidelity on 1,024 CIFAR-10 examples")
    plt.grid(alpha=0.22)
    plt.legend(frameon=False)
    finish("headline_kid.png")

    plt.figure(figsize=(7.2, 4.3))
    for label, group, key in [
        ("Naive Euler", naive, "naive"),
        ("PDD—Euler", primary["euler"], "euler"),
        ("PDD—midpoint", primary["midpoint"], "midpoint"),
    ]:
        plt.plot(NFES, series(group, "trajectory_mse"), "o-", lw=2.5, ms=6, label=label, color=COLORS[key])
    plt.yscale("log")
    plt.xlabel("Sampling NFE")
    plt.ylabel("MSE to 128-NFE teacher trajectory (log scale)")
    plt.xticks(NFES)
    plt.title("Paired trajectory preservation")
    plt.grid(alpha=0.22, which="both")
    plt.legend(frameon=False)
    finish("trajectory_error.png")

    plt.figure(figsize=(7.2, 4.3))
    for label, group, key in [
        ("Naive Euler", naive, "naive"),
        ("PDD—Euler", primary["euler"], "euler"),
        ("PDD—midpoint", primary["midpoint"], "midpoint"),
    ]:
        plt.plot(NFES, series(group, "diversity"), "o-", lw=2.5, ms=6, label=label, color=COLORS[key])
    plt.axhline(DATA["setup"]["reference"]["diversity"], color="#16a34a", ls="--", lw=2, label="128-NFE reference")
    plt.xlabel("Sampling NFE")
    plt.ylabel("Inception-feature diversity (higher is better)")
    plt.xticks(NFES)
    plt.title("Diversity retained across block sizes")
    plt.grid(alpha=0.22)
    plt.legend(frameon=False)
    finish("diversity.png")

    plt.figure(figsize=(7.2, 4.3))
    for label, group, key in [
        ("Naive Euler", naive, "naive"),
        ("PDD—Euler", primary["euler"], "euler"),
        ("PDD—midpoint", primary["midpoint"], "midpoint"),
    ]:
        plt.plot(series(group, "latency_ms"), series(group, "kid_x1000"), "o-", lw=2.5, ms=6, label=label, color=COLORS[key])
        for n, x, y in zip(NFES, series(group, "latency_ms"), series(group, "kid_x1000")):
            plt.annotate(str(n), (x, y), xytext=(4, 4), textcoords="offset points", fontsize=8)
    plt.xlabel("Measured latency, batch 64 (ms)")
    plt.ylabel("KID × 1,000 (lower is better)")
    plt.title("Quality–latency frontier; labels are NFEs")
    plt.grid(alpha=0.22)
    plt.legend(frameon=False)
    finish("latency_frontier.png")

    paper_e = DATA["paper"]["imagenet_1nfe_fid"]["euler"]
    paper_m = DATA["paper"]["imagenet_1nfe_fid"]["midpoint"]
    obs_e = primary["euler"]["1"]["kid_x1000"]
    obs_m = primary["midpoint"]["1"]["kid_x1000"]
    relative = [100 * (paper_m - paper_e) / paper_e, 100 * (obs_m - obs_e) / obs_e]
    plt.figure(figsize=(6.5, 4.2))
    bars = plt.bar(["Paper: ImageNet FID", "This run: CIFAR-10 KID"], relative, color=["#16a34a", "#dc2626"])
    plt.axhline(0, color="black", lw=1)
    plt.ylabel("Midpoint relative to Euler at 1 NFE (%)")
    plt.title("Target-method direction at matched budgets")
    plt.text(0.98, 0.95, "Below zero favors midpoint", transform=plt.gca().transAxes, ha="right", va="top", fontsize=9)
    for bar, val in zip(bars, relative):
        plt.text(bar.get_x() + bar.get_width() / 2, val + (0.08 if val >= 0 else -0.08), f"{val:+.2f}%", ha="center", va="bottom" if val >= 0 else "top")
    plt.grid(axis="y", alpha=0.22)
    finish("target_direction.png")


if __name__ == "__main__":
    main()
