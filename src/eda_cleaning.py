#!/usr/bin/env python3
"""
Generate quick cleaning visuals for CW1 diamonds data.
Saves PNGs into ./figures.
"""

import os
from pathlib import Path

import matplotlib

# Ensure matplotlib cache is writable in this environment
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mplcache")
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent
TRAIN_PATH = DATA_DIR / "CW1_train.csv"
FIG_DIR = DATA_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_data():
    return pd.read_csv(TRAIN_PATH)


def clean(df: pd.DataFrame):
    mask_dim = (df["x"] > 0) & (df["y"] > 0) & (df["z"] > 0)
    mask_depth = (df["depth"] > 45) & (df["depth"] < 75)
    mask_table = (df["table"] > 40) & (df["table"] < 80)
    mask_xyz = (df["x"] < 30) & (df["y"] < 30) & (df["z"] < 30)
    return df, df[mask_dim & mask_depth & mask_table & mask_xyz]


def hist_before_after(ax, series_raw, series_clean, bins, title, xlabel):
    ax.hist(series_raw, bins=bins, alpha=0.45, label="raw", color="#d62728")
    ax.hist(series_clean, bins=bins, alpha=0.55, label="clean", color="#1f77b4")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()


def make_figures():
    raw = load_data()
    raw, clean_df = clean(raw)

    # Depth & table histograms
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    hist_before_after(
        axes[0], raw["depth"], clean_df["depth"], bins=np.arange(40, 82, 2), title="Depth", xlabel="depth"
    )
    hist_before_after(
        axes[1], raw["table"], clean_df["table"], bins=np.arange(40, 82, 2), title="Table", xlabel="table"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hist_depth_table.png", dpi=150)
    plt.close(fig)

    # x, y, z histograms
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    hist_before_after(
        axes[0], raw["x"], clean_df["x"], bins=np.linspace(0, 32, 32), title="X (length)", xlabel="x (mm)"
    )
    hist_before_after(
        axes[1], raw["y"], clean_df["y"], bins=np.linspace(0, 32, 32), title="Y (width)", xlabel="y (mm)"
    )
    hist_before_after(
        axes[2], raw["z"], clean_df["z"], bins=np.linspace(0, 32, 32), title="Z (depth)", xlabel="z (mm)"
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hist_xyz.png", dpi=150)
    plt.close(fig)

    # x vs y scatter highlighting removed points
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(raw["x"], raw["y"], s=5, alpha=0.25, label="raw", color="#d62728")
    ax.scatter(clean_df["x"], clean_df["y"], s=5, alpha=0.35, label="clean", color="#1f77b4")
    ax.set_xlim(0, 35)
    ax.set_ylim(0, 35)
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.set_title("x vs y before/after cleaning")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scatter_xy.png", dpi=150)
    plt.close(fig)

    # Outcome vs depth to show negative tail and trimming
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(raw["depth"], raw["outcome"], s=5, alpha=0.2, label="raw", color="#d62728")
    ax.scatter(clean_df["depth"], clean_df["outcome"], s=5, alpha=0.3, label="clean", color="#1f77b4")
    ax.set_xlabel("depth")
    ax.set_ylabel("outcome")
    ax.set_title("Outcome vs depth (raw vs clean)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "scatter_outcome_depth.png", dpi=150)
    plt.close(fig)

    print(f"Saved figures to {FIG_DIR}")


if __name__ == "__main__":
    make_figures()
