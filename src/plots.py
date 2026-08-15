"""Plotting helpers.  Every function is safe when optional channels are absent."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import os

# Keep Matplotlib/font caches inside the repository so the command works in a
# restricted or headless environment without trying to write to the home dir.
os.environ.setdefault("MPLCONFIGDIR", str(Path(".cache/matplotlib").resolve()))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(".cache").resolve()))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "rpm": "#1f77b4",
    "maf": "#d62728",
    "throttle": "#2ca02c",
    "boost_rel": "#9467bd",
    "lambda_1": "#ff7f0e",
    "ignition_timing": "#17becf",
    "fbkc": "#d62728",
    "flkc": "#8c564b",
    "knock_sum": "#111111",
}


def _finish(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _plot_dual(data: pd.DataFrame, left: str, right: str, title: str, path: Path, xlim=None) -> None:
    if left not in data or right not in data or data.empty:
        return
    fig, ax = plt.subplots(figsize=(13, 5))
    ax2 = ax.twinx()
    ax.plot(data["time_s"], data[left], color=COLORS.get(left), linewidth=1.0, label=left)
    ax2.plot(data["time_s"], data[right], color=COLORS.get(right), linewidth=1.0, alpha=0.85, label=right)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(left, color=COLORS.get(left))
    ax2.set_ylabel(right, color=COLORS.get(right))
    ax.grid(alpha=0.22)
    ax.set_title(title)
    if xlim:
        ax.set_xlim(*xlim)
    _finish(fig, path)


def plot_overview(data: pd.DataFrame, log_stem: str, output_dir: Path) -> None:
    _plot_dual(data, "rpm", "maf", f"{log_stem}: RPM and MAF", output_dir / f"{log_stem}_rpm_maf.png")
    _plot_dual(
        data,
        "throttle",
        "boost_rel",
        f"{log_stem}: throttle and manifold relative pressure",
        output_dir / f"{log_stem}_throttle_boost.png",
    )

    channels = [c for c in ["lambda_1", "ignition_timing", "fbkc", "flkc", "knock_sum"] if c in data]
    if not channels or data.empty:
        return
    fig, axes = plt.subplots(len(channels), 1, figsize=(13, 2.2 * len(channels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, channel in zip(axes, channels):
        ax.plot(data["time_s"], data[channel], color=COLORS.get(channel, "#333333"), linewidth=0.9)
        ax.set_ylabel(channel)
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"{log_stem}: fueling, ignition and knock channels")
    _finish(fig, output_dir / f"{log_stem}_fuel_ignition_knock.png")


def plot_end_detail(data: pd.DataFrame, log_stem: str, output_dir: Path, seconds: float = 30.0) -> None:
    if data.empty:
        return
    end = float(data["time_s"].max())
    start = max(float(data["time_s"].min()), end - seconds)
    window = data[data["time_s"].between(start, end)]
    _plot_dual(
        window,
        "rpm",
        "maf",
        f"{log_stem}: final {seconds:.0f} s — RPM and MAF",
        output_dir / f"{log_stem}_end_{int(seconds)}s_rpm_maf.png",
        (start, end),
    )
    _plot_dual(
        window,
        "throttle",
        "boost_rel",
        f"{log_stem}: final {seconds:.0f} s — throttle and boost",
        output_dir / f"{log_stem}_end_{int(seconds)}s_throttle_boost.png",
        (start, end),
    )
    channels = [c for c in ["rpm", "maf", "throttle", "boost_rel", "lambda_1", "ignition_timing", "fbkc", "flkc"] if c in window]
    if not channels:
        return
    fig, axes = plt.subplots(len(channels), 1, figsize=(13, 1.8 * len(channels)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, channel in zip(axes, channels):
        ax.plot(window["time_s"], window[channel], color=COLORS.get(channel, "#333333"), linewidth=1.1)
        ax.set_ylabel(channel)
        ax.grid(alpha=0.22)
    axes[-1].set_xlabel("Time (s)")
    axes[0].set_title(f"{log_stem}: detailed final {seconds:.0f} s")
    _finish(fig, output_dir / f"{log_stem}_end_{int(seconds)}s_detail.png")


def plot_maf_relationships(points: pd.DataFrame, output_dir: Path) -> None:
    """Compare normal and candidate acceleration samples in MAF relationships."""
    if points.empty or "maf" not in points:
        return
    relationships = [
        ("rpm", "MAF vs RPM", "Engine speed (rpm)"),
        ("throttle", "MAF vs throttle", "Throttle opening (%)"),
        ("boost_rel", "MAF vs manifold relative pressure", "Manifold relative pressure (kPa)"),
    ]
    for x, title, xlabel in relationships:
        if x not in points:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 6))
        normal = points[~points["is_suspicious"]]
        suspicious = points[points["is_suspicious"]]
        ax.scatter(normal[x], normal["maf"], s=5, alpha=0.10, color="#4c78a8", label="other acceleration")
        ax.scatter(suspicious[x], suspicious["maf"], s=9, alpha=0.38, color="#e45756", label="candidate events")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mass airflow (g/s)")
        ax.set_title(title)
        ax.grid(alpha=0.2)
        ax.legend()
        _finish(fig, output_dir / f"maf_vs_{x}.png")


def plot_event_comparison(
    event_rows: pd.DataFrame,
    logs: Dict[str, pd.DataFrame],
    output_dir: Path,
    channels: Sequence[str] = ("rpm", "maf", "boost_rel", "lambda_1", "ignition_timing", "fbkc", "flkc"),
) -> None:
    if event_rows.empty:
        return
    available = [c for c in channels if any(c in frame for frame in logs.values())]
    fig, axes = plt.subplots(len(available), 1, figsize=(12, 2.1 * len(available)), sharex=True)
    axes = np.atleast_1d(axes)
    for _, event in event_rows.iterrows():
        data = logs[event["log"]]
        segment = data[data["time_s"].between(event["start_s"], event["end_s"])]
        if segment.empty:
            continue
        progress = (segment["time_s"] - segment["time_s"].iloc[0]) / max(event["duration_s"], 0.01) * 100
        is_suspicious = bool(event["is_suspicious"])
        style = {"color": "#e45756" if is_suspicious else "#4c78a8", "alpha": 0.85, "linewidth": 1.4}
        label = ("candidate: " if is_suspicious else "comparison: ") + str(event["event_id"])
        for i, channel in enumerate(available):
            if channel in segment:
                axes[i].plot(progress, segment[channel], label=label if i == 0 else None, **style)
                axes[i].set_ylabel(channel)
                axes[i].grid(alpha=0.2)
    axes[0].legend(fontsize=7, ncol=2)
    axes[-1].set_xlabel("Event progress (%)")
    axes[0].set_title("Candidate acceleration vs matched comparison")
    _finish(fig, output_dir / "candidate_vs_normal.png")
