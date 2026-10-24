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
    "knock_correction_advance": "#1f77b4",
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


def plot_reported_timing_detail(
    data: pd.DataFrame,
    log_stem: str,
    output_dir: Path,
    seconds: float = 30.0,
    highlight: Optional[tuple[float, float]] = None,
) -> None:
    """Create a compact RPM/MAF and timing-correction plot for sharing."""
    required = {
        "time_s",
        "rpm",
        "maf",
        "ignition_timing",
        "knock_correction_advance",
        "flkc",
        "fbkc",
        "knock_sum",
    }
    if data.empty or not required.issubset(data.columns):
        return

    end = float(data["time_s"].max())
    start = max(float(data["time_s"].min()), end - seconds)
    window = data[data["time_s"].between(start, end)]
    if window.empty:
        return

    channels = [
        ("rpm", "Engine speed (rpm)"),
        ("maf", "MAF (g/s)"),
        ("ignition_timing", "Total timing (deg)"),
        ("knock_correction_advance", "KCA (deg)"),
        ("flkc", "FLKC (deg)"),
        ("fbkc", "FBKC (deg)"),
        ("knock_sum", "Knock Sum"),
    ]
    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 12.5), sharex=True)

    for ax, (channel, label) in zip(axes, channels):
        if channel == "knock_sum":
            ax.step(
                window["time_s"],
                window[channel],
                where="post",
                color=COLORS[channel],
                linewidth=1.1,
            )
        else:
            ax.plot(
                window["time_s"],
                window[channel],
                color=COLORS[channel],
                linewidth=1.1,
            )
        ax.set_ylabel(label, color=COLORS[channel])
        if channel in {"knock_correction_advance", "flkc", "fbkc"}:
            ax.axhline(0, color="#666666", linewidth=0.6, alpha=0.45)

    for ax in axes:
        ax.grid(alpha=0.2)
        ax.set_xlim(start, end)
        if highlight is not None:
            shade_start = max(start, float(highlight[0]))
            shade_end = min(end, float(highlight[1]))
            if shade_start < shade_end:
                ax.axvspan(shade_start, shade_end, color="#ffbf69", alpha=0.18)

    axes[-1].set_xlabel("Time (s)")
    title = f"{log_stem}: reported symptom - timing and knock detail"
    if highlight is not None:
        title += " (shaded: strongest oscillation)"
    axes[0].set_title(title, loc="left")
    _finish(fig, output_dir / f"{log_stem}_end_{int(seconds)}s_timing_knock_detail.png")


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


def plot_high_load_boost(logs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Show whether boost rises beyond base-boost levels at high engine speed."""
    parts = []
    for name, data in logs.items():
        required = {"rpm", "throttle", "boost_rel"}
        if data.empty or not required.issubset(data.columns):
            continue
        part = data.loc[
            (data["throttle"] >= 50.0) & (data["rpm"] >= 2000.0) & (data["boost_rel"] > 0.0),
            ["rpm", "boost_rel"],
        ].copy()
        part["log"] = name
        parts.append(part)
    if not parts:
        return

    points = pd.concat(parts, ignore_index=True)
    bins = np.arange(2000.0, 6500.0 + 250.0, 250.0)
    points["rpm_bin"] = pd.cut(points["rpm"], bins=bins, right=False)
    envelope = points.groupby("rpm_bin", observed=True).agg(
        rpm=("rpm", "median"),
        boost_max=("boost_rel", "max"),
    )

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    ax.scatter(
        points["rpm"],
        points["boost_rel"],
        s=10,
        alpha=0.13,
        color="#4c78a8",
        edgecolors="none",
        label="Samples (throttle >= 50%)",
    )
    ax.plot(
        envelope["rpm"],
        envelope["boost_max"],
        color="#1f4e79",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="Maximum in each 250 rpm band",
    )

    overall_max = float(points["boost_rel"].max())
    ax.axhline(overall_max, color="#d62728", linestyle="--", linewidth=1.4)
    ax.text(
        6425,
        overall_max + 1.2,
        f"Overall maximum: {overall_max:.0f} kPa ({overall_max / 100:.2f} bar)",
        color="#b22222",
        ha="right",
        va="bottom",
        fontsize=10,
    )

    high_rpm = points[points["rpm"] >= 5000.0]
    if not high_rpm.empty:
        idx = high_rpm["boost_rel"].idxmax()
        x = float(points.loc[idx, "rpm"])
        y = float(points.loc[idx, "boost_rel"])
        ax.scatter([x], [y], s=55, color="#d62728", zorder=5)
        ax.annotate(
            f"5,000+ rpm max: {y:.0f} kPa",
            xy=(x, y),
            xytext=(-105, -42),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "#555555"},
            fontsize=10,
        )

    ax.set_xlim(2000, 6500)
    ax.set_ylim(0, max(70, overall_max + 8))
    ax.set_xlabel("Engine speed (rpm)")
    ax.set_ylabel("Manifold relative pressure (kPa)")
    ax.set_title("High-load boost vs engine speed across all logs")
    ax.grid(alpha=0.22)
    ax.legend(loc="lower right", fontsize=9)
    _finish(fig, output_dir / "high_load_boost_vs_rpm.png")


def plot_knock_correction_timeline(logs: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    """Align correction channels by elapsed time and mark Knock Sum increases."""
    usable = [
        (name, data)
        for name, data in logs.items()
        if not data.empty
        and "time_s" in data
        and any(channel in data for channel in ["fbkc", "flkc", "knock_correction_advance"])
    ]
    if not usable:
        return

    fig, axes = plt.subplots(len(usable), 1, figsize=(13, 2.0 * len(usable)), squeeze=False)
    axes = axes[:, 0]
    legend_handles = {}
    for ax, (name, data) in zip(axes, usable):
        for channel, label in [
            ("fbkc", "FBKC"),
            ("flkc", "FLKC"),
            ("knock_correction_advance", "KCA"),
        ]:
            if channel not in data or data[channel].dropna().empty:
                continue
            (line,) = ax.plot(
                data["time_s"],
                data[channel],
                color=COLORS[channel],
                linewidth=0.8,
                alpha=0.88,
                label=label,
            )
            legend_handles[label] = line

        if "knock_sum" in data:
            increases = data["knock_sum"].diff() > 0
            for x in data.loc[increases, "time_s"]:
                marker = ax.axvline(x, color="#d62728", linewidth=0.6, alpha=0.22, label="Knock Sum increase")
                legend_handles["Knock Sum increase"] = marker

        ax.axhline(0, color="#666666", linewidth=0.6, alpha=0.5)
        ax.set_ylabel(Path(name).stem.removeprefix("romraiderlog_").removeprefix("20260814_"), rotation=0, ha="right")
        ax.set_ylim(-20, 5)
        ax.grid(alpha=0.16)

    axes[0].set_title("All-log knock correction timeline", loc="left")
    axes[-1].set_xlabel("Elapsed time in each log (s)")
    fig.supylabel("Correction (degrees)", x=0.01)
    fig.legend(
        list(legend_handles.values()),
        list(legend_handles.keys()),
        loc="upper right",
        bbox_to_anchor=(0.985, 0.993),
        ncol=4,
        fontsize=9,
    )
    _finish(fig, output_dir / "all_logs_knock_correction_timeline.png")


def plot_knock_timing_transition_summary(summary: pd.DataFrame, output_dir: Path) -> None:
    """Compare rapid correction/timing transitions and Knock Sum increases by log."""
    if summary.empty:
        return
    labels = summary["log"].map(lambda name: Path(name).stem.removeprefix("romraiderlog_").removeprefix("20260814_"))
    y = np.arange(len(summary))
    fig, axes = plt.subplots(1, 2, figsize=(13, max(5.2, 0.65 * len(summary) + 1.8)), sharey=True)

    rates = [
        ("flkc_large_steps_per_min", "FLKC |step| >= 2 deg/min", "#8c564b"),
        ("kca_large_steps_per_min", "KCA |step| >= 2 deg/min", "#1f77b4"),
        ("ignition_large_steps_per_min", "Timing |step| >= 5 deg/min", "#17becf"),
    ]
    offsets = [-0.22, 0.0, 0.22]
    for (column, label, color), offset in zip(rates, offsets):
        if column in summary:
            axes[0].barh(y + offset, summary[column].fillna(0), height=0.20, color=color, alpha=0.86, label=label)
    axes[0].set_xlabel("Large adjacent-sample transitions per loaded minute")
    axes[0].set_yticks(y, labels)
    axes[0].grid(axis="x", alpha=0.2)
    axes[0].legend(fontsize=8, loc="lower right")

    knock_rate_raw = summary["knock_sum_increases_per_min"]
    knock_rate = knock_rate_raw.fillna(0)
    axes[1].barh(y, knock_rate, height=0.48, color="#d62728", alpha=0.78)
    for index, (rate, count, raw_rate) in enumerate(
        zip(knock_rate, summary["knock_sum_increases"].fillna(0), knock_rate_raw)
    ):
        label = "not logged" if pd.isna(raw_rate) else f"{rate:.1f}/min (n={int(count)})"
        axes[1].text(rate + 0.15, index, label, va="center", fontsize=9)
    axes[1].set_xlabel("Knock Sum increases per logged minute")
    axes[1].grid(axis="x", alpha=0.2)
    axes[1].tick_params(axis="y", labelleft=True)
    axes[1].set_yticks(y, labels)
    axes[1].set_xlim(0, max(5.0, float(knock_rate.max()) * 1.35))

    axes[0].invert_yaxis()
    fig.suptitle("Correction switching under load and logged knock counts", y=0.995)
    _finish(fig, output_dir / "knock_timing_transition_summary.png")


def plot_knock_timing_compact_summary(
    summary: pd.DataFrame,
    output_dir: Path,
    reported_log: Optional[str] = None,
) -> None:
    """Create a compact symptom-vs-reference summary suitable for sharing."""
    if summary.empty or reported_log is None:
        return
    symptom = summary[summary["log"] == reported_log]
    other = summary[summary["log"] != reported_log]
    if symptom.empty or other.empty:
        return
    symptom = symptom.iloc[0]

    metrics = [
        ("kca_large_steps_per_min", "KCA changes >= 2 deg", "#1f77b4"),
        ("flkc_large_steps_per_min", "FLKC changes >= 2 deg", "#8c564b"),
        ("knock_sum_increases_per_min", "Knock Sum increases", "#d62728"),
    ]
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), gridspec_kw={"height_ratios": [1.35, 1.0]})

    y = np.arange(len(metrics))
    for index, (column, _label, color) in enumerate(metrics):
        references = other[column].dropna()
        value = symptom.get(column, np.nan)
        if references.empty or pd.isna(value):
            continue
        low, median, high = float(references.min()), float(references.median()), float(references.max())
        axes[0].hlines(
            index,
            low,
            high,
            color="#a9a9a9",
            linewidth=7,
            alpha=0.55,
            zorder=1,
            label="Other-log range" if index == 0 else None,
        )
        axes[0].scatter([median], [index], s=65, color="#666666", zorder=3, label="Other-log median" if index == 0 else None)
        axes[0].scatter(
            [value],
            [index],
            s=90,
            marker="D",
            color=color,
            zorder=4,
            label="Reported symptom log" if index == 0 else None,
        )
        axes[0].text(value + 1.0, index, f"{value:.1f}/min", va="center", color=color, fontsize=10, weight="bold")

    axes[0].set_yticks(y, [label for _, label, _ in metrics])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Events per minute")
    axes[0].set_title("Reported symptom log compared with other logs", loc="left")
    axes[0].grid(axis="x", alpha=0.18)
    axes[0].legend(loc="lower right", fontsize=9)

    ranges = [
        ("kca", "KCA", "#1f77b4"),
        ("flkc", "FLKC", "#8c564b"),
        ("fbkc", "FBKC", "#d62728"),
    ]
    all_y = np.arange(len(ranges))
    for index, (prefix, _label, color) in enumerate(ranges):
        low = summary[f"{prefix}_min"].min(skipna=True)
        high = summary[f"{prefix}_max"].max(skipna=True)
        if pd.isna(low) or pd.isna(high):
            continue
        axes[1].hlines(index, low, high, color=color, linewidth=7, alpha=0.72)
        axes[1].scatter([low, high], [index, index], s=42, color=color, zorder=3)
        axes[1].text(low + 0.4, index, f"{low:+.1f}", ha="left", va="bottom", fontsize=10)
        axes[1].text(high - 0.4, index, f"{high:+.1f}", ha="right", va="bottom", fontsize=10)
    axes[1].axvline(0, color="#555555", linewidth=0.8, alpha=0.65)
    axes[1].set_yticks(all_y, [label for _, label, _ in ranges])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Correction range across all logs (degrees)")
    axes[1].set_title("Observed correction ranges", loc="left")
    axes[1].grid(axis="x", alpha=0.18)

    fig.suptitle("Knock and timing-correction summary", fontsize=16, weight="bold")
    fig.text(
        0.5,
        0.012,
        "Switching rate uses adjacent loaded samples: RPM >= 1,500 and throttle >= 30%.",
        ha="center",
        fontsize=9,
        color="#555555",
    )
    _finish(fig, output_dir / "knock_timing_compact_summary.png")


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
