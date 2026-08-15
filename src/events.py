"""Acceleration-event extraction and exploratory hunting indicators."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


@dataclass(frozen=True)
class EventConfig:
    throttle_min: float = 15.0
    rpm_min: float = 1000.0
    min_duration_s: float = 1.5
    min_rpm_gain: float = 180.0
    bridge_gap_s: float = 0.35
    shift_drop_rpm_s: float = -900.0


def _samples(seconds: float, dt: float, minimum: int = 1) -> int:
    return max(minimum, int(round(seconds / max(dt, 0.01))))


def _interpolated(data: pd.DataFrame, column: str) -> pd.Series:
    if column not in data:
        return pd.Series(np.nan, index=data.index, dtype=float)
    return data[column].astype(float).interpolate(limit_direction="both")


def _bridge_short_false(mask: np.ndarray, max_gap: int) -> np.ndarray:
    out = mask.copy()
    false_starts = np.flatnonzero((~out) & np.r_[True, out[:-1]])
    for start in false_starts:
        end = start
        while end < len(out) and not out[end]:
            end += 1
        if start > 0 and end < len(out) and end - start <= max_gap:
            out[start:end] = True
    return out


def _true_runs(mask: np.ndarray) -> Iterable[Tuple[int, int]]:
    padded = np.r_[False, mask, False].astype(int)
    changes = np.diff(padded)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return zip(starts, ends)


def _oscillation_count(values: pd.Series, dt: float, floor_prominence: float) -> Tuple[int, float]:
    values = values.astype(float).interpolate(limit_direction="both")
    if len(values) < _samples(1.2, dt, 5) or values.isna().all():
        return 0, float("nan")
    smooth = values.rolling(_samples(0.20, dt), center=True, min_periods=1).mean()
    trend = smooth.rolling(_samples(1.5, dt), center=True, min_periods=1).mean()
    residual = (smooth - trend).to_numpy()
    prominence = max(floor_prominence, float(np.nanstd(residual)) * 0.55)
    distance = _samples(0.35, dt)
    peaks, _ = find_peaks(residual, prominence=prominence, distance=distance)
    troughs, _ = find_peaks(-residual, prominence=prominence, distance=distance)
    cycles = min(len(peaks), len(troughs))
    return int(cycles), float(np.nanstd(residual))


def extract_acceleration_bounds(data: pd.DataFrame, config: EventConfig = EventConfig()) -> List[Tuple[int, int]]:
    """Find sustained acceleration without restricting the RPM band."""
    if data.empty or not {"time_s", "rpm", "throttle"}.issubset(data.columns):
        return []
    time = _interpolated(data, "time_s").to_numpy()
    rpm = _interpolated(data, "rpm")
    throttle = _interpolated(data, "throttle")
    dt = float(np.nanmedian(np.diff(time))) if len(time) > 1 else 0.05
    rpm_smooth = rpm.rolling(_samples(0.25, dt), center=True, min_periods=1).mean()
    rpm_rate = np.gradient(rpm_smooth.to_numpy(), time)

    mask = ((throttle >= config.throttle_min) & (rpm >= config.rpm_min)).to_numpy()
    # Large negative RPM rates are normally shifts or clutch operation; split there.
    mask &= rpm_rate > config.shift_drop_rpm_s
    mask = _bridge_short_false(mask, _samples(config.bridge_gap_s, dt))

    bounds: List[Tuple[int, int]] = []
    for start, end in _true_runs(mask):
        duration = time[end] - time[start]
        segment = rpm_smooth.iloc[start : end + 1]
        gain = float(segment.iloc[-1] - segment.iloc[0])
        # Allow a local dip (the symptom of interest) but require overall acceleration.
        if duration >= config.min_duration_s and gain >= config.min_rpm_gain:
            bounds.append((start, end))
    return bounds


def _stat(segment: pd.DataFrame, column: str, operation: str) -> float:
    if column not in segment or segment[column].dropna().empty:
        return float("nan")
    series = segment[column]
    return float(getattr(series, operation)())


def summarize_event(data: pd.DataFrame, start: int, end: int, log_name: str, event_number: int) -> Dict[str, float]:
    segment = data.iloc[start : end + 1]
    time = segment["time_s"]
    dt = float(time.diff().median()) if len(segment) > 1 else 0.05
    rpm = _interpolated(segment, "rpm")
    throttle = _interpolated(segment, "throttle")
    rpm_rate = np.gradient(
        rpm.rolling(_samples(0.2, dt), center=True, min_periods=1).mean().to_numpy(),
        time.to_numpy(),
    )
    rpm_cycles, rpm_residual_std = _oscillation_count(rpm, dt, 25.0)
    maf_cycles, maf_residual_std = _oscillation_count(_interpolated(segment, "maf"), dt, 0.8)
    boost_cycles, boost_residual_std = _oscillation_count(_interpolated(segment, "boost_rel"), dt, 2.0)

    knock_delta = float("nan")
    if "knock_sum" in segment and segment["knock_sum"].notna().sum() >= 2:
        knock_delta = float(segment["knock_sum"].dropna().iloc[-1] - segment["knock_sum"].dropna().iloc[0])

    vvt_diff = float("nan")
    if {"vvt_left", "vvt_right"}.issubset(segment.columns):
        vvt_diff = float((segment["vvt_left"] - segment["vvt_right"]).abs().max())

    throttle_std = float(throttle.std())
    # This score ranks candidates; it is not a mechanical diagnosis.
    stability_factor = float(np.clip(1.5 - throttle_std / 12.0, 0.25, 1.5))
    hunting_score = stability_factor * (
        1.4 * rpm_cycles
        + 1.0 * maf_cycles
        + 0.8 * boost_cycles
        + min(3.0, max(0.0, rpm_residual_std - 20.0) / 35.0)
        + min(2.0, max(0.0, maf_residual_std - 0.5) / 3.0)
    )

    return {
        "log": log_name,
        "event_id": f"{log_name}:E{event_number:03d}",
        "start_s": float(time.iloc[0]),
        "end_s": float(time.iloc[-1]),
        "duration_s": float(time.iloc[-1] - time.iloc[0]),
        "start_row": int(start),
        "end_row": int(end),
        "rpm_start": float(rpm.iloc[0]),
        "rpm_end": float(rpm.iloc[-1]),
        "rpm_min": float(rpm.min()),
        "rpm_max": float(rpm.max()),
        "rpm_gain": float(rpm.iloc[-1] - rpm.iloc[0]),
        "rpm_rate_min": float(np.nanmin(rpm_rate)),
        "rpm_rate_max": float(np.nanmax(rpm_rate)),
        "rpm_oscillation_cycles": rpm_cycles,
        "rpm_residual_std": rpm_residual_std,
        "throttle_mean": float(throttle.mean()),
        "throttle_max": float(throttle.max()),
        "throttle_std": throttle_std,
        "boost_max_kpa": _stat(segment, "boost_rel", "max"),
        "boost_mean_kpa": _stat(segment, "boost_rel", "mean"),
        "boost_residual_std": boost_residual_std,
        "boost_oscillation_cycles": boost_cycles,
        "maf_max_g_s": _stat(segment, "maf", "max"),
        "maf_mean_g_s": _stat(segment, "maf", "mean"),
        "maf_std_g_s": _stat(segment, "maf", "std"),
        "maf_range_g_s": _stat(segment, "maf", "max") - _stat(segment, "maf", "min"),
        "maf_residual_std": maf_residual_std,
        "maf_oscillation_cycles": maf_cycles,
        "lambda_min": _stat(segment, "lambda_1", "min"),
        "lambda_median": _stat(segment, "lambda_1", "median"),
        "ignition_min_deg": _stat(segment, "ignition_timing", "min"),
        "ignition_median_deg": _stat(segment, "ignition_timing", "median"),
        "fbkc_min_deg": _stat(segment, "fbkc", "min"),
        "flkc_min_deg": _stat(segment, "flkc", "min"),
        "iam_median": _stat(segment, "iam", "median"),
        "knock_sum_delta": knock_delta,
        "vvt_abs_diff_max_deg": vvt_diff,
        "hunting_score": float(hunting_score),
    }


def extract_events(data: pd.DataFrame, log_name: str, config: EventConfig = EventConfig()) -> pd.DataFrame:
    rows = [
        summarize_event(data, start, end, log_name, number)
        for number, (start, end) in enumerate(extract_acceleration_bounds(data, config), start=1)
    ]
    return pd.DataFrame(rows)


def knock_change_windows(data: pd.DataFrame, seconds: float = 3.0) -> pd.DataFrame:
    """List Knock Sum changes and a ±seconds extraction window."""
    if data.empty or "knock_sum" not in data:
        return pd.DataFrame()
    delta = data["knock_sum"].diff()
    changed = data.loc[delta.fillna(0) != 0, ["time_s", "knock_sum"]].copy()
    changed["delta"] = delta.loc[changed.index]
    changed["window_start_s"] = (changed["time_s"] - seconds).clip(lower=float(data["time_s"].min()))
    changed["window_end_s"] = (changed["time_s"] + seconds).clip(upper=float(data["time_s"].max()))
    return changed.reset_index(drop=True)
