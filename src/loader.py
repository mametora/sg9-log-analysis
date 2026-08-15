"""Robust loader for RomRaider CSV logs.

RomRaider can repeat a header when the selected parameter set changes while
logging.  A regular ``pandas.read_csv`` call then either fails or shifts data.
This module parses each header segment independently and joins the union of
columns without modifying the raw file.
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


CANONICAL_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"^Time\s*\(msec\)", "time_ms"),
    (r"^Engine Speed", "rpm"),
    (r"^Throttle Opening Angle", "throttle"),
    (r"^Accelerator Pedal", "accelerator_pedal"),
    (r"^Mass Airflow Sensor Voltage", "maf_voltage"),
    (r"^Mass Airflow", "maf"),
    (r"^Manifold Relative Pressure", "boost_rel"),
    (r"^Manifold Absolute Pressure", "map_abs"),
    (r"^A/F Sensor #1", "lambda_1"),
    (r"^A/F Correction #1", "af_correction_1"),
    (r"^A/F Learning #1", "af_learning_1"),
    (r"^Ignition Total Timing", "ignition_timing"),
    (r"^Knock Correction Advance", "knock_correction_advance"),
    (r"^Feedback Knock Correction", "fbkc"),
    (r"^Fine Learning Knock Correction", "flkc"),
    (r"^IAM", "iam"),
    (r"^Knock Sum", "knock_sum"),
    (r"^Intake VVT Advance Angle Left", "vvt_left"),
    (r"^Intake VVT Advance Angle Right", "vvt_right"),
    (r"^Primary Wastegate Duty Cycle", "wgdc"),
    (r"^Battery Voltage", "battery_voltage"),
)


@dataclass
class LogMetadata:
    path: Path
    columns_original: List[str]
    canonical_to_original: Dict[str, str]
    units: Dict[str, Optional[str]]
    header_segments: int
    malformed_rows: int
    row_count: int
    duration_s: float
    median_period_s: float
    mean_period_s: float
    p95_period_s: float
    missing_values: int
    non_numeric_columns: List[str]


@dataclass
class LoadedLog:
    data: pd.DataFrame
    metadata: LogMetadata


def canonical_name(column: str) -> Optional[str]:
    """Return a stable short name for a known RomRaider parameter."""
    for pattern, name in CANONICAL_PATTERNS:
        if re.search(pattern, column, flags=re.IGNORECASE):
            return name
    return None


def extract_unit(column: str) -> Optional[str]:
    """Extract the last parenthesized unit, ignoring logger footnote markers."""
    matches = re.findall(r"\(([^()]*)\)", column)
    return matches[-1].strip() if matches else None


def _is_header(row: Sequence[str]) -> bool:
    return bool(row) and row[0].strip().lower().startswith("time (msec)")


def _parse_dynamic_csv(path: Path) -> Tuple[pd.DataFrame, int, int]:
    records: List[Dict[str, str]] = []
    header: Optional[List[str]] = None
    header_segments = 0
    malformed_rows = 0

    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.reader(handle):
            if not row or not any(cell.strip() for cell in row):
                continue
            if _is_header(row):
                header = [cell.strip() for cell in row]
                header_segments += 1
                continue
            if header is None or len(row) != len(header):
                malformed_rows += 1
                continue
            records.append(dict(zip(header, (cell.strip() for cell in row))))

    return pd.DataFrame.from_records(records), header_segments, malformed_rows


def load_log(path: Path) -> LoadedLog:
    """Load one log and add canonical columns plus ``time_s``."""
    path = Path(path)
    raw, header_segments, malformed_rows = _parse_dynamic_csv(path)
    original_columns = list(raw.columns)
    canonical_to_original: Dict[str, str] = {}
    units: Dict[str, Optional[str]] = {}
    non_numeric: List[str] = []

    numeric = pd.DataFrame(index=raw.index)
    for column in original_columns:
        converted = pd.to_numeric(raw[column], errors="coerce")
        source_non_null = raw[column].replace("", np.nan).notna()
        if int(converted.notna().sum()) != int(source_non_null.sum()):
            non_numeric.append(column)
        numeric[column] = converted
        stable = canonical_name(column)
        if stable and stable not in numeric.columns:
            numeric[stable] = converted
            canonical_to_original[stable] = column
            units[stable] = extract_unit(column)

    if "time_ms" in numeric:
        numeric["time_s"] = numeric["time_ms"] / 1000.0
        dt = numeric["time_s"].diff()
        dt = dt[(dt > 0) & np.isfinite(dt)]
        duration = float(numeric["time_s"].iloc[-1] - numeric["time_s"].iloc[0]) if len(numeric) else 0.0
    else:
        numeric["time_s"] = np.nan
        dt = pd.Series(dtype=float)
        duration = 0.0

    metadata = LogMetadata(
        path=path,
        columns_original=original_columns,
        canonical_to_original=canonical_to_original,
        units=units,
        header_segments=header_segments,
        malformed_rows=malformed_rows,
        row_count=len(numeric),
        duration_s=duration,
        median_period_s=float(dt.median()) if len(dt) else float("nan"),
        mean_period_s=float(dt.mean()) if len(dt) else float("nan"),
        p95_period_s=float(dt.quantile(0.95)) if len(dt) else float("nan"),
        missing_values=int(numeric[original_columns].isna().sum().sum()) if original_columns else 0,
        non_numeric_columns=non_numeric,
    )
    return LoadedLog(numeric, metadata)


def load_logs(paths: Iterable[Path]) -> List[LoadedLog]:
    """Load all paths in deterministic filename order."""
    return [load_log(path) for path in sorted(map(Path, paths))]


def inventory_frame(logs: Sequence[LoadedLog]) -> pd.DataFrame:
    rows = []
    for log in logs:
        m = log.metadata
        rows.append(
            {
                "log": m.path.name,
                "rows": m.row_count,
                "duration_s": m.duration_s,
                "median_period_s": m.median_period_s,
                "mean_period_s": m.mean_period_s,
                "p95_period_s": m.p95_period_s,
                "header_segments": m.header_segments,
                "malformed_rows_skipped": m.malformed_rows,
                "missing_values": m.missing_values,
                "non_numeric_columns": "; ".join(m.non_numeric_columns),
                "available_parameters": "; ".join(sorted(m.canonical_to_original)),
            }
        )
    return pd.DataFrame(rows)


def statistics_frame(logs: Sequence[LoadedLog]) -> pd.DataFrame:
    """Return required descriptive statistics for every numeric source column."""
    rows = []
    for log in logs:
        for stable, original in log.metadata.canonical_to_original.items():
            series = log.data[stable].dropna()
            if series.empty:
                continue
            rows.append(
                {
                    "log": log.metadata.path.name,
                    "parameter": stable,
                    "original_column": original,
                    "unit": log.metadata.units.get(stable),
                    "count": int(series.count()),
                    "missing": int(log.data[stable].isna().sum()),
                    "min": float(series.min()),
                    "max": float(series.max()),
                    "mean": float(series.mean()),
                    "median": float(series.median()),
                    "std": float(series.std()),
                }
            )
    return pd.DataFrame(rows)
