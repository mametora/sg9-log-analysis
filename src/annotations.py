"""Optional, user-supplied annotations kept separate from analysis logic."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass(frozen=True)
class ReportedSymptom:
    log: str
    near_end_s: float = 15.0
    detail_tail_s: float = 30.0


@dataclass(frozen=True)
class InspectionWindow:
    log: str
    start_s: float
    end_s: float
    context_s: float = 0.0
    label: str = "Inspection window"


@dataclass(frozen=True)
class AnalysisAnnotations:
    reported_symptom: Optional[ReportedSymptom] = None
    inspection_windows: List[InspectionWindow] = field(default_factory=list)


def load_annotations(path: Optional[Path]) -> AnalysisAnnotations:
    """Read optional JSON annotations; return an empty configuration when omitted."""
    if path is None:
        return AnalysisAnnotations()
    payload = json.loads(Path(path).read_text(encoding="utf-8"))

    reported = None
    if payload.get("reported_symptom"):
        item = payload["reported_symptom"]
        reported = ReportedSymptom(
            log=str(item["log"]),
            near_end_s=float(item.get("near_end_s", 15.0)),
            detail_tail_s=float(item.get("detail_tail_s", 30.0)),
        )
        if reported.near_end_s <= 0 or reported.detail_tail_s <= 0:
            raise ValueError("reported_symptom time spans must be positive")

    windows = []
    for item in payload.get("inspection_windows", []):
        window = InspectionWindow(
            log=str(item["log"]),
            start_s=float(item["start_s"]),
            end_s=float(item["end_s"]),
            context_s=float(item.get("context_s", 0.0)),
            label=str(item.get("label", "Inspection window")),
        )
        if window.end_s <= window.start_s or window.context_s < 0:
            raise ValueError("inspection window must have end_s > start_s and context_s >= 0")
        windows.append(window)
    return AnalysisAnnotations(reported_symptom=reported, inspection_windows=windows)
