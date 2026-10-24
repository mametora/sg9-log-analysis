"""Source-preserving data, comparison time coordinates, and saved workspaces."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from uuid import uuid4

import numpy as np
from src.loader import LoadedLog, extract_unit, load_log

LABELS = {
    "rpm": "エンジン回転数", "throttle": "スロットル開度",
    "accelerator_pedal": "アクセル開度", "maf": "吸入空気量",
    "maf_voltage": "MAFセンサー電圧", "boost_rel": "相対過給圧",
    "map_abs": "絶対吸気圧", "lambda_1": "空燃比（λ）",
    "af_correction_1": "A/F補正", "af_learning_1": "A/F学習値",
    "ignition_timing": "点火時期", "fbkc": "フィードバックノック補正",
    "flkc": "学習ノック補正", "iam": "IAM", "knock_sum": "Knock Sum",
    "knock_correction_advance": "ノック補正進角", "vvt_left": "吸気VVT（左）",
    "vvt_right": "吸気VVT（右）", "wgdc": "WGDC（記録値）",
    "battery_voltage": "バッテリー電圧",
}
COLORS = ["#397bd4", "#cf792e", "#279882", "#a063c2", "#d25672", "#7e8b27"]
DEFAULT_COLORS = {"rpm": COLORS[0], "boost_rel": COLORS[1], "maf": COLORS[2],
                  "throttle": COLORS[3], "fbkc": COLORS[4], "flkc": COLORS[5]}
STAT_LABELS = {"max": "最大", "min": "最小", "mean": "平均", "median": "中央値"}


def finite(value: object) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("時刻や範囲には有限の数値を指定してください。")
    return result


def time_range(start: object, end: object) -> tuple[float, float]:
    start, end = finite(start), finite(end)
    if end <= start:
        raise ValueError("終了時刻は開始時刻より後にしてください。")
    return start, end


@dataclass
class Document:
    id: str
    path: Path
    name: str
    loaded: LoadedLog
    elapsed: np.ndarray
    columns: dict[str, str]
    units: dict[str, str]
    enabled: bool = True
    anchor: float = 0.0
    digest: str = ""

    @classmethod
    def open(cls, path: Path) -> Document:
        path = Path(path).resolve()
        loaded = load_log(path)
        t = loaded.data["time_s"].to_numpy(dtype=float, copy=True)
        valid = t[np.isfinite(t)]
        if len(valid) < 2 or np.max(valid) <= np.min(valid):
            raise ValueError("有効な時間列がありません。RomRaiderの Time (msec) 列を確認してください。")
        t -= valid[0]
        columns = dict(loaded.metadata.canonical_to_original)
        known = set(columns.values())
        for original in loaded.metadata.columns_original:
            if original not in known and original != "Time (msec)":
                columns[original] = original
        columns.pop("time_ms", None)
        units = {key: extract_unit(original) or "" for key, original in columns.items()}
        return cls(uuid4().hex, path, path.stem, loaded, t, columns, units,
                   digest=hashlib.sha256(path.read_bytes()).hexdigest())

    @property
    def x(self) -> np.ndarray:
        return self.elapsed - self.anchor

    def values(self, parameter: str) -> np.ndarray:
        return self.loaded.data[self.columns[parameter]].to_numpy(dtype=float)

    def mask(self, bounds: tuple[float, float] | None = None) -> np.ndarray:
        mask = np.isfinite(self.x)
        if bounds:
            epsilon = 1e-9 * max(1, abs(bounds[0]), abs(bounds[1]))
            mask &= (self.x >= bounds[0]-epsilon) & (self.x <= bounds[1]+epsilon)
        return mask

    def statistics(self, parameter: str, bounds: tuple[float, float] | None) -> dict:
        y = self.values(parameter)
        y = y[self.mask(bounds) & np.isfinite(y)]
        if not len(y):
            return {"count": 0, **{key: None for key in STAT_LABELS}}
        return {"count": len(y), "min": float(y.min()), "max": float(y.max()),
                "mean": float(y.mean()), "median": float(np.median(y))}

    def plot_arrays(self, parameter: str) -> tuple[np.ndarray, np.ndarray]:
        """Break at missing samples, clock resets and gaps; never sort or fill."""
        x, y = self.x, self.values(parameter)
        dt = np.diff(x)
        positive = dt[np.isfinite(dt) & (dt > 0)]
        period = float(np.median(positive)) if len(positive) else 0.0
        breaks = np.flatnonzero(~np.isfinite(dt) | (dt <= 0) | (dt > max(period * 5, 0.5))) + 1
        return np.insert(x, breaks, np.nan), np.insert(y, breaks, np.nan)

    def sample(self, parameter: str, x: float) -> tuple[float, float] | None:
        """Nearest recorded value, only within half a typical sampling interval."""
        distances = np.abs(self.x - x)
        if not np.isfinite(distances).any():
            return None
        i = int(np.nanargmin(distances))
        period = self.loaded.metadata.median_period_s
        tolerance = max(period * 0.6, 0.005) if math.isfinite(period) else 0.005
        value = self.values(parameter)[i]
        if distances[i] > tolerance or not np.isfinite(value):
            return None
        return float(self.elapsed[i]), float(value)

    def warnings(self) -> list[str]:
        m = self.loaded.metadata
        messages = []
        if m.header_segments > 1:
            messages.append(f"ヘッダ区間：{m.header_segments}（取得項目の変更に対応済み）")
        if m.malformed_rows:
            messages.append(f"形式不一致で除外した行：{m.malformed_rows}")
        if m.missing_values:
            messages.append(f"欠損値：{m.missing_values}")
        if m.non_numeric_columns:
            messages.append("数値に変換できない値：" + ", ".join(m.non_numeric_columns))
        resets = int(np.sum(np.diff(self.elapsed) <= 0))
        if resets:
            messages.append(f"時刻の重複・巻き戻り：{resets}（区間をまたぐ線は非表示）")
        if not np.isfinite(self.elapsed).all():
            messages.append("時刻不明の行はグラフ・区間出力から除外")
        return messages


@dataclass
class Panel:
    title: str = "グラフ"
    parameters: list[str] = field(default_factory=list)
    right: list[str] = field(default_factory=list)
    colors: dict[str, str] = field(default_factory=dict)
    statistics: dict[str, list[str]] = field(default_factory=dict)
    y_left: tuple[float, float] | None = None
    y_right: tuple[float, float] | None = None
    bounds: tuple[float, float] | None = None

    def color(self, parameter: str) -> str:
        if parameter not in self.colors:
            self.colors[parameter] = DEFAULT_COLORS.get(parameter, COLORS[int(hashlib.sha256(parameter.encode()).hexdigest()[:4], 16) % len(COLORS)])
        return self.colors[parameter]

    @classmethod
    def parse(cls, data: dict) -> Panel:
        panel = cls(title=str(data.get("title", "グラフ")),
                    parameters=list(dict.fromkeys(map(str, data.get("parameters", [])))),
                    right=list(map(str, data.get("right", []))),
                    colors={str(k): str(v) for k, v in data.get("colors", {}).items()},
                    statistics={str(k): [v for v in values if v in STAT_LABELS]
                                for k, values in data.get("statistics", {}).items()})
        for side in ("y_left", "y_right", "bounds"):
            if data.get(side) is not None:
                setattr(panel, side, time_range(*data[side]))
        if any(not re.fullmatch(r"#[0-9a-fA-F]{6}", color) for color in panel.colors.values()):
            raise ValueError("保存された線の色が不正です。")
        return panel


@dataclass
class Annotation:
    start: float
    end: float
    label: str
    note: str = ""


@dataclass
class Workspace:
    documents: list[Document] = field(default_factory=list)
    panels: list[Panel] = field(default_factory=lambda: [Panel("エンジン回転数", ["rpm"]),
                                                          Panel("吸気・過給", ["boost_rel", "maf"], ["maf"])])
    bounds: tuple[float, float] = (0.0, 10.0)
    columns: int = 1
    linked: bool = True
    cursors: list[float] = field(default_factory=lambda: [0.0, 1.0])
    annotations: list[Annotation] = field(default_factory=list)
    stat_all: bool = False
    active_panel: int = 0

    def extent(self) -> tuple[float, float]:
        times = [d.x[np.isfinite(d.x)] for d in self.documents if d.enabled]
        if not times:
            return 0.0, 10.0
        lo = min(float(t.min()) for t in times)
        hi = max(float(t.max()) for t in times)
        return lo, max(hi, lo + .001)

    def add(self, document: Document) -> bool:
        if any(d.path == document.path for d in self.documents):
            return False
        self.documents.append(document)
        return True

    def preset(self, name: str) -> dict:
        """Map statistics to log positions, so no file-specific ID leaks in presets."""
        panels = []
        ids = {d.id: str(i) for i, d in enumerate(self.documents)}
        for p in self.panels:
            entry = asdict(p)
            entry["bounds"] = None
            entry["statistics"] = {}
            for key, value in p.statistics.items():
                doc_id, sep, parameter = key.partition("|")
                if sep and doc_id in ids:
                    entry["statistics"][ids[doc_id] + "|" + parameter] = value
            panels.append(entry)
        return {"format": "romraider-preset", "version": 1, "name": name,
                "panels": panels, "columns": self.columns, "linked": self.linked,
                "stat_all": self.stat_all}

    def apply_preset(self, data: dict) -> list[str]:
        validate_format(data, "romraider-preset")
        panels = [Panel.parse(p) for p in data["panels"]]
        if not panels:
            raise ValueError("プリセットにグラフがありません。")
        for p in panels:
            mapped = {}
            for key, value in p.statistics.items():
                index, sep, parameter = key.partition("|")
                if sep and index.isdigit() and int(index) < len(self.documents):
                    mapped[self.documents[int(index)].id + "|" + parameter] = value
            p.statistics = mapped
        columns = min(2, max(1, int(data.get("columns", 1))))
        self.panels = panels
        self.columns = columns
        self.linked = bool(data.get("linked", True))
        self.stat_all = bool(data.get("stat_all", False))
        self.active_panel = 0
        return sorted({key for p in panels for key in p.parameters
                       if not any(key in d.columns for d in self.documents)})

    def session(self, destination: Path) -> dict:
        return {"format": "romraider-workspace", "version": 1,
                "documents": [{"id": d.id, "path": os.path.relpath(d.path, destination.resolve().parent),
                               "name": d.name, "enabled": d.enabled, "anchor": d.anchor,
                               "sha256": d.digest} for d in self.documents],
                "panels": [asdict(p) for p in self.panels], "bounds": self.bounds,
                "columns": self.columns, "linked": self.linked, "cursors": self.cursors,
                "annotations": [asdict(a) for a in self.annotations], "stat_all": self.stat_all,
                "active_panel": self.active_panel}

    @classmethod
    def restore(cls, data: dict, documents: list[Document]) -> Workspace:
        validate_format(data, "romraider-workspace")
        bounds = time_range(*data["bounds"])
        panels = [Panel.parse(p) for p in data["panels"]]
        if not panels:
            raise ValueError("分析状態にグラフがありません。")
        cursors = [finite(v) for v in data.get("cursors", [0, 1])]
        if len(cursors) != 2:
            raise ValueError("カーソル設定が不正です。")
        annotations = []
        for a in data.get("annotations", []):
            start, end = finite(a["start"]), finite(a["end"])
            if end < start:
                raise ValueError("注釈の時間範囲が不正です。")
            annotations.append(Annotation(start, end, str(a["label"]), str(a.get("note", ""))))
        return cls(documents, panels, bounds, min(2, max(1, int(data.get("columns", 1)))),
                   bool(data.get("linked", True)), cursors, annotations, bool(data.get("stat_all", False)),
                   max(0, min(len(panels)-1, int(data.get("active_panel", 0)))))


def validate_format(data: dict, expected: str) -> None:
    if not isinstance(data, dict) or data.get("format") != expected or data.get("version") != 1:
        raise ValueError("対応していない保存形式です。")


def protect_sources(destination: Path, documents: list[Document]) -> None:
    if destination.resolve() in {d.path.resolve() for d in documents}:
        raise ValueError("元ログへの上書きはできません。別の保存先を選択してください。")


def write_json(destination: Path, data: dict, documents: list[Document] = ()) -> None:
    destination = Path(destination)
    protect_sources(destination, documents)
    temporary = destination.with_name(destination.name + "." + uuid4().hex + ".tmp")
    try:
        temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def export_segment(document: Document, bounds: tuple[float, float], destination: Path,
                   documents: list[Document]) -> None:
    protect_sources(destination, documents)
    frame = document.loaded.data.loc[document.mask(bounds), document.loaded.metadata.columns_original].copy()
    name = "Comparison Time (sec)"
    while name in frame:
        name = "_" + name
    frame.insert(0, name, document.x[document.mask(bounds)])
    frame.to_csv(destination, index=False, encoding="utf-8-sig")
