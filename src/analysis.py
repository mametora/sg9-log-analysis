"""Command-line entry point for repeatable SG9 RomRaider log analysis."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.spatial import cKDTree

from .annotations import AnalysisAnnotations, ReportedSymptom, load_annotations
from .events import extract_events, knock_change_windows
from .loader import LoadedLog, inventory_frame, load_logs, statistics_frame
from .plots import (
    plot_end_detail,
    plot_event_comparison,
    plot_high_load_boost,
    plot_knock_correction_timeline,
    plot_knock_timing_compact_summary,
    plot_knock_timing_transition_summary,
    plot_maf_relationships,
    plot_overview,
    plot_reported_timing_detail,
)


def _knock_timing_summary(logs: Sequence[LoadedLog]) -> pd.DataFrame:
    """Summarize correction switching during load without treating it as knock."""
    rows = []
    for log in logs:
        data = log.data
        if data.empty or "time_s" not in data:
            continue
        dt = data["time_s"].diff()
        loaded = pd.Series(True, index=data.index)
        if "rpm" in data:
            loaded &= data["rpm"] >= 1500.0
        if "throttle" in data:
            loaded &= data["throttle"] >= 30.0
        adjacent_loaded = loaded & loaded.shift(fill_value=False) & dt.between(0.001, 0.2)
        loaded_minutes = float(dt.where(adjacent_loaded).sum() / 60.0)
        row = {
            "log": log.metadata.path.name,
            "loaded_minutes": loaded_minutes,
            "knock_sum_increases": 0,
            "knock_sum_positive_delta": 0.0,
        }
        if "knock_sum" in data:
            delta = data["knock_sum"].diff()
            knock_valid = data["knock_sum"].notna() & data["knock_sum"].shift().notna() & dt.between(0.001, 0.2)
            knock_minutes = float(dt.where(knock_valid).sum() / 60.0)
            positive = (delta > 0) & knock_valid
            row["knock_sum_increases"] = int(positive.sum())
            row["knock_sum_positive_delta"] = float(delta.where(positive, 0).sum())
            row["knock_logged_minutes"] = knock_minutes
            row["knock_sum_increases_per_min"] = int(positive.sum()) / knock_minutes if knock_minutes > 0 else np.nan
        else:
            row["knock_logged_minutes"] = np.nan
            row["knock_sum_increases_per_min"] = np.nan

        for channel, prefix, threshold in [
            ("fbkc", "fbkc", 2.0),
            ("flkc", "flkc", 2.0),
            ("knock_correction_advance", "kca", 2.0),
            ("ignition_timing", "ignition", 5.0),
        ]:
            if channel not in data or data[channel].dropna().empty:
                row[f"{prefix}_min"] = np.nan
                row[f"{prefix}_max"] = np.nan
                row[f"{prefix}_large_steps"] = np.nan
                row[f"{prefix}_large_steps_per_min"] = np.nan
                continue
            series = data[channel]
            valid = adjacent_loaded & series.notna() & series.shift().notna()
            step = series.diff().abs().where(valid)
            count = int((step >= threshold).sum())
            row[f"{prefix}_min"] = float(series.min())
            row[f"{prefix}_max"] = float(series.max())
            row[f"{prefix}_large_steps"] = count
            row[f"{prefix}_large_steps_per_min"] = count / loaded_minutes if loaded_minutes > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)

def _event_samples(logs: Sequence[LoadedLog], events: pd.DataFrame) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    wanted = ["time_s", "rpm", "throttle", "boost_rel", "maf", "lambda_1", "ignition_timing", "fbkc", "flkc"]
    for log in logs:
        own = events[events["log"] == log.metadata.path.name]
        for _, event in own.iterrows():
            available = [c for c in wanted if c in log.data]
            part = log.data.iloc[int(event["start_row"]) : int(event["end_row"]) + 1][available].copy()
            part["log"] = event["log"]
            part["event_id"] = event["event_id"]
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _maf_reference(samples: pd.DataFrame, events: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Estimate expected MAF from low-score accelerations at similar RPM/throttle/boost."""
    samples = samples.copy()
    required = ["rpm", "throttle", "boost_rel", "maf"]
    if samples.empty or not set(required).issubset(samples.columns):
        events["maf_outlier_fraction"] = np.nan
        events["maf_abs_z_p95"] = np.nan
        return samples, events

    density_by_id = events.set_index("event_id")["hunting_score_density"]
    reported_by_id = events.set_index("event_id")["user_reported_near_end"]
    sample_density = samples["event_id"].map(density_by_id)
    sample_reported = samples["event_id"].map(reported_by_id).fillna(False)
    complete = samples[required].notna().all(axis=1)
    reference = complete & ~sample_reported & (sample_density <= density_by_id.quantile(0.55))
    if reference.sum() < 100:
        reference = complete & ~sample_reported
    if reference.sum() < 30:
        events["maf_outlier_fraction"] = np.nan
        events["maf_abs_z_p95"] = np.nan
        return samples, events

    scale = np.array([500.0, 10.0, 20.0])
    ref_x = samples.loc[reference, ["rpm", "throttle", "boost_rel"]].to_numpy() / scale
    ref_y = samples.loc[reference, "maf"].to_numpy()
    tree = cKDTree(ref_x)
    eval_idx = samples.index[complete]
    eval_x = samples.loc[eval_idx, ["rpm", "throttle", "boost_rel"]].to_numpy() / scale
    k = min(41, len(ref_x))
    distance, neighbor_idx = tree.query(eval_x, k=k)
    if k == 1:
        neighbor_idx = neighbor_idx[:, None]
        distance = distance[:, None]
    neighbors = ref_y[neighbor_idx]
    expected = np.nanmedian(neighbors, axis=1)
    mad = np.nanmedian(np.abs(neighbors - expected[:, None]), axis=1)
    # A 1 g/s floor prevents huge z-scores where the reference is nearly constant.
    robust_sigma = np.maximum(1.0, 1.4826 * mad)
    residual = samples.loc[eval_idx, "maf"].to_numpy() - expected
    samples.loc[eval_idx, "maf_expected"] = expected
    samples.loc[eval_idx, "maf_residual"] = residual
    samples.loc[eval_idx, "maf_robust_z"] = residual / robust_sigma
    samples.loc[eval_idx, "reference_distance"] = np.asarray(distance)[:, -1]

    agg = samples.groupby("event_id")["maf_robust_z"].agg(
        maf_abs_z_p95=lambda s: float(s.abs().quantile(0.95)),
        maf_outlier_fraction=lambda s: float((s.abs() >= 3.0).mean()),
        maf_negative_outlier_fraction=lambda s: float((s <= -3.0).mean()),
    )
    events = events.merge(agg, left_on="event_id", right_index=True, how="left")
    return samples, events


def _classify_events(
    events: pd.DataFrame,
    log_durations: Dict[str, float],
    reported: Optional[ReportedSymptom] = None,
) -> pd.DataFrame:
    events = events.copy()
    if events.empty:
        return events
    events["hunting_score_density"] = events["hunting_score"] / events["duration_s"].clip(lower=1.0) * 10.0
    events["user_reported_near_end"] = False
    if reported is not None:
        events["user_reported_near_end"] = (
            (events["log"] == reported.log)
            & ((events["log"].map(log_durations) - events["end_s"]) <= reported.near_end_s)
        )
    return events


def _rank_events(events: pd.DataFrame) -> pd.DataFrame:
    events = events.copy()
    if events.empty:
        return events
    density = events["hunting_score_density"]
    density_z = (density - density.median()) / max(float((density - density.median()).abs().median()) * 1.4826, 0.5)
    outliers = events.get("maf_outlier_fraction", pd.Series(0.0, index=events.index)).fillna(0)
    negative_maf = events.get("maf_negative_outlier_fraction", pd.Series(0.0, index=events.index)).fillna(0)
    knock_flag = (events["fbkc_min_deg"].fillna(0) < -1.0).astype(float)
    events["candidate_score"] = density_z.clip(-3, 8) + 3.0 * outliers + 1.5 * negative_maf + 0.5 * knock_flag
    cutoff = float(events["candidate_score"].quantile(0.93))
    events["auto_candidate"] = events["candidate_score"] >= cutoff
    events["is_suspicious"] = events["auto_candidate"] | events["user_reported_near_end"]
    events["auto_score_rank"] = events["candidate_score"].rank(method="min", ascending=False).astype(int)
    events["candidate_kind"] = np.select(
        [events["user_reported_near_end"], events["auto_candidate"]],
        ["reported symptom", "automatic candidate"],
        default="other acceleration",
    )
    events = events.sort_values(
        ["is_suspicious", "user_reported_near_end", "candidate_score"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    events["candidate_rank"] = np.arange(1, len(events) + 1)
    return events


def _matching_events(events: pd.DataFrame, count: int = 3) -> pd.DataFrame:
    reported = events[events["user_reported_near_end"]].head(1)
    automatic = events[events["auto_candidate"] & ~events["user_reported_near_end"]].sort_values("candidate_score", ascending=False)
    candidates = pd.concat([reported, automatic], ignore_index=True).drop_duplicates("event_id").head(count)
    normal = events[~events["is_suspicious"]]
    if candidates.empty or normal.empty:
        return candidates
    features = ["rpm_min", "rpm_max", "throttle_mean", "boost_mean_kpa"]
    scale = normal[features].std().replace(0, 1).fillna(1)
    selected = [candidates]
    used = set()
    for _, candidate in candidates.iterrows():
        pool = normal[~normal["event_id"].isin(used)].copy()
        different_log = pool[pool["log"] != candidate["log"]]
        if not different_log.empty:
            pool = different_log
        distance = (((pool[features] - candidate[features]) / scale) ** 2).sum(axis=1)
        match = pool.loc[[distance.idxmin()]]
        used.add(match.iloc[0]["event_id"])
        selected.append(match)
    return pd.concat(selected, ignore_index=True)


def _fmt(value, digits: int = 2) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def _reported_core_analysis(
    logs: Sequence[LoadedLog],
    reported_config: Optional[ReportedSymptom] = None,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Find the strongest 5 s oscillation near the reported log end and compare like-for-like steps."""
    if reported_config is None:
        return {}, pd.DataFrame()
    reported = next((log for log in logs if log.metadata.path.name == reported_config.log), None)
    if reported is None or reported.data.empty:
        return {}, pd.DataFrame()
    data = reported.data
    end = float(data["time_s"].max())
    tail = data[data["time_s"] >= end - reported_config.detail_tail_s].copy()
    dt = float(tail["time_s"].diff().median())
    one_s = max(3, int(round(1.0 / dt)))
    five_s = max(5, int(round(5.0 / dt)))
    rpm_trend = tail["rpm"].rolling(one_s, center=True, min_periods=1).median()
    oscillation = (tail["rpm"] - rpm_trend).rolling(five_s, center=True, min_periods=five_s // 2).std()
    oscillation = oscillation.where(tail["throttle"] >= 30)
    center_index = oscillation.idxmax()
    center = float(tail.loc[center_index, "time_s"])
    start, stop = center - 2.5, center + 2.5
    core = data[data["time_s"].between(start, stop)].copy()
    # Exclude the final pedal lift from the oscillation metrics; it is an input
    # transition, not part of the repeating high-load symptom.
    high_load = core["throttle"] >= 70.0
    if high_load.sum() >= 20:
        core = core.loc[high_load].copy()
        start, stop = float(core["time_s"].min()), float(core["time_s"].max())

    residual = core["rpm"] - core["rpm"].rolling(one_s, center=True, min_periods=1).mean()
    frequencies, power = welch(residual.to_numpy(), fs=1.0 / dt, nperseg=min(128, len(core)))
    band = (frequencies >= 1.0) & (frequencies <= 9.0)
    dominant_hz = float(frequencies[band][np.argmax(power[band])]) if band.any() else float("nan")

    bounds = {
        "rpm": (float(core["rpm"].min()), float(core["rpm"].max())),
        "throttle": (float(core["throttle"].quantile(0.10)), float(core["throttle"].max())),
        "boost_rel": (float(core["boost_rel"].min()), float(core["boost_rel"].max())),
    }
    channels = ["rpm", "maf", "throttle", "boost_rel", "lambda_1", "ignition_timing"]
    rows = []
    for channel in channels:
        core_step = core[channel].diff().abs()
        reference_steps = []
        for log in logs:
            if log.metadata.path.name == reported_config.log or log.data.empty or channel not in log.data:
                continue
            frame = log.data
            mask = (
                frame["rpm"].between(*bounds["rpm"])
                & frame["throttle"].between(max(70.0, bounds["throttle"][0]), 100.0)
                & frame["boost_rel"].between(bounds["boost_rel"][0] - 5, bounds["boost_rel"][1] + 5)
            )
            adjacent = mask & mask.shift(fill_value=False) & (frame["time_s"].diff() < 0.12)
            reference_steps.extend(frame[channel].diff().abs().loc[adjacent].dropna().tolist())
        reference_typical = float("nan")
        if reference_steps:
            reference_typical = float(np.median(reference_steps))
            if reference_typical == 0:
                reference_typical = float(np.quantile(reference_steps, 0.75))
        rows.append(
            {
                "parameter": channel,
                "reported_core_min": float(core[channel].min()),
                "reported_core_max": float(core[channel].max()),
                "reported_median_abs_step": float(core_step.median()),
                "matched_reference_typical_abs_step": reference_typical,
                "step_ratio": float(core_step.median() / reference_typical) if reference_typical > 0 else np.nan,
            }
        )
    summary = {
        "start_s": start,
        "end_s": stop,
        "dominant_hz": dominant_hz,
        "fbkc_min": float(core["fbkc"].min()) if "fbkc" in core else np.nan,
        "fbkc_max": float(core["fbkc"].max()) if "fbkc" in core else np.nan,
        "knock_sum_min": float(core["knock_sum"].min()) if "knock_sum" in core else np.nan,
        "knock_sum_max": float(core["knock_sum"].max()) if "knock_sum" in core else np.nan,
        "flkc_min": float(core["flkc"].min()) if "flkc" in core else np.nan,
        "flkc_max": float(core["flkc"].max()) if "flkc" in core else np.nan,
    }
    return summary, pd.DataFrame(rows)


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str], limit: int = 12) -> str:
    if frame.empty:
        return "該当なし"
    view = frame.loc[:, [c for c in columns if c in frame]].head(limit).copy()
    headers = list(view.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in view.iterrows():
        values = []
        for value in row:
            values.append(_fmt(value) if isinstance(value, (float, np.floating)) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _learning_summary(logs: Sequence[LoadedLog]) -> pd.DataFrame:
    rows = []
    for log in logs:
        if log.data.empty:
            continue
        row = {"log": log.metadata.path.name}
        for channel in ["iam", "flkc", "af_learning_1"]:
            if channel in log.data and not log.data[channel].dropna().empty:
                row[f"{channel}_min"] = float(log.data[channel].min())
                row[f"{channel}_median"] = float(log.data[channel].median())
                row[f"{channel}_max"] = float(log.data[channel].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _write_report(
    report_path: Path,
    logs: Sequence[LoadedLog],
    inventory: pd.DataFrame,
    statistics: pd.DataFrame,
    events: pd.DataFrame,
    knock_changes: pd.DataFrame,
    learning: pd.DataFrame,
    core_summary: Dict[str, float],
    core_metrics: pd.DataFrame,
    annotations: AnalysisAnnotations,
) -> None:
    reported_event = events[events["user_reported_near_end"]].sort_values("end_s").tail(1)
    top = events[events["is_suspicious"]].sort_values("candidate_rank")

    segmented_names = inventory.loc[inventory["header_segments"] > 1, "log"].tolist()
    empty_names = inventory.loc[inventory["rows"] == 0, "log"].tolist()
    segmented_sentence = (
        f"複数ヘッダを持つログ（{', '.join(segmented_names)}）は、各区間を個別解釈して列の和集合に統合しました。"
        if segmented_names
        else "記録途中でヘッダ構成が変わるログはありませんでした。"
    )
    empty_sentence = (
        f"データ行のないログ（{', '.join(empty_names)}）は解析対象から除外しました。"
        if empty_names
        else "すべてのログに解析対象データがありました。"
    )
    valid_periods = inventory["median_period_s"].dropna()
    typical_period = float(valid_periods.median()) if not valid_periods.empty else np.nan

    iam_values = learning.get("iam_median", pd.Series(dtype=float)).dropna()
    if not iam_values.empty and iam_values.nunique() == 1:
        iam_sentence = f"IAMは取得できた全ログで{iam_values.iloc[0]:.3f}固定でした。"
    elif not iam_values.empty:
        iam_sentence = f"IAM中央値はログ間で{iam_values.min():.3f}〜{iam_values.max():.3f}の範囲でした。"
    else:
        iam_sentence = "IAMは今回のログから比較できませんでした。"
    flkc_values = learning.dropna(subset=["flkc_median"]) if "flkc_median" in learning else pd.DataFrame()
    flkc_sentence = (
        f"FLKC中央値はログ間で{flkc_values['flkc_median'].min():+.2f}〜{flkc_values['flkc_median'].max():+.2f}°でした。"
        if not flkc_values.empty
        else "FLKCは今回のログから比較できませんでした。"
    )

    lines = [
        "# SG9 RomRaiderログ 初期探索解析",
        "",
        "## 結論（初期段階）",
        "",
        "この解析は候補の順位付けであり、部品故障や異常燃焼を確定するものではありません。",
    ]
    if not reported_event.empty:
        e = reported_event.iloc[0]
        lines += [
            "",
            f"体感申告ログの終盤では、{e.start_s:.2f}〜{e.end_s:.2f}秒に長い加速イベントを検出しました。",
            f"RPMは{e.rpm_start:.0f}→{e.rpm_end:.0f} rpm、MAF最大{e.maf_max_g_s:.2f} g/s、過給圧最大{e.boost_max_kpa:.1f} kPaです。",
            f"この区間のMAF周期指標は{int(e.maf_oscillation_cycles)}回、RPM周期指標は{int(e.rpm_oscillation_cycles)}回ですが、スロットル変化も含むため、これだけでMAFセンサー異常とは判断できません。",
            f"同区間ではFBKC最小{_fmt(e.fbkc_min_deg)}°、FLKC最小{_fmt(e.flkc_min_deg)}°、Knock Sum差{_fmt(e.knock_sum_delta, 0)}を記録しました。時間的な併発はありますが因果は未確定です。",
        ]
        if not pd.isna(e.maf_negative_outlier_fraction):
            lines.append(
                f"正常寄り加速から作ったRPM・スロットル・過給圧条件付きMAF基準に対し、低側3σ外れの割合は{100*e.maf_negative_outlier_fraction:.1f}%でした。"
            )
    if core_summary:
        metric = core_metrics.set_index("parameter")
        rpm_ratio = metric.loc["rpm", "step_ratio"]
        maf_ratio = metric.loc["maf", "step_ratio"]
        throttle_ratio = metric.loc["throttle", "step_ratio"]
        lines += [
            "",
            f"終盤で最も強い周期変動は{core_summary['start_s']:.2f}〜{core_summary['end_s']:.2f}秒に集中し、主周期成分は約{core_summary['dominant_hz']:.1f} Hzでした。",
            f"条件を合わせた他ログの高負荷サンプルと比べ、隣接サンプル間変化の代表値はRPM約{rpm_ratio:.1f}倍、MAF約{maf_ratio:.1f}倍、Throttle約{throttle_ratio:.1f}倍です。",
            "MAFだけが単独で飛んだのではなく、RPM、Throttle、過給圧、Lambda、点火時期にも同周期の揺れがあります。したがって、現ログからはMAFセンサー単独故障よりも、実トルク／吸気系の連動変動または共通の計測・電気的要因を先に残すのが妥当です。",
            f"この核心5秒ではFBKCは{_fmt(core_summary['fbkc_min'])}〜{_fmt(core_summary['fbkc_max'])}°、Knock Sumは{_fmt(core_summary['knock_sum_min'],0)}〜{_fmt(core_summary['knock_sum_max'],0)}で変化していません。少なくとも記録上、FBKCやKnock Sum変化がこの周期変動を直接開始した形ではありません。",
        ]
    lines += [
        "",
        "Primary Wastegate Duty Cycleは、当該車両でSSM項目が有効と確認できない限り、表示値を実WGDCとは解釈しません。",
        f"{iam_sentence} ECUリセットとの関連は、リセット前ログまたは学習進行を追った時系列がないため証明できません。",
        "",
        "## 読み込み確認",
        "",
        f"Timestampは `Time (msec)` のログ開始からの経過ミリ秒で、解析では秒へ変換しています。ログ間の代表的な周期は約{_fmt(typical_period, 3)}秒です。",
        segmented_sentence + " 統合後の空欄は、その区間で未取得という意味です。",
        empty_sentence + " 詳細統計は `log_inventory.csv` と `parameter_statistics.csv` に保存しています。",
        "",
        _markdown_table(inventory, ["log", "rows", "duration_s", "median_period_s", "header_segments", "missing_values"]),
        "",
        "取得単位は列名から保存しています。今回確認できた中心項目は rpm、Throttle (%)、MAF (g/s)、Manifold Relative Pressure (kPa)、Lambda、点火時期 (degrees)、ノック補正、IAM、Knock Sum、左右VVT、WGDCです。MAF電圧、アクセルペダル、絶対圧、A/F Learning、Battery Voltageは今回のCSVにありません。",
        "",
        "## ハンチング候補（自動探索）",
        "",
        "候補スコアは、一定寄りのスロットル中のRPM・MAF・過給圧の周期性、条件付きMAF外れ、FBKCを組み合わせた探索指標です。ギアチェンジ相当の急なRPM低下ではイベントを分割しています。",
        "",
        _markdown_table(top, ["candidate_rank", "candidate_kind", "log", "start_s", "end_s", "rpm_min", "rpm_max", "throttle_mean", "maf_outlier_fraction", "fbkc_min_deg", "flkc_min_deg", "knock_sum_delta", "candidate_score"]),
        "",
        "上位候補には長い緩加速や運転操作由来の変動も入り得ます。PNGの波形と実際の体感時刻を突き合わせて候補を絞る必要があります。",
        "",
        "## ECU学習値のログ間比較",
        "",
        _markdown_table(learning, ["log", "iam_min", "iam_median", "iam_max", "flkc_min", "flkc_median", "flkc_max"]),
        "",
        f"{iam_sentence} {flkc_sentence} 走行条件が同一とは限らないため、ECUリセット後の学習進行や症状変化との因果は確定できません。",
    ]

    log_by_name = {log.metadata.path.name: log for log in logs}
    for inspection in annotations.inspection_windows:
        selected = log_by_name.get(inspection.log)
        context_start = inspection.start_s - inspection.context_s
        context_end = inspection.end_s + inspection.context_s
        lines += ["", f"## {inspection.label}", ""]
        if selected is None or selected.data.empty:
            lines.append(f"指定ログ `{inspection.log}` に解析可能なサンプルがありません。")
            continue
        window = selected.data[selected.data["time_s"].between(context_start, context_end)]
        overlapping = events[
            (events["log"] == inspection.log)
            & (events["start_s"] <= inspection.end_s)
            & (events["end_s"] >= inspection.start_s)
        ]
        if window.empty:
            lines.append("指定範囲にサンプルはありません。")
            continue
        fbkc_min = window["fbkc"].min() if "fbkc" in window else np.nan
        flkc_min = window["flkc"].min() if "flkc" in window else np.nan
        knock_delta = window["knock_sum"].max() - window["knock_sum"].min() if "knock_sum" in window else np.nan
        lines.append(
            f"{context_start:.2f}〜{context_end:.2f}秒を確認したところ、FBKC最小{_fmt(fbkc_min)}°、"
            f"FLKC最小{_fmt(flkc_min)}°、Knock Sum範囲差{_fmt(knock_delta, 0)}でした。"
        )
        lines.append("この窓は自動抽出した持続加速イベントと重なります。" if not overlapping.empty else "この窓は今回の持続加速イベント条件には該当しません。")

    lines += [
        "",
        "## MAFについて分かったこと",
        "",
        "- MAFの絶対値だけでなく、同程度のRPM・スロットル・過給圧における近傍データの中央値から期待値を作り、外れ率をイベント単位で算出しました。",
        "- MAFと過給圧は多くの加速で同方向に変化します。候補区間のMAF上下が、吸入空気量の実変動かセンサー信号の乱れかは、このログ項目だけでは分離できません。",
        "- 今回はMAF Sensor Voltageが記録されていないため、g/s換算値の異常と生電圧の異常を照合できません。",
        "",
        "## Knock関連",
        "",
        f"Knock Sumの変化点をログ別に抽出し、前後3秒の切り出し範囲を `knock_changes.csv` に保存しました（{len(knock_changes)}件）。",
        "Knock Sum増加、FBKC/FLKC、点火時期、RPM/MAF/過給圧の順序は各窓で確認できますが、シフトや機械ノイズを除外できないため、Knock Sum変化だけを異常燃焼の証拠にはしません。",
        "",
        "## 原因候補と現時点で判断できないこと",
        "",
        "- 吸入空気量または過給圧の実変動、MAF計測の乱れ、点火補正、燃調、スロットル操作、クラッチ／ギア操作を候補として残します。",
        "- MAF、過給圧、RPMの周期変動が同時なら実トルク変動の可能性が上がりますが、どれが起点かは現項目だけでは決まりません。",
        "- 低IAMと大きな負のFLKCは注目点ですが、症状の原因か結果か、学習リセット後の一時状態かは比較ログ不足です。",
        "- ログが症状途中で停止した可能性がある場合、注釈対象ログ末尾より後の挙動は判断できません。",
        "",
        "## 次回取得したい項目・方法",
        "",
        "1. 現在の項目を維持しつつ、Mass Airflow Sensor Voltage、Accelerator Pedal、Manifold Absolute Pressure、A/F Learning #1、Battery Voltageを追加する。",
        "2. ロガー負荷で周期が落ちすぎないよう、追加は上記の切り分け必須項目から少数ずつ行う。",
        "3. 症状を感じた瞬間に同乗者の操作や音声で印を付け、発生後10〜20秒はログを止めない。安全を最優先する。",
        "4. 同じギア・近いスロットルで、症状あり／なしを各2〜3回記録する。公道では再現を無理に狙わない。",
        "5. ECUリセット日時、走行距離、燃料、外気温、ギア、症状体感時刻をメモし、学習進行との比較材料にする。",
        "",
        "## 出力図",
        "",
        "各ログの全体図、注釈指定ログの終盤詳細図、MAF対RPM／Throttle／Boost、候補対比較加速を `output/plots/` に保存しました。図のラベルは環境依存フォントを避けるため英語です。",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    paths: Iterable[Path],
    output_root: Path,
    annotations: Optional[AnalysisAnnotations] = None,
) -> None:
    annotations = annotations or AnalysisAnnotations()
    logs = load_logs(paths)
    if not logs:
        raise SystemExit("No CSV files were supplied.")
    report_dir = output_root / "reports"
    plot_dir = output_root / "plots"
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    inventory = inventory_frame(logs)
    statistics = statistics_frame(logs)
    inventory.to_csv(report_dir / "log_inventory.csv", index=False)
    statistics.to_csv(report_dir / "parameter_statistics.csv", index=False)

    event_parts = [extract_events(log.data, log.metadata.path.name) for log in logs if not log.data.empty]
    events = pd.concat([part for part in event_parts if not part.empty], ignore_index=True) if any(not p.empty for p in event_parts) else pd.DataFrame()
    durations = {log.metadata.path.name: log.metadata.duration_s for log in logs}
    events = _classify_events(events, durations, annotations.reported_symptom)
    samples = _event_samples(logs, events)
    samples, events = _maf_reference(samples, events)
    events = _rank_events(events)
    events.to_csv(report_dir / "events.csv", index=False)
    if not samples.empty:
        samples.to_csv(report_dir / "event_samples_with_maf_reference.csv", index=False)

    knock_parts = []
    for log in logs:
        changes = knock_change_windows(log.data)
        if not changes.empty:
            changes.insert(0, "log", log.metadata.path.name)
            knock_parts.append(changes)
    knock_changes = pd.concat(knock_parts, ignore_index=True) if knock_parts else pd.DataFrame()
    knock_changes.to_csv(report_dir / "knock_changes.csv", index=False)
    learning = _learning_summary(logs)
    learning.to_csv(report_dir / "learning_summary.csv", index=False)
    knock_timing = _knock_timing_summary(logs)
    knock_timing.to_csv(report_dir / "knock_timing_summary.csv", index=False)
    core_summary, core_metrics = _reported_core_analysis(logs, annotations.reported_symptom)
    core_metrics.to_csv(report_dir / "reported_core_comparison.csv", index=False)

    log_map = {log.metadata.path.name: log.data for log in logs}
    for log in logs:
        if not log.data.empty:
            plot_overview(log.data, log.metadata.path.stem, plot_dir)
    reported = annotations.reported_symptom
    if reported is not None and reported.log in log_map:
        plot_end_detail(
            log_map[reported.log],
            Path(reported.log).stem,
            plot_dir,
            seconds=reported.detail_tail_s,
        )
        highlight = None
        if core_summary:
            highlight = (core_summary["start_s"], core_summary["end_s"])
        plot_reported_timing_detail(
            log_map[reported.log],
            Path(reported.log).stem,
            plot_dir,
            seconds=reported.detail_tail_s,
            highlight=highlight,
        )
    if not samples.empty:
        sample_flags = samples["event_id"].map(events.set_index("event_id")["is_suspicious"]).fillna(False)
        samples["is_suspicious"] = sample_flags.astype(bool)
        plot_maf_relationships(samples, plot_dir)
    plot_high_load_boost(log_map, plot_dir)
    plot_knock_correction_timeline(log_map, plot_dir)
    plot_knock_timing_transition_summary(knock_timing, plot_dir)
    plot_knock_timing_compact_summary(
        knock_timing,
        plot_dir,
        reported_log=reported.log if reported is not None else None,
    )
    plot_event_comparison(_matching_events(events), log_map, plot_dir)

    _write_report(
        report_dir / "initial_analysis.md",
        logs,
        inventory,
        statistics,
        events,
        knock_changes,
        learning,
        core_summary,
        core_metrics,
        annotations,
    )
    print(f"Loaded {len(logs)} logs ({sum(m.metadata.row_count for m in logs):,} rows).")
    print(f"Extracted {len(events)} acceleration events; {int(events['is_suspicious'].sum()) if not events.empty else 0} ranked candidates.")
    print(f"Wrote reports to {report_dir} and plots to {plot_dir}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path, help="RomRaider CSV files")
    parser.add_argument("--output", type=Path, default=Path("output"), help="Output root (default: output)")
    parser.add_argument(
        "--annotations",
        type=Path,
        help="Optional JSON file containing private symptom and inspection annotations",
    )
    args = parser.parse_args()
    run(args.paths, args.output, load_annotations(args.annotations))


if __name__ == "__main__":
    main()
