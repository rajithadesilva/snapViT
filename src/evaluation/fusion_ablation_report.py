"""Build readable CSV, Markdown, and plot summaries for a fusion campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import os
from collections import defaultdict
from pathlib import Path
from typing import Any


RUN_COLUMNS = [
    "stage",
    "variant",
    "seed",
    "status",
    "best_val_loss",
    "recall@1",
    "recall@5",
    "mrr",
    "runtime_seconds",
    "trainable_parameters",
    "total_parameters",
    "error",
    "run_dir",
    "heldout_runtime_seconds",
    "heldout_recall@1",
    "heldout_recall@5",
    "heldout_mrr",
    "localization_mean_m",
    "localization_median_m",
    "localization_success_1m",
    "localization_success_2m",
    "localization_success_3m",
]


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        return value if isinstance(value, dict) else {}
    except (OSError, ValueError):
        return {}


def _nested(source: dict[str, Any], *keys: str) -> Any:
    current: Any = source
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _first(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _localization_success(localization: dict[str, Any], threshold_m: float) -> Any:
    suffix = f"{float(threshold_m)}m_ratio"
    return _first(
        localization.get(f"nearest_topk_within_{suffix}"),
        localization.get(f"topk_recall_within_{suffix}"),
    )


def _infer_identity(run_dir: Path, root: Path, config: dict[str, Any], status: dict[str, Any]) -> tuple[str, str, str]:
    relative = run_dir.relative_to(root)
    parts = relative.parts
    variant = str(_first(status.get("variant"), config.get("ground_fusion_variant"), ""))
    if not variant:
        variant = next((part.upper() for part in parts if part.upper() in {f"N{i}" for i in range(10)}), "")
    seed_value = _first(status.get("seed"), config.get("seed"))
    if seed_value is None:
        seed_value = next((part.split("_", 1)[1] for part in parts if part.startswith("seed_")), "")
    stage = str(_first(status.get("stage"), config.get("stage"), ""))
    if not stage:
        stage = next((part for part in parts if part.lower().startswith("stage")), "")
    return stage, variant, str(seed_value)


def _history_metrics(run_dir: Path) -> dict[str, Any]:
    candidates = [run_dir / "history.csv", run_dir / "history" / "loss_history.csv"]
    for path in candidates:
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        valid = []
        for row in rows:
            try:
                valid.append((float(row["val_loss"]), row))
            except (KeyError, TypeError, ValueError):
                continue
        if valid:
            _, best = min(valid, key=lambda item: item[0])
            return best
    return {}


def discover_runs(root: Path) -> list[dict[str, Any]]:
    """Discover one record per experiment directory containing status.json."""
    records: list[dict[str, Any]] = []
    for status_path in sorted(root.rglob("status.json")):
        run_dir = status_path.parent
        status = _read_json(status_path)
        config = _read_json(run_dir / "config.json") or _read_json(run_dir / "resolved_config.json")
        result = _read_json(run_dir / "result.json") or _read_json(run_dir / "metrics.json")
        history = _history_metrics(run_dir)
        evaluation = _read_json(run_dir / "evaluation" / "evaluation_summary.json")
        evaluation_status = _read_json(
            run_dir / "evaluation" / "evaluation_status.json"
        )
        if not evaluation:
            evaluation = _read_json(run_dir / "evaluation_summary.json")
        stage, variant, seed = _infer_identity(run_dir, root, config, status)

        best = result.get("best_metrics", {}) if isinstance(result.get("best_metrics"), dict) else {}
        val = result.get("validation", {}) if isinstance(result.get("validation"), dict) else {}
        retrieval = evaluation.get("retrieval", {}) if isinstance(evaluation.get("retrieval"), dict) else {}
        localization = evaluation.get("localization", {}) if isinstance(evaluation.get("localization"), dict) else {}
        records.append(
            {
                "stage": stage,
                "variant": variant,
                "seed": seed,
                "status": str(_first(status.get("status"), result.get("status"), "unknown")),
                "best_val_loss": _first(best.get("val_loss"), val.get("loss"), result.get("best_val_loss"), history.get("val_loss")),
                "recall@1": _first(best.get("recall@1"), val.get("recall@1"), result.get("recall@1")),
                "recall@5": _first(best.get("recall@5"), val.get("recall@5"), result.get("recall@5")),
                "mrr": _first(best.get("mrr"), val.get("mrr"), result.get("mrr")),
                "runtime_seconds": _first(status.get("runtime_seconds"), result.get("runtime_seconds")),
                "trainable_parameters": _first(result.get("trainable_parameters"), status.get("trainable_parameters")),
                "total_parameters": _first(result.get("total_parameters"), status.get("total_parameters")),
                "error": str(_first(status.get("error"), result.get("error"), "")),
                "run_dir": str(run_dir.relative_to(root)),
                "heldout_runtime_seconds": evaluation_status.get("runtime_seconds"),
                "heldout_recall@1": retrieval.get("recall@1"),
                "heldout_recall@5": retrieval.get("recall@5"),
                "heldout_mrr": retrieval.get("mrr"),
                "localization_mean_m": localization.get("nearest_topk_distance_mean_m"),
                "localization_median_m": localization.get("nearest_topk_distance_median_m"),
                "localization_success_1m": _localization_success(localization, 1.0),
                "localization_success_2m": _localization_success(localization, 2.0),
                "localization_success_3m": _localization_success(localization, 3.0),
            }
        )

    heldout_status = _read_json(root / "heldout_status.json")
    evaluations = heldout_status.get("evaluations", [])
    if isinstance(evaluations, list):
        for evaluation in evaluations:
            if not isinstance(evaluation, dict) or not evaluation.get("canonical_full"):
                continue
            summary = evaluation.get("summary", {})
            retrieval = summary.get("retrieval", {}) if isinstance(summary, dict) else {}
            localization = summary.get("localization", {}) if isinstance(summary, dict) else {}
            records.append(
                {
                    "stage": "heldout_full",
                    "variant": str(evaluation.get("variant", "")),
                    "seed": str(evaluation.get("seed", "")),
                    "status": str(_first(evaluation.get("status"), evaluation.get("state"), "unknown")),
                    "best_val_loss": None,
                    "recall@1": None,
                    "recall@5": None,
                    "mrr": None,
                    "runtime_seconds": evaluation.get("runtime_seconds"),
                    "trainable_parameters": None,
                    "total_parameters": None,
                    "error": str(evaluation.get("error", "")),
                    "run_dir": "winner/evaluation_full",
                    "heldout_runtime_seconds": evaluation.get("runtime_seconds"),
                    "heldout_recall@1": retrieval.get("recall@1"),
                    "heldout_recall@5": retrieval.get("recall@5"),
                    "heldout_mrr": retrieval.get("mrr"),
                    "localization_mean_m": localization.get("nearest_topk_distance_mean_m"),
                    "localization_median_m": localization.get("nearest_topk_distance_median_m"),
                    "localization_success_1m": _localization_success(localization, 1.0),
                    "localization_success_2m": _localization_success(localization, 2.0),
                    "localization_success_3m": _localization_success(localization, 3.0),
                }
            )
    return records


def _float(value: Any) -> float | None:
    try:
        converted = float(value)
        return converted if math.isfinite(converted) else None
    except (TypeError, ValueError):
        return None


def aggregate_variants(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["variant"] and record["status"] in {"complete", "completed", "success"}:
            grouped[(record["stage"], record["variant"])].append(record)
    output = []
    for (stage, variant), group in sorted(grouped.items()):
        row: dict[str, Any] = {"stage": stage, "variant": variant, "num_runs": len(group)}
        for metric in (
            "best_val_loss",
            "recall@1",
            "recall@5",
            "mrr",
            "heldout_recall@1",
            "heldout_recall@5",
            "heldout_mrr",
            "localization_mean_m",
            "localization_median_m",
            "localization_success_1m",
            "localization_success_2m",
            "localization_success_3m",
            "heldout_runtime_seconds",
        ):
            values = [value for value in (_float(item.get(metric)) for item in group) if value is not None]
            row[f"{metric}_mean"] = statistics.fmean(values) if values else None
            row[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else (0.0 if values else None)
            row[f"{metric}_count"] = len(values)
        row["heldout_num_runs"] = row.get("heldout_recall@1_count", 0)
        row["runtime_seconds_total"] = sum(_float(item.get("runtime_seconds")) or 0.0 for item in group)
        row["heldout_runtime_seconds_total"] = sum(
            _float(item.get("heldout_runtime_seconds")) or 0.0 for item in group
        )
        row["trainable_parameters"] = _first(*[item.get("trainable_parameters") for item in group])
        output.append(row)

    for stage in {row["stage"] for row in output}:
        stage_rows = [row for row in output if row["stage"] == stage]
        baseline = next((row for row in stage_rows if row["variant"] == "N0"), None)
        for metric, ascending in (("best_val_loss_mean", True), ("recall@1_mean", False), ("mrr_mean", False)):
            ranked = [row for row in stage_rows if _float(row.get(metric)) is not None]
            ranked.sort(key=lambda row: float(row[metric]), reverse=not ascending)
            previous_value = None
            tied_rank = 1
            for position, row in enumerate(ranked, 1):
                value = float(row[metric])
                if previous_value is None or value != previous_value:
                    tied_rank = position
                    previous_value = value
                row[metric.replace("_mean", "_rank")] = tied_rank
        for row in stage_rows:
            ranks = [row.get("best_val_loss_rank"), row.get("recall@1_rank"), row.get("mrr_rank")]
            row["rank_sum"] = sum(ranks) if all(rank is not None for rank in ranks) else None
            if baseline:
                for metric in ("best_val_loss_mean", "recall@1_mean", "recall@5_mean", "mrr_mean"):
                    value, base = _float(row.get(metric)), _float(baseline.get(metric))
                    row[f"delta_{metric}_vs_n0"] = value - base if value is not None and base is not None else None
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str] | None = None) -> None:
    if columns is None:
        columns = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        if not columns:
            return
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _fmt(value: Any, digits: int = 4) -> str:
    numeric = _float(value)
    return "—" if numeric is None else f"{numeric:.{digits}f}"


def _selection_summary(root: Path) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    candidates = list(root.glob("*selection*.json")) + list(root.glob("*winner*.json")) + [root / "campaign_state.json"]
    for path in candidates:
        if path.is_file():
            merged.update(_read_json(path))
    return merged


def _write_markdown(root: Path, records: list[dict[str, Any]], variants: list[dict[str, Any]]) -> None:
    completed = sum(item["status"] in {"complete", "completed", "success"} for item in records)
    failed = [item for item in records if item["status"] in {"failed", "error"}]
    total_runtime = sum(_float(item.get("runtime_seconds")) or 0.0 for item in records)
    total_runtime += sum(
        _float(item.get("heldout_runtime_seconds")) or 0.0
        for item in records
        if item.get("stage") != "heldout_full"
    )
    selection = _selection_summary(root)
    winner = _first(selection.get("winner"), selection.get("winning_variant"), selection.get("selected_variant"))
    finalists = _first(selection.get("finalists"), selection.get("top_three"), selection.get("top3"))
    setup = _read_json(root / "campaign_config.json") or _read_json(root / "config.json")
    lines = [
        "# Height-aware Fusion Ablation Results",
        "",
        f"Runs discovered: **{len(records)}** · completed: **{completed}** · failed: **{len(failed)}** · recorded runtime: **{total_runtime / 3600:.2f} h**",
        "",
        "## Campaign setup",
        "",
    ]
    setup_written = False
    for key in (
        "base_checkpoint",
        "data_roots",
        "split_seed",
        "seeds",
        "device",
        "training_scene_count",
        "validation_scene_count",
        "excluded_scene_count",
    ):
        if key in setup:
            lines.append(f"- {key.replace('_', ' ').title()}: `{setup[key]}`")
            setup_written = True
    if not setup_written:
        lines.append("See per-run configurations for the fully resolved setup.")
    if finalists is not None:
        lines.append(f"- Stage-1 finalists: **{finalists}**")
    if winner is not None:
        lines.append(f"- Validation-selected winner: **{winner}**")

    validation_stages = sorted(
        {
            row["stage"]
            for row in variants
            if _float(row.get("best_val_loss_mean")) is not None
        }
    )
    for stage in validation_stages:
        lines.extend(
            [
                "",
                f"## {stage or 'Validation'} leaderboard",
                "",
                "| Variant | Runs | Validation loss | Δ loss vs N0 | Recall@1 | Δ R@1 vs N0 | Recall@5 | MRR | Δ MRR vs N0 | Rank sum |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        ranked = [row for row in variants if row["stage"] == stage]
        ranked.sort(key=lambda row: (_float(row.get("rank_sum")) is None, _float(row.get("rank_sum")) or math.inf, _float(row.get("best_val_loss_mean")) or math.inf))
        for row in ranked:
            lines.append(
                f"| {row['variant']} | {row['num_runs']} | {_fmt(row.get('best_val_loss_mean'))} ± {_fmt(row.get('best_val_loss_std'))} "
                f"| {_fmt(row.get('delta_best_val_loss_mean_vs_n0'))} "
                f"| {_fmt(row.get('recall@1_mean'))} ± {_fmt(row.get('recall@1_std'))} "
                f"| {_fmt(row.get('delta_recall@1_mean_vs_n0'))} "
                f"| {_fmt(row.get('recall@5_mean'))} ± {_fmt(row.get('recall@5_std'))} "
                f"| {_fmt(row.get('mrr_mean'))} ± {_fmt(row.get('mrr_std'))} "
                f"| {_fmt(row.get('delta_mrr_mean_vs_n0'))} | {_fmt(row.get('rank_sum'), 0)} |"
            )

    heldout = [row for row in variants if _float(row.get("heldout_recall@1_mean")) is not None]
    if heldout:
        lines.extend(
            [
                "",
                "## Held-out run1 confirmation",
                "",
                "| Variant | Successful seeds | Recall@1 | Recall@5 | MRR | Top-5 ≤1m | Top-5 ≤2m | Top-5 ≤3m | Mean (m) | Median (m) | Eval runtime (h) |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(
            heldout,
            key=lambda item: (item["stage"] != "heldout_full", item["variant"]),
        ):
            label = (
                f"{row['variant']} (canonical full)"
                if row["stage"] == "heldout_full"
                else row["variant"]
            )
            lines.append(
                f"| {label} | {int(row.get('heldout_num_runs', 0))}/{int(row.get('num_runs', 0))} "
                f"| {_fmt(row.get('heldout_recall@1_mean'))} | {_fmt(row.get('heldout_recall@5_mean'))} "
                f"| {_fmt(row.get('heldout_mrr_mean'))} "
                f"| {_fmt(row.get('localization_success_1m_mean'))} "
                f"| {_fmt(row.get('localization_success_2m_mean'))} "
                f"| {_fmt(row.get('localization_success_3m_mean'))} "
                f"| {_fmt(row.get('localization_mean_m_mean'), 3)} "
                f"| {_fmt(row.get('localization_median_m_mean'), 3)} "
                f"| {_fmt((_float(row.get('heldout_runtime_seconds_total')) or 0.0) / 3600.0, 2)} |"
            )

    lines.extend(["", "## Failures", ""])
    heldout_status = _read_json(root / "heldout_status.json")
    heldout_failures = [
        item
        for item in heldout_status.get("evaluations", [])
        if isinstance(item, dict) and item.get("state") == "failed"
    ] if isinstance(heldout_status.get("evaluations", []), list) else []
    if failed:
        lines.extend(f"- `{row['run_dir']}`: {row['error'] or row['status']}" for row in failed)
    if heldout_failures:
        lines.extend(
            f"- Held-out `{item.get('name', 'evaluation')}`: {item.get('error', 'failed')}"
            for item in heldout_failures
        )
    if not failed and not heldout_failures:
        lines.append("No failed experiments recorded.")
    lines.extend(["", "Raw results: [`per_run.csv`](per_run.csv), [`by_variant.csv`](by_variant.csv).", ""])
    (root / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def _write_plots(root: Path, variants: list[dict[str, Any]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/snapvit-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/snapvit-cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    stages = sorted({row["stage"] for row in variants})
    preferred_stage = "stage2" if "stage2" in stages else (stages[-1] if stages else "")
    plot_rows = [row for row in variants if row["stage"] == preferred_stage and _float(row.get("best_val_loss_mean")) is not None]
    if not plot_rows:
        return
    labels = [row["variant"] for row in plot_rows]
    values = [float(row["best_val_loss_mean"]) for row in plot_rows]
    errors = [float(row["best_val_loss_std"] or 0.0) for row in plot_rows]
    figure, axis = plt.subplots(figsize=(max(7, len(labels) * 0.7), 4.5))
    axis.bar(labels, values, yerr=errors, capsize=3)
    axis.set_ylabel("Best validation loss (lower is better)")
    axis.set_title("Fusion ablation leaderboard")
    figure.tight_layout()
    figure.savefig(root / "leaderboard.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 5))
    plotted = False
    for history_path in sorted(root.rglob("history.csv")) + sorted(root.rglob("loss_history.csv")):
        try:
            with history_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            epochs = [int(row["epoch"]) for row in rows if row.get("val_loss") not in (None, "")]
            losses = [float(row["val_loss"]) for row in rows if row.get("val_loss") not in (None, "")]
            if epochs and len(epochs) == len(losses):
                axis.plot(epochs, losses, alpha=0.75, label=str(history_path.parent.relative_to(root)))
                plotted = True
        except (OSError, KeyError, ValueError):
            continue
    if plotted:
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Validation loss")
        axis.set_title("Validation curves")
        if len(axis.lines) <= 15:
            axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(root / "validation_curves.png", dpi=160)
    plt.close(figure)


def build_report(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run_dir.mkdir(parents=True, exist_ok=True)
    records = discover_runs(run_dir)
    aggregate_records = records
    stage1_selection = _read_json(run_dir / "stage1_selection.json")
    finalists = _first(
        stage1_selection.get("finalists"),
        stage1_selection.get("top_variants"),
    )
    if isinstance(finalists, list) and finalists:
        eligible = set(finalists)
        aggregate_records = [
            record
            for record in records
            if record.get("stage") != "stage2"
            or record.get("variant") in eligible
        ]
    variants = aggregate_variants(aggregate_records)
    _write_csv(run_dir / "per_run.csv", records, RUN_COLUMNS)
    _write_csv(run_dir / "by_variant.csv", variants)
    _write_markdown(run_dir, records, variants)
    _write_plots(run_dir, variants)
    return records, variants


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh fusion ablation Markdown/CSV reports.")
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    records, variants = build_report(Path(args.run_dir).resolve())
    print(f"Report updated: {len(records)} runs, {len(variants)} variants")


if __name__ == "__main__":
    main()
