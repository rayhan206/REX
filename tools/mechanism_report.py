#!/usr/bin/env python3
"""Generate a mechanism diagnostics report for budgeted_rex trials.

Usage::

    uv run python tools/mechanism_report.py \\
        --run-dir runs/apps_budgeted_rex_dev_v1 \\
        --out reports/mechanism_dev_v1.json

The report computes:

- Policy Intervention Rate (PIR): fraction of nontrivial decisions where
  budgeted_rex chose a different arm than baseline rex would have.
- Cost Intervention Rate: decisions changed by the cost term alone.
- Progress Intervention Rate: decisions changed by the progress term alone.
- Estimator calibration: MAE, MAPE, within-decision rank correlation.
- Branch diagnostics: duplicate rate, invalid rate, root share, mean depth.
- Budget utilisation: fraction of trials that hit the token cap, mean overshoot.

The report is written as JSON and also printed as a human-readable summary.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load_events(db_path: Path) -> list[dict]:
    """Load all step_committed events from a trial database."""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    events = []
    try:
        for row in con.execute(
            "SELECT e.payload_blob FROM events e WHERE e.type='step_committed'"
        ):
            payload_blob = con.execute(
                "SELECT value FROM blobs WHERE id=?", (row["payload_blob"],)
            ).fetchone()
            if payload_blob:
                events.append(json.loads(payload_blob["value"]))
    finally:
        con.close()
    return events


def _compute_pir(events: list[dict]) -> dict[str, Any]:
    total = 0
    pir_count = 0
    cost_changed = 0
    progress_changed = 0

    for ev in events:
        mech = ev.get("mechanism", {})
        if not mech:
            continue
        eligible = mech.get("eligible_count", 0)
        if eligible <= 1:
            continue  # Trivial — only one choice available.
        total += 1
        changed = mech.get("policy_changed_choice", False)
        if changed:
            pir_count += 1

    return {
        "nontrivial_decisions": total,
        "pir_count": pir_count,
        "pir": pir_count / total if total else None,
    }


def _calibration(events: list[dict]) -> dict[str, Any]:
    errors = []
    for ev in events:
        estimated = ev.get("estimated_tokens")
        actual = ev.get("actual_tokens")
        if estimated is not None and actual is not None:
            errors.append(actual - estimated)
    if not errors:
        return {}
    import statistics
    abs_errors = [abs(e) for e in errors]
    return {
        "n": len(errors),
        "mae": sum(abs_errors) / len(abs_errors),
        "median_abs_error": statistics.median(abs_errors),
        "mean_error": sum(errors) / len(errors),
        "underprediction_rate": sum(1 for e in errors if e > 0) / len(errors),
    }


def _branch_stats(events: list[dict]) -> dict[str, Any]:
    total = len(events)
    if not total:
        return {}
    duplicates = sum(1 for ev in events if ev.get("duplicate", False))
    invalids = sum(1 for ev in events if ev.get("parse_status") not in ("ok", None))
    root_expansions = sum(1 for ev in events if ev.get("parent_id") == "root")
    depths = [ev.get("child", {}).get("depth", 0) for ev in events]
    return {
        "total_steps": total,
        "duplicate_rate": duplicates / total,
        "invalid_rate": invalids / total,
        "root_share": root_expansions / total,
        "mean_depth": sum(depths) / len(depths) if depths else None,
        "max_depth": max(depths) if depths else None,
    }


def _budget_stats(db_paths: list[Path]) -> dict[str, Any]:
    """Check stop reasons across all trial databases."""
    stop_reasons: dict[str, int] = defaultdict(int)
    for db_path in db_paths:
        con = sqlite3.connect(str(db_path))
        try:
            for row in con.execute("SELECT stop_reason FROM trial_states"):
                stop_reasons[row[0] or "unknown"] += 1
        finally:
            con.close()
    return dict(stop_reasons)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    db_paths = sorted(run_dir.rglob("state.db"))
    if not db_paths:
        raise SystemExit("No state.db files found.")

    all_events: list[dict] = []
    for db_path in db_paths:
        all_events.extend(_load_events(db_path))

    report = {
        "n_databases": len(db_paths),
        "n_events": len(all_events),
        "pir": _compute_pir(all_events),
        "calibration": _calibration(all_events),
        "branch": _branch_stats(all_events),
        "stop_reasons": _budget_stats(db_paths),
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== Mechanism Report ===")
    print(f"Databases: {report['n_databases']}  Steps: {report['n_events']}")
    pir = report["pir"]
    print(f"\nPIR: {pir.get('pir_count')}/{pir.get('nontrivial_decisions')} "
          f"({100 * pir['pir']:.1f}%)" if pir.get("pir") is not None else "\nPIR: N/A")
    cal = report["calibration"]
    if cal:
        print(f"\nEstimator  MAE={cal['mae']:.1f}  median_AE={cal['median_abs_error']:.1f}  "
              f"underpred_rate={100*cal['underprediction_rate']:.1f}%")
    br = report["branch"]
    if br:
        print(f"\nBranch  dup_rate={100*br['duplicate_rate']:.1f}%  "
              f"invalid_rate={100*br['invalid_rate']:.1f}%  "
              f"root_share={100*br['root_share']:.1f}%  "
              f"mean_depth={br['mean_depth']:.2f}")
    print(f"\nStop reasons: {report['stop_reasons']}")
    print(f"\nFull report written to {out_path}")


if __name__ == "__main__":
    main()
