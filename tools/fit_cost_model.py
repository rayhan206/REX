#!/usr/bin/env python3
"""Fit the frozen linear cost model on development-only traces.

Usage::

    uv run python tools/fit_cost_model.py \\
        --run-dir runs/apps_development_pilot_local \\
        --out configs/cost_model_dev_v1.json

The script reads all SQLite databases under ``--run-dir`` for the methods
specified in ``--methods`` (default: all), extracts per-request features and
actual token counts, fits a non-negative linear model with separate intercepts
for generation and repair requests, then writes the frozen coefficients to
``--out`` as JSON.

The resulting JSON file is consumed by a development configuration via
``cost_model.coefficients_file``.  After fitting, copy the
``coefficients_hash`` printed to stdout into that YAML field to enable hash
validation on load.

This script MUST be run only on development tasks.  Never refit on or using
held-out task data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    import numpy as np
except ImportError:
    raise SystemExit("numpy is required: uv pip install numpy")

from rexbench.budgets import request_cost_features
from rexbench.store import Store
from rexbench.util import fingerprint


def _iter_requests(db_path: Path, methods: set[str] | None = None) -> list[dict]:
    """Yield per-request feature rows from a trial SQLite database."""
    store = Store(db_path.parent, readonly=True)
    rows = []
    try:
        for row in store.db.execute(
            "SELECT r.id, r.request_blob, r.context_blob, g.result_blob, "
            "p.method FROM requests r "
            "JOIN generations g ON g.request_id=r.id "
            "JOIN planned_trials p ON p.id=r.trial_id "
            "WHERE r.state='committed'"
        ):
            if methods is not None and row["method"] not in methods:
                continue
            req = store.object(row["request_blob"])
            ctx = store.object(row["context_blob"])
            result = store.object(row["result_blob"])
            input_tokens = result.get("input_tokens")
            output_tokens = result.get("output_tokens")
            if input_tokens is None or output_tokens is None:
                continue
            parent_id = ctx.get("parent_id", "root")
            arm_row = store.db.execute(
                "SELECT state FROM arms WHERE trial_id=(SELECT trial_id FROM requests WHERE id=?) "
                "AND arm_id=?", (row["id"], parent_id)
            ).fetchone()
            arm = json.loads(arm_row["state"]) if arm_row else {}
            candidate_code = ""
            if arm.get("code_hash"):
                candidate_code = store.get(arm["code_hash"]).decode("utf-8", errors="replace")
            features = request_cost_features(
                prompt=ctx.get("prompt", ""),
                candidate_code=candidate_code,
                depth=int(arm.get("depth", 0)),
                kind=req.get("kind", "generation"),
                prior_output_tokens=0.0,
            )
            rows.append({
                "method": row["method"],
                "kind": features.kind,
                "prompt_chars": features.prompt_chars,
                "candidate_chars": features.candidate_chars,
                "depth": features.depth,
                "feedback_chars": features.feedback_chars,
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "total_tokens": int(input_tokens) + int(output_tokens),
            })
    finally:
        store.close()
    return rows


def fit(rows: list[dict]) -> dict:
    """Fit non-negative least squares on request features → total tokens."""
    if not rows:
        raise ValueError("No committed requests found; nothing to fit.")

    n = len(rows)
    # Build feature matrix: [gen_intercept, rep_intercept, prompt_chars,
    #                         candidate_chars, depth, feedback_chars, output_reserve]
    # We model total_tokens = intercept_kind + b * features + output_reserve.
    # For simplicity output_reserve is estimated as the mean output_tokens.
    output_mean = float(np.mean([r["output_tokens"] for r in rows]))

    X = []
    y = []
    for r in rows:
        is_gen = 1.0 if r["kind"] == "generation" else 0.0
        is_rep = 1.0 - is_gen
        X.append([
            is_gen,         # root_intercept
            is_rep,         # repair_intercept
            r["prompt_chars"],
            r["candidate_chars"],
            r["depth"],
            r["feedback_chars"],
        ])
        y.append(r["total_tokens"] - output_mean)  # remove output_reserve from target

    X_arr = np.array(X, dtype=float)
    y_arr = np.array(y, dtype=float)

    # Cyclic coordinate descent for non-negative least squares.  A plain OLS
    # fit followed by clipping is not NNLS and produced badly miscalibrated
    # coefficients in the original implementation.
    scales = np.linalg.norm(X_arr, axis=0)
    scales[scales == 0.0] = 1.0
    design = X_arr / scales
    coeffs_scaled = np.zeros(design.shape[1], dtype=float)
    prediction = design @ coeffs_scaled
    for _ in range(10_000):
        largest_change = 0.0
        for column in range(design.shape[1]):
            values = design[:, column]
            denominator = float(values @ values)
            if denominator == 0.0:
                continue
            residual_without_column = y_arr - prediction + values * coeffs_scaled[column]
            updated = max(0.0, float(values @ residual_without_column) / denominator)
            change = updated - coeffs_scaled[column]
            if change:
                prediction += values * change
                coeffs_scaled[column] = updated
                largest_change = max(largest_change, abs(change))
        if largest_change < 1e-9:
            break
    coeffs = coeffs_scaled / scales
    root_intercept = float(coeffs[0])
    repair_intercept = float(coeffs[1])
    prompt_char_coef = float(coeffs[2])
    candidate_char_coef = float(coeffs[3])
    depth_coef = float(coeffs[4])
    error_feedback_coef = float(coeffs[5])
    output_reserve = float(max(0.0, output_mean))

    # Evaluation metrics.
    preds = X_arr @ coeffs + output_reserve
    errors = preds - np.array([r["total_tokens"] for r in rows])
    mae = float(np.mean(np.abs(errors)))
    actuals = np.array([r["total_tokens"] for r in rows], dtype=float)
    mape = float(np.mean(np.abs(errors) / np.maximum(1.0, actuals)) * 100)

    return {
        "schema_version": 1,
        "description": (
            "Frozen linear cost model fitted on development-only traces. "
            "DO NOT refit using held-out data."
        ),
        "root_intercept": root_intercept,
        "repair_intercept": repair_intercept,
        "prompt_char_coef": prompt_char_coef,
        "candidate_char_coef": candidate_char_coef,
        "depth_coef": depth_coef,
        "error_feedback_coef": error_feedback_coef,
        "output_reserve": output_reserve,
        "fit_mae_tokens": mae,
        "fit_mape_pct": mape,
        "fit_n_requests": n,
        "fit_source": "development_only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", required=True, type=Path, help="Directory containing trial run folders")
    parser.add_argument("--out", required=True, type=Path, help="Output JSON path for frozen coefficients")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Restrict to these method names (default: all)")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")

    # Each run uses one append-only state.sqlite plus content-addressed artifacts.
    db_paths = sorted(run_dir.rglob("state.sqlite"))
    if not db_paths:
        raise SystemExit(f"No state.sqlite files found under {run_dir}")

    all_rows: list[dict] = []
    methods = set(args.methods) if args.methods else None
    for db_path in db_paths:
        all_rows.extend(_iter_requests(db_path, methods))

    print(f"Collected {len(all_rows)} committed requests from {len(db_paths)} databases.")

    if not all_rows:
        raise SystemExit("No committed requests with token usage found.")

    coefficients = fit(all_rows)
    coefficients["fit_methods"] = sorted({row["method"] for row in all_rows})
    coefficients["feature_definition"] = "request_cost_features_v1"
    coefficients["fit_databases"] = [f"{path.parent.name}/{path.name}" for path in db_paths]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coefficients, indent=2), encoding="utf-8")

    h = fingerprint(json.loads(out_path.read_text(encoding="utf-8")))
    print(f"Written to {out_path}")
    print(f"MAE: {coefficients['fit_mae_tokens']:.2f} tokens")
    print(f"MAPE: {coefficients['fit_mape_pct']:.2f}%")
    print(f"\ncoefficients_hash (paste into the matching development config):\n  {h}")


if __name__ == "__main__":
    main()
