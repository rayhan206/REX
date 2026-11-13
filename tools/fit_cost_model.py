#!/usr/bin/env python3
"""Fit the frozen linear cost model on development-only traces.

Usage::

    uv run python tools/fit_cost_model.py \\
        --run-dir runs/apps_development_pilot_local \\
        --out configs/cost_model_dev_v1.json

The script reads all SQLite databases under ``--run-dir`` for the methods
specified in ``--methods`` (default: all), extracts per-request features and
actual token counts, fits a simple OLS linear model with separate intercepts
for generation and repair requests, then writes the frozen coefficients to
``--out`` as JSON.

The resulting JSON file is consumed by ``configs/budgeted_rex_dev_v1.yaml``
via ``cost_model.coefficients_file``.  After fitting, copy the
``coefficients_hash`` printed to stdout into that YAML field to enable hash
validation on load.

This script MUST be run only on development tasks.  Never refit on or using
held-out task data.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

try:
    import numpy as np
except ImportError:
    raise SystemExit("numpy is required: uv pip install numpy")

from rexbench.util import fingerprint


def _iter_requests(db_path: Path) -> list[dict]:
    """Yield per-request feature rows from a trial SQLite database."""
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    rows = []
    try:
        # Join requests with their stored context blobs.
        for row in con.execute(
            "SELECT r.id, r.state, r.request_blob, r.context_blob, r.response_blob "
            "FROM requests r WHERE r.state = 'committed'"
        ):
            import json as _json
            req_blob = con.execute(
                "SELECT value FROM blobs WHERE id=?", (row["request_blob"],)
            ).fetchone()
            ctx_blob = con.execute(
                "SELECT value FROM blobs WHERE id=?", (row["context_blob"],)
            ).fetchone()
            if req_blob is None or ctx_blob is None:
                continue
            req = _json.loads(req_blob["value"])
            ctx = _json.loads(ctx_blob["value"])
            # Look up actual token usage from the generation result.
            if row["response_blob"]:
                res_blob = con.execute(
                    "SELECT value FROM blobs WHERE id=?", (row["response_blob"],)
                ).fetchone()
                if res_blob:
                    res = _json.loads(res_blob["value"])
                    input_tokens = res.get("input_tokens")
                    output_tokens = res.get("output_tokens")
                    if input_tokens is not None and output_tokens is not None:
                        prompt: str = ctx.get("prompt", "")
                        parent_id: str = ctx.get("parent_id", "root")
                        # Candidate code length from the prompt (heuristic: everything
                        # after the last "```python" header is candidate code).
                        candidate_chars = 0
                        if "```python" in prompt:
                            candidate_chars = len(prompt.split("```python")[-1])
                        rows.append({
                            "kind": req.get("kind", "generation"),
                            "prompt_chars": len(prompt),
                            "candidate_chars": candidate_chars,
                            "depth": 0 if parent_id == "root" else 1,
                            "feedback_chars": 0,
                            "input_tokens": int(input_tokens),
                            "output_tokens": int(output_tokens),
                            "total_tokens": int(input_tokens) + int(output_tokens),
                        })
    finally:
        con.close()
    return rows


def fit(rows: list[dict]) -> dict:
    """Fit OLS on request features → total tokens, separate intercepts."""
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

    # OLS: (X^T X)^{-1} X^T y
    try:
        coeffs, _residuals, _rank, _sv = np.linalg.lstsq(X_arr, y_arr, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise ValueError(f"OLS failed: {exc}") from exc

    root_intercept = float(max(0.0, coeffs[0]))
    repair_intercept = float(max(0.0, coeffs[1]))
    prompt_char_coef = float(max(0.0, coeffs[2]))
    candidate_char_coef = float(max(0.0, coeffs[3]))
    depth_coef = float(max(0.0, coeffs[4]))
    error_feedback_coef = float(max(0.0, coeffs[5]))
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

    # Collect all state.db files under run_dir.
    db_paths = sorted(run_dir.rglob("state.db"))
    if not db_paths:
        raise SystemExit(f"No state.db files found under {run_dir}")

    all_rows: list[dict] = []
    for db_path in db_paths:
        all_rows.extend(_iter_requests(db_path))

    print(f"Collected {len(all_rows)} committed requests from {len(db_paths)} databases.")

    if not all_rows:
        raise SystemExit("No committed requests with token usage found.")

    coefficients = fit(all_rows)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(coefficients, indent=2), encoding="utf-8")

    h = fingerprint(json.loads(out_path.read_text(encoding="utf-8")))
    print(f"Written to {out_path}")
    print(f"MAE: {coefficients['fit_mae_tokens']:.2f} tokens")
    print(f"MAPE: {coefficients['fit_mape_pct']:.2f}%")
    print(f"\ncoefficients_hash (paste into budgeted_rex_dev_v1.yaml):\n  {h}")


if __name__ == "__main__":
    main()
