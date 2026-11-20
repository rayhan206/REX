#!/usr/bin/env python3
"""Deterministic synthetic mechanism demo; not a benchmark result."""
from __future__ import annotations

import json

import numpy as np

from rexbench.config import Policy
from rexbench.policies import new_arm, observe_improvement, select
from rexbench.types import ArmState


def main() -> None:
    policy = Policy(C=5.0, progress_weight=0.5, gamma=1.0)
    root = ArmState(arm_id="root")
    cheap_progressing = new_arm("cheap_progressing", 0.50, policy.C)
    expensive_stalled = new_arm("expensive_stalled", 0.55, policy.C)

    for index in range(4):
        observe_improvement(
            cheap_progressing,
            new_arm(f"progress-{index}", min(0.95, 0.65 + index * 0.05), policy.C),
        )
        observe_improvement(
            expensive_stalled,
            new_arm(f"stalled-{index}", 0.50, policy.C),
        )

    arms = [root, cheap_progressing, expensive_stalled]
    estimates = {"root": 700.0, "cheap_progressing": 600.0, "expensive_stalled": 1500.0}

    for seed in range(10_000):
        methods = [
            "rex",
            "budgeted_rex_control",
            "budgeted_rex_progress",
            "budgeted_rex_cost",
            "budgeted_rex",
        ]
        decisions = {
            method: select(method, arms, estimates, np.random.default_rng(seed), policy, {})
            for method in methods
        }
        rex_choice, rex = decisions["rex"]
        extension_choice, extension = decisions["budgeted_rex"]
        if rex_choice == "expensive_stalled" and extension_choice == "cheap_progressing":
            choices = {method: decision[0] for method, decision in decisions.items()}
            print(json.dumps({
                "scope": "synthetic mechanism demonstration; not empirical benchmark evidence",
                "comparison": "rex_and_matched_ablations_vs_budgeted_rex",
                "seed": seed,
                "same_thompson_draws": rex["draws"] == extension["draws"],
                "choices": choices,
                "predicted_next_tokens": {
                    method: estimates[choice] for method, choice in choices.items()
                },
                "predicted_next_token_difference": estimates[extension_choice] - estimates[rex_choice],
                "thompson_draws": extension["draws"],
                "budgeted_priorities": extension["priorities"],
            }, indent=2))
            return
    raise SystemExit("No intervention found in the deterministic seed search.")


if __name__ == "__main__":
    main()
