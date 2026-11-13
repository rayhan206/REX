"""Pure selectors. I/O, prompts and persistence belong to the controller.

Budgeted REx (``budgeted_rex``) is a new method that sits alongside the
archived methods (``rex``, ``rex_progress_cost``, etc.) without touching them.
Archived methods are reproduced exactly to maintain comparability with the
sealed pilot results.

Design principles for ``budgeted_rex``
--------------------------------------
1. ``progress_weight=0`` and ``gamma=0`` reproduce baseline ``rex`` selection
   when fed the same base Thompson draws.
2. Progress is tracked in a *separate* Beta posterior (``improve_alpha`` /
   ``improve_beta``) so it never corrupts the baseline arm quality posterior.
3. Expected progress uses ``headroom`` to prevent impossible gains on
   near-solved arms.
4. Relative cost normalises by the cheapest admissible arm in the current
   decision so the absolute token scale does not interact with
   ``progress_weight``.
5. The counterfactual baseline choice under the same draws is stored in every
   decision record for mechanism analysis (PIR).
"""
from __future__ import annotations

from typing import Any

import numpy as np

from ..config import Policy
from ..types import ArmState


# ---------------------------------------------------------------------------
# Arm construction helpers
# ---------------------------------------------------------------------------

def new_arm(arm_id: str, h: float, C: float, **kwargs) -> ArmState:
    return ArmState(arm_id=arm_id, h=h, alpha=1 + C * h, beta=1 + C * (1 - h), **kwargs)


# ---------------------------------------------------------------------------
# Posterior update functions
# ---------------------------------------------------------------------------

def observe(parent: ArmState, child: ArmState, eta: float, terminal: bool) -> float:
    """Legacy fractional-shaping update used by all archived methods.

    Upstream exits before posterior mutation on terminal success.  Child
    evidence is still saved.
    """
    reward = (
        1. if terminal
        else eta * max(0., child.h - parent.h) if parent.arm_id != "root"
        else 0.
    )
    if not terminal:
        parent.alpha += reward
        parent.beta += 1 - reward
        parent.pulls += 1
    return reward


def observe_improvement(parent: ArmState, child: ArmState) -> float:
    """Separate progress posterior update used only by ``budgeted_rex``.

    Tracks whether expanding ``parent`` produces positive visible progress.
    Keeps the progress signal strictly positive (no pretending that partial
    improvement is a mostly-failed Bernoulli outcome as in the legacy update).

    Returns the raw delta for event logging.
    """
    delta = max(0.0, child.h - parent.h)
    parent.expansion_count += 1
    parent.improvement_sum += delta

    if delta > 0.0:
        parent.improve_alpha += 1.0
    else:
        parent.improve_beta += 1.0

    return delta


# ---------------------------------------------------------------------------
# Selector
# ---------------------------------------------------------------------------

def select(
    method: str,
    arms: list[ArmState],
    estimates: dict[str, float],
    rng: np.random.Generator,
    cfg: Policy,
    policy_state: dict[str, Any],
) -> tuple[str, dict]:
    eligible = [a for a in arms if a.arm_id in estimates]
    if not eligible:
        raise ValueError("No admissible action")

    scores: dict[str, float] = {}
    draws: dict[str, float] = {}

    if method == "resample":
        chosen = "root"

    elif method == "greedy":
        scores = {
            a.arm_id: cfg.greedy_root_score if a.arm_id == "root" else a.h
            for a in eligible
        }
        best = max(scores.values())
        ties = [a.arm_id for a in eligible if scores[a.arm_id] == best]
        chosen = str(rng.choice(ties))

    elif method == "bfs":
        chosen = next((a.arm_id for a in eligible if a.pulls < cfg.bfs_branching), "")

    elif method == "fixed_width":
        endpoints = policy_state.get("endpoints", [])
        if len(endpoints) < cfg.fixed_width:
            chosen = "root"
        else:
            cursor = policy_state.get("cursor", 0)
            chosen = endpoints[cursor % cfg.fixed_width]

    elif method == "budgeted_rex":
        chosen, scores, draws = _select_budgeted_rex(eligible, estimates, rng, cfg)

    else:
        # Archived methods: resample / rex / rex_progress / rex_cost / rex_progress_cost.
        _, gamma = cfg.toggles(method)
        for a in eligible:
            draws[a.arm_id] = float(rng.beta(a.alpha, a.beta))
            scores[a.arm_id] = draws[a.arm_id] / max(1., estimates[a.arm_id]) ** gamma
        chosen = max(scores, key=lambda x: scores[x])

    if chosen not in estimates:
        raise ValueError("Required structural action exceeds context")

    return chosen, {
        "draws": draws,
        "priorities": scores,
        "estimates": estimates.copy(),
        "chosen": chosen,
        "parameters": {a.arm_id: [a.alpha, a.beta, a.pulls] for a in eligible},
    }


def _select_budgeted_rex(
    eligible: list[ArmState],
    estimates: dict[str, float],
    rng: np.random.Generator,
    cfg: Policy,
) -> tuple[str, dict[str, float], dict[str, float]]:
    """Core budgeted_rex scoring logic.

    Separated for clarity and to allow targeted unit testing without the full
    ``select`` dispatch layer.

    Score for arm ``a``:
        base_draw   ~ Beta(a.alpha, a.beta)
        p_improve   = a.improve_alpha / (a.improve_alpha + a.improve_beta)
        mean_delta  = a.improvement_sum / a.expansion_count  (or cfg.initial_expected_delta)
        headroom    = 1.0 - a.h
        exp_progress = p_improve * min(headroom, mean_delta)
        utility     = base_draw + cfg.progress_weight * exp_progress
        rel_cost    = estimates[a] / min_estimate
        score       = utility / rel_cost ** cfg.gamma

    When progress_weight=0 and gamma=0, scores collapse to base Thompson draws
    (utility = base_draw, rel_cost = 1 for all), reproducing baseline rex
    selection given the same rng state.
    """
    min_est = min(estimates[a.arm_id] for a in eligible)

    draws: dict[str, float] = {}
    scores: dict[str, float] = {}

    for a in eligible:
        base_draw = float(rng.beta(a.alpha, a.beta))
        draws[a.arm_id] = base_draw

        p_improve = a.improve_alpha / (a.improve_alpha + a.improve_beta)
        mean_delta = (
            a.improvement_sum / a.expansion_count
            if a.expansion_count > 0
            else cfg.initial_expected_delta
        )
        headroom = max(0.0, 1.0 - a.h)
        expected_progress = p_improve * min(headroom, mean_delta)

        utility = base_draw + cfg.progress_weight * expected_progress

        relative_cost = estimates[a.arm_id] / max(1.0, min_est)
        score = utility / max(1e-9, relative_cost ** cfg.gamma)

        scores[a.arm_id] = score

    chosen = max(scores, key=lambda x: scores[x])

    return chosen, scores, draws


# ---------------------------------------------------------------------------
# Advance helpers
# ---------------------------------------------------------------------------

def advance(method: str, parent: ArmState, child: ArmState, state: dict, cfg: Policy):
    if method == "fixed_width":
        endpoints = list(state.get("endpoints", []))
        if len(endpoints) < cfg.fixed_width:
            endpoints.append(child.arm_id)
        else:
            cursor = state.get("cursor", 0)
            endpoints[cursor % cfg.fixed_width] = child.arm_id
            state["cursor"] = cursor + 1
        state["endpoints"] = endpoints


# ---------------------------------------------------------------------------
# Mechanism diagnostics helper
# ---------------------------------------------------------------------------

def counterfactual_baseline_choice(
    eligible: list[ArmState],
    estimates: dict[str, float],
    base_draws: dict[str, float],
) -> str:
    """Compute what baseline rex would have chosen given the same Thompson draws.

    Used to populate the ``baseline_choice`` and ``policy_changed_choice``
    fields in every ``budgeted_rex`` decision event without consuming extra RNG
    state.
    """
    scores = {a.arm_id: base_draws[a.arm_id] for a in eligible if a.arm_id in base_draws}
    return max(scores, key=lambda x: scores[x]) if scores else ""
