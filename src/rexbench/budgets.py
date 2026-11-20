from __future__ import annotations

from dataclasses import dataclass

from .types import GenerationResult, Ledger


@dataclass
class CostEstimator:
    chars_per_token: float = 4.
    output_tokens: float = 512.

    def estimate(self, prompt: str) -> float:
        return max(1., len(prompt) / self.chars_per_token + self.output_tokens)

    def observe(self, prompt: str, result: GenerationResult):
        if result.input_tokens is not None and result.input_tokens > 0:
            ratio = min(10., max(1., len(prompt) / result.input_tokens))
            self.chars_per_token = .8 * self.chars_per_token + .2 * ratio
        if result.output_tokens is not None:
            self.output_tokens = .8 * self.output_tokens + .2 * result.output_tokens


@dataclass
class CostFeatures:
    """Features extracted for each pending arm request before selection."""
    kind: str          # "generation" or "repair"
    prompt_chars: int
    candidate_chars: int
    depth: int
    feedback_chars: int
    prior_output_tokens: float  # current estimator output EWMA


def request_cost_features(
    prompt: str,
    candidate_code: str,
    depth: int,
    kind: str,
    prior_output_tokens: float,
) -> CostFeatures:
    """Extract the frozen predictor inputs used during fitting and search.

    ``feedback_chars`` is measured from the stable repair-prompt markers.  Keeping
    this extraction in one place prevents the fitted model and the live selector
    from silently using different feature definitions.
    """
    marker = "One failed repair test:\n"
    end_marker = "\n\nFix the program"
    feedback_chars = 0
    if marker in prompt:
        feedback = prompt.split(marker, 1)[1]
        feedback = feedback.split(end_marker, 1)[0]
        feedback_chars = len(feedback)
    # Candidate and feedback text also appear inside the full prompt.  Remove
    # them from the structural prompt feature so the three length features do
    # not double-count the same characters during fitting or prediction.
    structural_prompt_chars = max(0, len(prompt) - len(candidate_code) - feedback_chars)
    return CostFeatures(
        kind=kind,
        prompt_chars=structural_prompt_chars,
        candidate_chars=len(candidate_code),
        depth=depth,
        feedback_chars=feedback_chars,
        prior_output_tokens=prior_output_tokens,
    )


@dataclass(frozen=True)
class FrozenLinearCostPredictor:
    """Development-fitted frozen linear cost model.

    Predicts total tokens (input + output) for a pending request.  Separate
    intercepts for root generation and repair allow the predictor to model the
    structural difference between prompt types without overfitting.

    All coefficients are fitted once on development-only traces and kept fixed
    for held-out evaluation.  The predictor is used by the budgeted_rex selector
    to rank arms by cost and to enforce the per-trial token budget soft cap.
    """
    root_intercept: float = 512.0
    repair_intercept: float = 640.0
    prompt_char_coef: float = 0.25
    candidate_char_coef: float = 0.10
    depth_coef: float = 20.0
    error_feedback_coef: float = 0.15
    output_reserve: float = 256.0

    def predict(self, features: CostFeatures) -> float:
        intercept = (
            self.root_intercept if features.kind == "generation" else self.repair_intercept
        )
        return max(
            1.0,
            intercept
            + self.prompt_char_coef * features.prompt_chars
            + self.candidate_char_coef * features.candidate_chars
            + self.depth_coef * features.depth
            + self.error_feedback_coef * features.feedback_chars
            + self.output_reserve,
        )

    @classmethod
    def from_config(cls, cfg) -> "FrozenLinearCostPredictor":
        """Construct from a ``FrozenCostModel`` config object."""
        return cls(
            root_intercept=cfg.root_intercept,
            repair_intercept=cfg.repair_intercept,
            prompt_char_coef=cfg.prompt_char_coef,
            candidate_char_coef=cfg.candidate_char_coef,
            depth_coef=cfg.depth_coef,
            error_feedback_coef=cfg.error_feedback_coef,
            output_reserve=cfg.output_reserve,
        )


def affordable_arms(
    estimates: dict[str, float],
    remaining_tokens: int,
    safety_margin: int,
) -> dict[str, float]:
    """Return the subset of arm estimates that fit within the remaining token budget.

    An arm is considered affordable when its estimated cost plus the safety margin
    does not exceed the remaining budget.  When no arms are affordable the caller
    should stop the trial with reason ``"token_budget_exhausted"``.
    """
    return {
        arm_id: cost
        for arm_id, cost in estimates.items()
        if cost + safety_margin <= remaining_tokens
    }


def adaptive_output_limit(
    remaining_tokens: int,
    predicted_input_tokens: float,
    safety_margin: int,
    minimum_output_tokens: int,
    max_output_tokens: int,
) -> int | None:
    """Compute a tighter output limit that respects the remaining token budget.

    Returns ``None`` when even the minimum output allowance does not fit, which
    signals the caller to stop.  Otherwise returns the clamped output limit.
    """
    available = remaining_tokens - int(predicted_input_tokens) - safety_margin
    if available < minimum_output_tokens:
        return None
    return min(max_output_tokens, available)


def charge(ledger: Ledger, result: GenerationResult):
    if result.input_tokens is None or result.output_tokens is None:
        ledger.unknown_usage += 1
    ledger.prompt_tokens += result.input_tokens or 0
    ledger.output_tokens += result.output_tokens or 0
    ledger.model_seconds += result.latency_seconds
    ledger.reservations = max(0, ledger.reservations - 1)


def checkpoint_snapshot(snapshots: list[dict], axis: str, budget: int, stop: str | None) -> tuple[dict | None, bool]:
    """Select only completed prefixes. Absorbing repair success extends to all future budgets."""
    if axis not in ("calls", "tokens"):
        raise ValueError("Unknown checkpoint axis")
    if axis == "tokens" and any(s["tokens"] is None for s in snapshots):
        return None, True
    valid = [s for s in snapshots if s[axis] <= budget]
    chosen = valid[-1] if valid else None
    covered = bool(snapshots) and (snapshots[-1][axis] >= budget or stop == "visible_success")
    return chosen, not covered
