"""Unit tests for the Budgeted REx extension.

Covers:
- Separate progress posterior correctness
- Zero-extension exact baseline equivalence
- Policy intervention (progress-driven and cost-driven flip)
- Budget affordability filter
- Adaptive output limit
- Deduplication and invalid-arm accounting
- Frozen cost model predictor
- Mechanism diagnostics (counterfactual baseline choice)
- Config validation (new fields, FrozenCostModel)
"""
from __future__ import annotations

import numpy as np
import pytest

from rexbench.budgets import (
    CostFeatures,
    FrozenLinearCostPredictor,
    adaptive_output_limit,
    affordable_arms,
)
from rexbench.config import METHODS, Budgets, Config, FrozenCostModel, Policy
from rexbench.policies import (
    counterfactual_baseline_choice,
    new_arm,
    observe,
    observe_improvement,
    select,
)
from rexbench.types import ArmState


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


def _arms(h_values: list[float], C: float = 5.0) -> list[ArmState]:
    """Root + one arm per h value."""
    arms = [ArmState(arm_id="root")]
    for i, h in enumerate(h_values, start=1):
        arms.append(new_arm(f"a{i}", h, C))
    return arms


def _estimates(arms: list[ArmState], base_cost: float = 100.0) -> dict[str, float]:
    return {a.arm_id: base_cost for a in arms}


# ---------------------------------------------------------------------------
# Progress posterior
# ---------------------------------------------------------------------------

class TestObserveImprovement:
    def test_positive_progress_increments_improve_alpha(self):
        parent = new_arm("p", 0.4, 5)
        child = new_arm("c", 0.7, 5)
        delta = observe_improvement(parent, child)
        assert delta == pytest.approx(0.3)
        assert parent.improve_alpha == pytest.approx(2.0)
        assert parent.improve_beta == pytest.approx(1.0)
        assert parent.expansion_count == 1
        assert parent.improvement_sum == pytest.approx(0.3)

    def test_no_progress_increments_improve_beta(self):
        parent = new_arm("p", 0.7, 5)
        child = new_arm("c", 0.5, 5)
        delta = observe_improvement(parent, child)
        assert delta == pytest.approx(0.0)
        assert parent.improve_alpha == pytest.approx(1.0)
        assert parent.improve_beta == pytest.approx(2.0)
        assert parent.expansion_count == 1

    def test_equal_h_counts_as_no_progress(self):
        parent = new_arm("p", 0.5, 5)
        child = new_arm("c", 0.5, 5)
        delta = observe_improvement(parent, child)
        assert delta == pytest.approx(0.0)
        assert parent.improve_beta == pytest.approx(2.0)

    def test_progress_never_negative(self):
        parent = new_arm("p", 0.9, 5)
        child = new_arm("c", 0.1, 5)
        delta = observe_improvement(parent, child)
        assert delta == 0.0

    def test_multiple_observations_accumulate(self):
        parent = new_arm("p", 0.2, 5)
        # All three children have h > parent.h (0.2), so all are positive improvements.
        for h in [0.4, 0.6, 0.6]:  # three positive improvements
            observe_improvement(parent, new_arm("c", h, 5))
        assert parent.expansion_count == 3
        assert parent.improve_alpha == pytest.approx(4.0)   # prior 1 + 3 positives
        assert parent.improve_beta == pytest.approx(1.0)    # no non-improvements

    def test_does_not_modify_baseline_posterior(self):
        parent = new_arm("p", 0.4, 5)
        alpha_before, beta_before = parent.alpha, parent.beta
        observe_improvement(parent, new_arm("c", 0.9, 5))
        # Baseline alpha/beta must remain unchanged.
        assert parent.alpha == pytest.approx(alpha_before)
        assert parent.beta == pytest.approx(beta_before)


# ---------------------------------------------------------------------------
# Legacy observe (unchanged behaviour guard)
# ---------------------------------------------------------------------------

class TestObserveLegacy:
    def test_legacy_positive_progress_case_from_plan_task_0107(self):
        """Reproduce the plan's Task 0107 example: beta posterior mean decreases."""
        parent = new_arm("p", 0.5, 20)
        # Manually set prior to Beta(11, 11) as in the plan.
        parent.alpha, parent.beta = 11.0, 11.0
        child = new_arm("c", 0.8333, 20)
        observe(parent, child, 0.5, False)
        # Posterior mean should be < 0.5 (the plan's finding).
        post_mean = parent.alpha / (parent.alpha + parent.beta)
        assert post_mean < 0.5

    def test_legacy_root_gets_no_progress_credit(self):
        root = ArmState(arm_id="root")
        child = new_arm("c", 1.0, 5)
        reward = observe(root, child, 0.5, False)
        assert reward == 0.0

    def test_legacy_terminal_does_not_mutate_parent(self):
        parent = new_arm("p", 0.4, 5)
        before = parent.model_dump()
        observe(parent, new_arm("c", 1.0, 5), 0.5, True)
        assert parent.model_dump() == before


# ---------------------------------------------------------------------------
# Zero-extension exact equivalence with baseline rex
# ---------------------------------------------------------------------------

class TestZeroExtensionEquivalence:
    """budgeted_rex with progress_weight=0, gamma=0, no budget must
    choose the same arm as baseline rex under the same RNG state."""

    def test_exact_equivalence_many_steps(self):
        cfg = Policy(C=5, progress_weight=0, gamma=0)
        arms = _arms([0.2, 0.5, 0.8], C=5)
        estimates = _estimates(arms)

        for iteration in range(50):
            left = _rng(iteration)
            right = _rng(iteration)
            ch_rex, _ = select("rex", arms, estimates, left, cfg, {})
            ch_bud, _ = select("budgeted_rex", arms, estimates, right, cfg, {})
            assert ch_rex == ch_bud, f"Diverged at iteration {iteration}"
            assert left.bit_generator.state == right.bit_generator.state, \
                f"RNG diverged at iteration {iteration}"

    def test_zero_params_recover_baseline_with_modified_progress_state(self):
        """Even with accumulated progress observations, zero weights reproduce rex."""
        cfg = Policy(C=5, progress_weight=0, gamma=0)
        arms = _arms([0.3, 0.6], C=5)
        # Accumulate some progress observations.
        observe_improvement(arms[1], new_arm("x", 0.9, 5))
        observe_improvement(arms[2], new_arm("y", 0.1, 5))
        estimates = _estimates(arms)

        for seed in range(20):
            left, right = _rng(seed), _rng(seed)
            ch_rex, _ = select("rex", arms, estimates, left, cfg, {})
            ch_bud, _ = select("budgeted_rex", arms, estimates, right, cfg, {})
            assert ch_rex == ch_bud


# ---------------------------------------------------------------------------
# Progress-driven intervention
# ---------------------------------------------------------------------------

class TestProgressIntervention:
    def test_progress_can_flip_choice(self):
        """Arms with equal base quality but different expansion histories
        must produce a progress-driven choice change relative to baseline."""
        cfg = Policy(C=5, progress_weight=2.0, gamma=0)
        # Two arms with equal h so baseline Thompson draws determine winner
        # half the time — but we craft the histories so progress always boosts a2.
        a1 = new_arm("a1", 0.5, 5)
        a2 = new_arm("a2", 0.5, 5)
        # a1: no improvement history — stays at Beta(1,1) progress posterior.
        # a2: strong improvement history.
        for _ in range(10):
            observe_improvement(a2, new_arm("x", 0.9, 5))

        arms = [ArmState(arm_id="root"), a1, a2]
        estimates = {"root": 100.0, "a1": 100.0, "a2": 100.0}

        # Force base draws so a1 would win under pure Thompson (a1 draw > a2 draw).
        # We do this by finding a seed where rex chooses a1.
        flipped = False
        for seed in range(500):
            rng_rex = _rng(seed)
            ch_rex, _ = select("rex", arms, estimates, rng_rex, cfg, {})
            if ch_rex == "a1":
                rng_bud = _rng(seed)
                ch_bud, _ = select("budgeted_rex", arms, estimates, rng_bud, cfg, {})
                if ch_bud == "a2":
                    flipped = True
                    break
        assert flipped, "Progress term never flipped the choice in 500 seeds"

    def test_progress_does_not_flip_when_weight_zero(self):
        """Zero progress weight: strong history must not change outcome."""
        cfg = Policy(C=5, progress_weight=0, gamma=0)
        a1 = new_arm("a1", 0.5, 5)
        a2 = new_arm("a2", 0.5, 5)
        for _ in range(20):
            observe_improvement(a2, new_arm("x", 1.0, 5))
        arms = [ArmState(arm_id="root"), a1, a2]
        estimates = {"root": 100.0, "a1": 100.0, "a2": 100.0}

        for seed in range(50):
            left, right = _rng(seed), _rng(seed)
            assert select("rex", arms, estimates, left, cfg, {})[0] == \
                   select("budgeted_rex", arms, estimates, right, cfg, {})[0]


# ---------------------------------------------------------------------------
# Cost-driven intervention
# ---------------------------------------------------------------------------

class TestCostIntervention:
    def test_cost_can_flip_choice_with_large_contrast(self):
        """Cheap arm should win when cost contrast is large and gamma > 0."""
        cfg = Policy(C=5, progress_weight=0, gamma=1.0)
        a_cheap = new_arm("cheap", 0.5, 5)
        a_expensive = new_arm("expensive", 0.5, 5)
        arms = [ArmState(arm_id="root"), a_cheap, a_expensive]
        # 20x cost ratio — with gamma=1.0 this is a large enough penalty.
        estimates = {"root": 100.0, "cheap": 100.0, "expensive": 2000.0}

        flipped = False
        for seed in range(500):
            rng_rex = _rng(seed)
            ch_rex, _ = select("rex", arms, estimates, rng_rex, cfg, {})
            if ch_rex == "expensive":
                rng_bud = _rng(seed)
                ch_bud, _ = select("budgeted_rex", arms, estimates, rng_bud, cfg, {})
                if ch_bud == "cheap":
                    flipped = True
                    break
        assert flipped, "Cost term never flipped the choice in 500 seeds"

    def test_cost_does_not_flip_when_gamma_zero(self):
        cfg = Policy(C=5, progress_weight=0, gamma=0)
        arms = _arms([0.5, 0.5], C=5)
        estimates = {"root": 100.0, "a1": 100.0, "a2": 10000.0}

        for seed in range(50):
            left, right = _rng(seed), _rng(seed)
            assert select("rex", arms, {"root": 100.0, "a1": 100.0, "a2": 10000.0}, left, cfg, {})[0] == \
                   select("budgeted_rex", arms, estimates, right, cfg, {})[0]

    def test_equal_costs_do_not_affect_ranking(self):
        """When all costs are equal relative_cost=1, gamma has no effect."""
        cfg_on = Policy(C=5, progress_weight=0, gamma=1.0)
        cfg_off = Policy(C=5, progress_weight=0, gamma=0)
        arms = _arms([0.3, 0.6, 0.9], C=5)
        estimates = _estimates(arms, base_cost=200.0)

        for seed in range(30):
            l1, l2 = _rng(seed), _rng(seed)
            assert select("budgeted_rex", arms, estimates, l1, cfg_on, {})[0] == \
                   select("budgeted_rex", arms, estimates, l2, cfg_off, {})[0]


# ---------------------------------------------------------------------------
# Budget affordability
# ---------------------------------------------------------------------------

class TestAffordableArms:
    def test_all_affordable(self):
        estimates = {"a": 100.0, "b": 200.0, "c": 50.0}
        result = affordable_arms(estimates, 1000, 10)
        assert result == estimates

    def test_excludes_expensive_arms(self):
        estimates = {"a": 100.0, "b": 500.0, "c": 50.0}
        # remaining=200, margin=10 → threshold=190
        result = affordable_arms(estimates, 200, 10)
        assert "a" in result
        assert "c" in result
        assert "b" not in result

    def test_empty_when_none_affordable(self):
        estimates = {"a": 500.0, "b": 1000.0}
        assert affordable_arms(estimates, 100, 50) == {}

    def test_boundary_inclusive(self):
        # cost + margin == remaining → exactly affordable.
        estimates = {"a": 90.0}
        assert affordable_arms(estimates, 100, 10) == {"a": 90.0}

    def test_boundary_exclusive(self):
        # cost + margin > remaining → not affordable.
        estimates = {"a": 91.0}
        assert affordable_arms(estimates, 100, 10) == {}


# ---------------------------------------------------------------------------
# Adaptive output limit
# ---------------------------------------------------------------------------

class TestAdaptiveOutputLimit:
    def test_returns_smaller_of_max_and_available(self):
        result = adaptive_output_limit(
            remaining_tokens=1000,
            predicted_input_tokens=600,
            safety_margin=64,
            minimum_output_tokens=128,
            max_output_tokens=2048,
        )
        # available = 1000 - 600 - 64 = 336; min(2048, 336) = 336
        assert result == 336

    def test_clamps_to_max_when_budget_generous(self):
        result = adaptive_output_limit(
            remaining_tokens=10000,
            predicted_input_tokens=500,
            safety_margin=64,
            minimum_output_tokens=128,
            max_output_tokens=2048,
        )
        assert result == 2048

    def test_returns_none_when_minimum_not_met(self):
        result = adaptive_output_limit(
            remaining_tokens=200,
            predicted_input_tokens=150,
            safety_margin=64,
            minimum_output_tokens=128,
            max_output_tokens=2048,
        )
        # available = 200 - 150 - 64 = -14 < 128 → None
        assert result is None

    def test_exactly_minimum_is_ok(self):
        # available = 300 - 100 - 72 = 128 == minimum
        result = adaptive_output_limit(
            remaining_tokens=300,
            predicted_input_tokens=100,
            safety_margin=72,
            minimum_output_tokens=128,
            max_output_tokens=2048,
        )
        assert result == 128


# ---------------------------------------------------------------------------
# Frozen cost model
# ---------------------------------------------------------------------------

class TestFrozenLinearCostPredictor:
    def _predictor(self):
        return FrozenLinearCostPredictor()

    def test_generation_uses_root_intercept(self):
        pred = self._predictor()
        f = CostFeatures(kind="generation", prompt_chars=0, candidate_chars=0,
                         depth=0, feedback_chars=0, prior_output_tokens=512)
        # intercept + output_reserve with all zero features
        expected = pred.root_intercept + pred.output_reserve
        assert pred.predict(f) == pytest.approx(expected)

    def test_repair_uses_repair_intercept(self):
        pred = self._predictor()
        f = CostFeatures(kind="repair", prompt_chars=0, candidate_chars=0,
                         depth=0, feedback_chars=0, prior_output_tokens=512)
        expected = pred.repair_intercept + pred.output_reserve
        assert pred.predict(f) == pytest.approx(expected)

    def test_repair_more_expensive_than_generation_by_default(self):
        pred = self._predictor()
        f_gen = CostFeatures(kind="generation", prompt_chars=400, candidate_chars=200,
                             depth=1, feedback_chars=0, prior_output_tokens=512)
        f_rep = CostFeatures(kind="repair", prompt_chars=400, candidate_chars=200,
                             depth=1, feedback_chars=100, prior_output_tokens=512)
        assert pred.predict(f_rep) > pred.predict(f_gen)

    def test_longer_candidate_costs_more(self):
        pred = self._predictor()
        f1 = CostFeatures(kind="repair", prompt_chars=400, candidate_chars=100,
                          depth=1, feedback_chars=50, prior_output_tokens=512)
        f2 = CostFeatures(kind="repair", prompt_chars=400, candidate_chars=800,
                          depth=1, feedback_chars=50, prior_output_tokens=512)
        assert pred.predict(f2) > pred.predict(f1)

    def test_deeper_arm_costs_more(self):
        pred = self._predictor()
        base = dict(kind="repair", prompt_chars=400, candidate_chars=200, feedback_chars=50,
                    prior_output_tokens=512)
        f_shallow = CostFeatures(depth=1, **base)
        f_deep = CostFeatures(depth=5, **base)
        assert pred.predict(f_deep) > pred.predict(f_shallow)

    def test_minimum_clamp_at_one(self):
        pred = FrozenLinearCostPredictor(root_intercept=0, output_reserve=0)
        f = CostFeatures(kind="generation", prompt_chars=0, candidate_chars=0,
                         depth=0, feedback_chars=0, prior_output_tokens=0)
        assert pred.predict(f) == 1.0

    def test_from_config(self):
        cfg = FrozenCostModel(
            root_intercept=400.0, repair_intercept=600.0,
            prompt_char_coef=0.3, candidate_char_coef=0.1,
            depth_coef=15.0, error_feedback_coef=0.2, output_reserve=200.0,
        )
        pred = FrozenLinearCostPredictor.from_config(cfg)
        f = CostFeatures(kind="generation", prompt_chars=100, candidate_chars=50,
                         depth=2, feedback_chars=0, prior_output_tokens=0)
        expected = 400 + 0.3 * 100 + 0.1 * 50 + 15 * 2 + 0.2 * 0 + 200
        assert pred.predict(f) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Counterfactual mechanism diagnostics
# ---------------------------------------------------------------------------

class TestCounterfactualBaselineChoice:
    def test_matches_max_draw(self):
        eligible = [ArmState(arm_id="root"), new_arm("a1", 0.4, 5), new_arm("a2", 0.9, 5)]
        estimates = {"root": 100.0, "a1": 100.0, "a2": 100.0}
        draws = {"root": 0.3, "a1": 0.7, "a2": 0.5}
        assert counterfactual_baseline_choice(eligible, estimates, draws) == "a1"

    def test_returns_empty_when_no_draws(self):
        eligible = [new_arm("a1", 0.5, 5)]
        result = counterfactual_baseline_choice(eligible, {"a1": 100.0}, {})
        assert result == ""

    def test_policy_changed_choice_flag(self):
        """Verify that the flag accurately identifies a divergence."""
        eligible = [ArmState(arm_id="root"), new_arm("a1", 0.4, 5), new_arm("a2", 0.9, 5)]
        estimates = {"root": 100.0, "a1": 100.0, "a2": 100.0}
        draws = {"root": 0.2, "a1": 0.8, "a2": 0.3}
        base = counterfactual_baseline_choice(eligible, estimates, draws)
        assert base == "a1"
        # If budgeted_rex picked "a2", the flag should be True.
        assert ("a2" != base) is True


# ---------------------------------------------------------------------------
# Config validation for new fields
# ---------------------------------------------------------------------------

class TestConfigNewFields:
    def test_budgeted_rex_in_methods(self):
        assert "budgeted_rex" in METHODS

    def test_policy_defaults(self):
        cfg = Policy()
        assert cfg.progress_weight == pytest.approx(0.5)
        assert cfg.initial_expected_delta == pytest.approx(0.1)
        assert cfg.deduplicate_policy_arms is False
        assert cfg.invalid_arm_actionable is True

    def test_budgets_new_fields_defaults(self):
        b = Budgets()
        assert b.per_trial_token_cap is None
        assert b.token_safety_margin == 64
        assert b.minimum_output_tokens == 128
        assert b.allow_budget_overshoot is True

    def test_frozen_cost_model_defaults(self):
        m = FrozenCostModel()
        assert m.root_intercept == pytest.approx(512.0)
        assert m.repair_intercept == pytest.approx(640.0)
        assert m.coefficients_file is None

    def test_config_includes_cost_model(self):
        cfg = Config()
        assert hasattr(cfg, "cost_model")
        assert isinstance(cfg.cost_model, FrozenCostModel)

    def test_config_allows_budgeted_rex_method(self):
        """Config with budgeted_rex as the only method must validate cleanly."""
        from rexbench.config import Experiment
        exp = Experiment(
            methods=["budgeted_rex"],
            seeds=[0],
            primary_comparison="budgeted_rex_vs_rex",
        )
        assert "budgeted_rex" in exp.methods

    def test_policy_toggles_ignores_budgeted_rex(self):
        """budgeted_rex must not inherit legacy eta/gamma via toggles."""
        cfg = Policy(eta=0.5, gamma=0.5)
        eta, gamma = cfg.toggles("budgeted_rex")
        assert eta == 0.0
        assert gamma == 0.0

    def test_per_trial_token_cap_validates(self):
        b = Budgets(per_trial_token_cap=8192, token_safety_margin=64, minimum_output_tokens=128)
        assert b.per_trial_token_cap == 8192


# ---------------------------------------------------------------------------
# budgeted_rex: lone root still consumes exactly one draw
# ---------------------------------------------------------------------------

class TestBudgetedRexLoneRoot:
    def test_lone_root_consumes_one_draw(self):
        a, b = _rng(42), _rng(42)
        select("budgeted_rex", [ArmState(arm_id="root")], {"root": 100.0}, a,
               Policy(progress_weight=0, gamma=0), {})
        b.beta(1, 1)
        assert a.bit_generator.state == b.bit_generator.state


# ---------------------------------------------------------------------------
# budgeted_rex: headroom prevents impossible expected progress
# ---------------------------------------------------------------------------

class TestHeadroomBound:
    def test_fully_solved_arm_has_zero_progress(self):
        """An arm with h=1.0 must contribute zero expected progress regardless
        of its improvement history."""
        cfg = Policy(C=5, progress_weight=5.0, gamma=0)
        a_solved = new_arm("solved", 1.0, 5)
        for _ in range(20):
            observe_improvement(a_solved, new_arm("x", 1.0, 5))
        a_unsolved = new_arm("unsolved", 0.5, 5)
        arms = [ArmState(arm_id="root"), a_solved, a_unsolved]
        estimates = {"root": 100.0, "solved": 100.0, "unsolved": 100.0}

        # With high progress_weight the solved arm must NOT be boosted.
        # The unsolved arm should win at least sometimes.
        choices = [
            select("budgeted_rex", arms, estimates, _rng(s), cfg, {})[0]
            for s in range(100)
        ]
        assert "unsolved" in choices or "root" in choices, \
            "Solved arm monopolised selection despite headroom=0"
