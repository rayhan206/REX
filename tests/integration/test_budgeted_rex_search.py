import json

import pytest

from conftest import initialize
from rexbench.backends.mock import MockBackend
from rexbench.judge.mock import MockJudge
from rexbench.prompts import prepare
from rexbench.search import run_trial


def _one_step_config(config):
    cfg = config.model_copy(deep=True)
    cfg.budgets.call_checkpoints = [1]
    cfg.budgets.coverage_extension_call_ceiling = 1
    cfg.budgets.token_checkpoints = []
    cfg.budgets.per_trial_token_cap = None
    cfg.cost_model.root_intercept = 100.0
    cfg.cost_model.repair_intercept = 200.0
    cfg.cost_model.prompt_char_coef = 1.0
    cfg.cost_model.candidate_char_coef = 0.0
    cfg.cost_model.depth_coef = 0.0
    cfg.cost_model.error_feedback_coef = 0.0
    cfg.cost_model.output_reserve = 0.0
    return cfg


def _estimated_tokens(store):
    row = store.db.execute(
        "SELECT payload FROM events WHERE kind='step_committed' ORDER BY number LIMIT 1"
    ).fetchone()
    return json.loads(row["payload"])["estimated_tokens"]


def test_budgeted_search_uses_frozen_cost_model_but_rex_keeps_legacy_estimator(
    tmp_path, task, config
):
    cfg = _one_step_config(config)
    backend_script = [{"text": "print(0)", "input_tokens": 40, "output_tokens": 20}]
    judge = MockJudge({"print(0)": ["PASS", "WRONG_ANSWER", "WRONG_ANSWER"]})

    budget_job = {"id": "budget", "task_id": "tiny", "method": "budgeted_rex", "seed": 0}
    budget_store = initialize(tmp_path / "budget", budget_job)
    run_trial(budget_store, task, budget_job, cfg, MockBackend(backend_script), judge, "e")

    root = budget_store.load("budget").arms[0]
    prompt, _ = prepare(task, root, "", 0, 0)
    assert _estimated_tokens(budget_store) == pytest.approx(100.0 + len(prompt))

    rex_job = {"id": "rex", "task_id": "tiny", "method": "rex", "seed": 0}
    rex_store = initialize(tmp_path / "rex", rex_job)
    run_trial(rex_store, task, rex_job, cfg, MockBackend(backend_script), judge, "e")
    assert _estimated_tokens(rex_store) == pytest.approx(512.0 + len(prompt) / 4.0)

    budget_store.close()
    rex_store.close()
