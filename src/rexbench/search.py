from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np

from .backends.base import RejectedBeforeInference
from .budgets import (CostEstimator, FrozenLinearCostPredictor, adaptive_output_limit,
                      affordable_arms, charge, request_cost_features)
from .config import BUDGETED_METHODS, Config
from .policies import (advance, counterfactual_baseline_choice, counterfactual_component_choices,
                       new_arm, observe, observe_improvement, select)
from .prompts import admissible, extract_code, prepare
from .store import Store
from .types import (EvaluationResult, GenerationRequest, Message, RepairTask, SearchCheckpoint,
                    Status, TestOutcome)
from .util import canonical, now, seed_for, sha


def run_trial(store: Store, task: RepairTask, job: dict, config: Config, backend, judge,
              experiment_id: str, campaign_check=lambda: None, fault=lambda boundary: None):
    """Serial, repair-only state machine. A persisted response is never regenerated."""
    trial_id, method, seed = job["id"], job["method"], job["seed"]
    state = store.load(trial_id)
    rng = np.random.default_rng(seed)
    if state is None:
        state = SearchCheckpoint(trial_id=trial_id, rng_state=dict(rng.bit_generator.state))
        with store.transaction():
            store.checkpoint(state, "running")
    else:
        rng.bit_generator.state = state.rng_state
    if state.stop_reason:
        return state

    def pause(reason, status="paused"):
        with store.transaction():
            store.event(trial_id, reason, {})
            store.db.execute("UPDATE trial_states SET status=?,stop_reason=? WHERE id=?", (status, reason, trial_id))

    while not state.stop_reason:
        step_started = time.monotonic()
        dispatched_here_seconds = 0.
        judged_here_seconds = 0.
        pending = store.pending(trial_id)
        if pending is None:
            calls = state.ledger.inference_attempts
            nominal = max(config.budgets.call_checkpoints)
            tokens = state.ledger.tokens
            token_covered = not config.budgets.token_checkpoints or (
                tokens is not None and tokens >= max(config.budgets.token_checkpoints))
            if calls >= nominal and token_covered:
                state.stop_reason = "checkpoints_covered"
            elif calls >= config.budgets.coverage_extension_call_ceiling:
                state.stop_reason = "extension_ceiling"
            if state.stop_reason:
                break
            campaign_check()
            estimator = CostEstimator(**state.estimator)
            frozen_predictor = FrozenLinearCostPredictor.from_config(config.cost_model)
            prompts, estimates, cases = {}, {}, {}
            cost_features = {}

            # For budgeted_rex, exclude policy-deduplicated arms when the flag is set.
            # Generation accounting is unaffected: duplicates are still charged.
            policy_arm_ids: set[str] | None = None
            if method in BUDGETED_METHODS and config.policy.deduplicate_policy_arms:
                # Keep only the canonical arm per code hash (earliest) and arms
                # with no code yet (root, or not yet generated).
                seen_code: dict[str, str] = {}
                policy_arm_ids = set()
                for a in state.arms:
                    if a.code_hash is None:
                        policy_arm_ids.add(a.arm_id)
                    elif a.code_hash not in seen_code:
                        seen_code[a.code_hash] = a.arm_id
                        policy_arm_ids.add(a.arm_id)

            for arm in state.arms:
                # For budgeted_rex with invalid_arm_actionable=False, skip arms
                # whose only generated code produced invalid completions and have
                # never produced a parseable code hash.
                if (method in BUDGETED_METHODS
                        and not config.policy.invalid_arm_actionable
                        and arm.code_hash is None
                        and arm.arm_id != "root"
                        and arm.invalid_count > 0):
                    continue
                # Deduplication filter.
                if policy_arm_ids is not None and arm.arm_id not in policy_arm_ids:
                    continue
                arm_code = store.get(arm.code_hash).decode() if arm.code_hash else ""
                prompt, failure_id = prepare(task, arm, arm_code, seed, state.iteration)
                if admissible(prompt, config.model.context_tokens, config.model.max_output_tokens):
                    prompts[arm.arm_id], cases[arm.arm_id] = prompt, failure_id
                    if method in BUDGETED_METHODS:
                        features = request_cost_features(
                            prompt=prompt,
                            candidate_code=arm_code,
                            depth=arm.depth,
                            kind="generation" if arm.arm_id == "root" else "repair",
                            prior_output_tokens=estimator.output_tokens,
                        )
                        cost_features[arm.arm_id] = features
                        estimates[arm.arm_id] = frozen_predictor.predict(features)
                    else:
                        # Preserve the archived estimator exactly for all baseline methods.
                        estimates[arm.arm_id] = estimator.estimate(prompt)

            # Budgeted REx: apply the per-trial token budget soft cap before selection.
            budget_active_estimates = estimates
            remaining_tokens: int | None = None
            if method in BUDGETED_METHODS and config.budgets.per_trial_token_cap is not None:
                spent = state.ledger.tokens
                if spent is not None:
                    remaining_tokens = config.budgets.per_trial_token_cap - spent
                    affordable = affordable_arms(estimates, remaining_tokens, config.budgets.token_safety_margin)
                    if not affordable:
                        state.stop_reason = "token_budget_exhausted"
                        break
                    budget_active_estimates = affordable

            try:
                parent_id, decision = select(
                    method, state.arms, budget_active_estimates, rng, config.policy, state.policy_state
                )
            except ValueError:
                state.stop_reason = "no_admissible_action"
                break

            prompt = prompts[parent_id]
            request_id = f"{trial_id}:{state.iteration + 1}"

            # Budgeted REx: optionally tighten the output limit to fit remaining budget.
            output_limit = config.model.max_output_tokens
            if method in BUDGETED_METHODS and remaining_tokens is not None:
                # The frozen predictor estimates total request cost.  Its frozen
                # output reserve is removed to obtain the input-side allowance.
                predicted_input = max(
                    1.0,
                    frozen_predictor.predict(cost_features[parent_id])
                    - frozen_predictor.output_reserve,
                )
                tighter = adaptive_output_limit(
                    remaining_tokens,
                    predicted_input,
                    config.budgets.token_safety_margin,
                    config.budgets.minimum_output_tokens,
                    config.model.max_output_tokens,
                )
                if tighter is None:
                    state.stop_reason = "token_budget_exhausted"
                    break
                output_limit = tighter

            request = GenerationRequest(experiment_id=experiment_id, trial_id=trial_id, attempt_id=request_id,
                                        messages=[Message(role="user", content=prompt)], prompt_hash=sha(prompt),
                                        backend=config.model.backend, model_digest=config.model.digest or backend.info()["digest"],
                                        temperature=config.model.temperature, output_limit=output_limit,
                                        context_tokens=config.model.context_tokens,
                                        generation_seed=seed_for("model", task.task_id, seed, state.iteration) % (2**31),
                                        kind="generation" if parent_id == "root" else "repair")

            # Mechanism diagnostics for budgeted_rex: record counterfactual baseline choice.
            mechanism: dict = {}
            if method in BUDGETED_METHODS:
                eligible_arms = [a for a in state.arms if a.arm_id in budget_active_estimates]
                base_choice = counterfactual_baseline_choice(
                    eligible_arms, budget_active_estimates, decision.get("draws", {})
                )
                component_choices = counterfactual_component_choices(
                    eligible_arms, budget_active_estimates, decision.get("draws", {}), config.policy
                )
                mechanism = {
                    "baseline_choice": base_choice,
                    "progress_only_choice": component_choices["progress_only"],
                    "cost_only_choice": component_choices["cost_only"],
                    "extension_choice": parent_id,
                    "policy_changed_choice": parent_id != base_choice,
                    "progress_changed_choice": component_choices["progress_only"] != base_choice,
                    "cost_changed_choice": component_choices["cost_only"] != base_choice,
                    "eligible_count": len(eligible_arms),
                    "budget_filtered_count": len(estimates) - len(budget_active_estimates),
                    "remaining_tokens": remaining_tokens,
                    "output_limit_used": output_limit,
                    "cost_estimator": "frozen_linear_v1",
                    "cost_model_hash": config.cost_model.coefficients_hash,
                    "extension_method": method,
                }

            context = {
                "parent_id": parent_id,
                "decision": decision,
                "feedback_id": cases[parent_id],
                "prompt": prompt,
                "extension": calls >= nominal,
                "mechanism": mechanism,
            }
            state.rng_state = dict(rng.bit_generator.state)
            state.ledger.reservations += 1
            with store.transaction():
                store.db.execute("INSERT INTO requests(id,trial_id,iteration,state,request_blob,context_blob) "
                                 "VALUES(?,?,?,'prepared',?,?)", (request_id, trial_id, state.iteration + 1,
                                 store.put(request.model_dump(mode="json")), store.put(context)))
                state.committed_event = store.event(trial_id, "request_prepared", {"request_id": request_id, **context})
                store.checkpoint(state, "running")
            fault("before_dispatch")
            pending = store.pending(trial_id)
        request = GenerationRequest.model_validate(store.object(pending["request_blob"]))
        context = store.object(pending["context_blob"])
        if pending["state"] == "rejected":
            pause("REQUEST_REJECTED")
            return state
        if pending["state"] in ("dispatched", "unknown") and not pending["response_blob"]:
            with store.transaction():
                store.db.execute("UPDATE requests SET state='unknown',error='REQUEST_OUTCOME_UNKNOWN' WHERE id=?",
                                 (request.attempt_id,))
                if not state.ledger.unknown_usage:
                    state.ledger.unknown_usage += 1
                store.checkpoint(state, "paused")
            pause("REQUEST_OUTCOME_UNKNOWN")
            return state
        if pending["state"] == "prepared":
            campaign_check()
            # Drift checks occur before the durable dispatch intent; a failed check consumed no inference.
            backend.info()
            started, timer = now(), time.monotonic()
            state.ledger.http_attempts += 1
            state.ledger.inference_attempts += 1
            with store.transaction():
                store.db.execute("UPDATE requests SET state='dispatched',started=? WHERE id=?",
                                 (started, request.attempt_id))
                store.checkpoint(state, "running")
                store.event(trial_id, "dispatch_intent", {"request_id": request.attempt_id})
            try:
                raw = backend.dispatch(request)
                dispatched_here_seconds = time.monotonic() - timer
                fault("after_dispatch")
            except RejectedBeforeInference as exc:
                with store.transaction():
                    store.db.execute("UPDATE requests SET state='rejected',error=? WHERE id=?",
                                     (str(exc), request.attempt_id))
                    state.ledger.inference_attempts -= 1
                    state.ledger.reservations -= 1
                    store.checkpoint(state, "paused")
                pause("REQUEST_REJECTED")
                return state
            except Exception as exc:
                with store.transaction():
                    store.db.execute("UPDATE requests SET state='unknown',error=? WHERE id=?",
                                     (type(exc).__name__, request.attempt_id))
                    state.ledger.unknown_usage += 1
                    store.checkpoint(state, "paused")
                pause("REQUEST_OUTCOME_UNKNOWN")
                return state
            # Persist the full raw payload before normalization, parsing, judging or usage updates.
            response_hash = store.put(raw)
            with store.transaction():
                store.db.execute("UPDATE requests SET state='responded',response_blob=?,finished=?,latency=? WHERE id=?",
                                 (response_hash, now(), dispatched_here_seconds, request.attempt_id))
                store.event(trial_id, "response_persisted", {"request_id": request.attempt_id, "raw": response_hash})
            fault("after_response")
            pending = store.pending(trial_id)
        raw = store.object(pending["response_blob"])
        result = backend.normalize(request, raw, pending["started"], pending["finished"], pending["latency"])
        # Include terminal generations in the drift check; the raw response is already durable.
        if backend.info()["digest"] != request.model_digest:
            pause("MODEL_DIGEST_DRIFT")
            return state
        if result.backend_error:
            pause("BACKEND_RESPONSE_ERROR")
            return state
        if result.input_tokens is None or result.output_tokens is None:
            with store.transaction():
                store.event(trial_id, "usage_unavailable", {"request_id": request.attempt_id,
                            "known_input_tokens": result.input_tokens, "known_output_tokens": result.output_tokens})
            pause("USAGE_UNKNOWN")
            return state
        code, parse_status = extract_code(result.text)
        code_hash = store.put(code) if code is not None else None
        if pending["evaluation_blob"]:
            evaluation = EvaluationResult.model_validate(store.object(pending["evaluation_blob"]))
        elif code is None:
            evaluation = EvaluationResult(outcomes=[TestOutcome(case_id=c.id, status=Status.WRONG_ANSWER,
                                                      stderr="INVALID_COMPLETION") for c in task.repair_tests],
                                           execution_fingerprint="invalid_completion_v1")
        else:
            judge_started = time.monotonic()
            evaluation = judge.evaluate(code, task.repair_tests, task.comparator, task.limits)
            judged_here_seconds = time.monotonic() - judge_started
            # Never convert infrastructure failure into a policy observation; response remains available.
            if evaluation.infrastructure_error:
                with store.transaction():
                    store.event(trial_id, "judge_infrastructure_failure", evaluation.model_dump(mode="json"))
                pause("JUDGE_INFRA_ERROR")
                return state
        if not pending["evaluation_blob"]:
            context["judge_elapsed_seconds"] = judged_here_seconds
            context["orchestration_seconds"] = max(0., time.monotonic() - step_started -
                                                     dispatched_here_seconds - judged_here_seconds)
            evaluation_hash = store.put(evaluation.model_dump(mode="json"))
            with store.transaction():
                store.db.execute("UPDATE requests SET state='judged',evaluation_blob=?,context_blob=? WHERE id=?",
                                 (evaluation_hash, store.put(context), request.attempt_id))
        fault("after_judge")
        parent = next(a for a in state.arms if a.arm_id == context["parent_id"])
        child = new_arm(f"arm-{state.iteration + 1}", evaluation.h, config.policy.C, code_hash=code_hash,
                        generation_id=request.attempt_id, parent_id=parent.arm_id, depth=parent.depth + 1,
                        order=state.iteration + 1, evaluation=evaluation)
        terminal = code is not None and evaluation.h == 1

        # Legacy posterior update (all archived methods).
        eta, _ = config.policy.toggles(method)
        reward = observe(parent, child, eta, terminal)

        # budgeted_rex: apply separate progress posterior update and track branch stats.
        progress_delta: float = 0.0
        if method in BUDGETED_METHODS and not terminal:
            progress_delta = observe_improvement(parent, child)
            parent.generated_token_sum += (result.input_tokens or 0) + (result.output_tokens or 0)
            if code is None:
                parent.invalid_count += 1

        advance(method, parent, child, state.policy_state, config.policy)
        state.arms.append(child)

        # Track duplicate generations for budgeted_rex branch-quality stats.
        duplicate = code_hash is not None and any(a.code_hash == code_hash for a in state.arms[:-1])
        if method in BUDGETED_METHODS and duplicate:
            parent.duplicate_count += 1

        # Update code-to-arm index used by policy deduplication.
        if method in BUDGETED_METHODS and code_hash is not None and code_hash not in state.code_to_arm_id:
            state.code_to_arm_id[code_hash] = child.arm_id

        best = next((a for a in state.arms if a.arm_id == state.best_id), None)
        if code is not None and (best is None or child.h > best.h):
            state.best_id = child.arm_id
            best = child
        charge(state.ledger, result)
        state.ledger.usable_completions += int(code is not None)
        state.ledger.policy_steps += 1
        state.ledger.judge_seconds += context.get("judge_elapsed_seconds", sum(o.wall_seconds + o.startup_seconds for o in evaluation.outcomes))
        state.ledger.orchestration_seconds += context.get("orchestration_seconds", 0.)
        estimator = CostEstimator(**state.estimator)
        estimator.observe(context["prompt"], result)
        state.estimator = asdict(estimator)
        state.iteration += 1
        state.stop_reason = "visible_success" if terminal else None
        snapshot = {"calls": state.ledger.inference_attempts, "tokens": state.ledger.tokens,
                    "best_id": state.best_id, "code_hash": best.code_hash if best else None,
                    "h": best.h if best else 0., "step": state.iteration, "visible_success": terminal}
        event = {"request_id": request.attempt_id, "parent_id": parent.arm_id, "child": child.model_dump(mode="json"),
                 "reward": reward, "decision": context["decision"], "feedback_id": context["feedback_id"],
                 "prompt": context["prompt"], "duplicate": duplicate, "parse_status": parse_status,
                 "estimated_tokens": context["decision"]["estimates"][parent.arm_id],
                 "actual_tokens": result.input_tokens + result.output_tokens,
                 "parent_after": [parent.alpha, parent.beta], "snapshot": snapshot, "extension": context["extension"],
                 "ledger": state.ledger.model_dump(mode="json"),
                 "mechanism": context.get("mechanism", {}),
                 "progress_delta": progress_delta}
        fault("during_update")
        with store.transaction():
            store.db.execute("INSERT INTO generations VALUES(?,?,?,?)", (request.attempt_id, code_hash,
                             parse_status, store.put(result.model_dump(mode="json"))))
            for arm in state.arms:
                store.db.execute("INSERT OR REPLACE INTO arms VALUES(?,?,?)",
                                 (trial_id, arm.arm_id, canonical(arm.model_dump(mode="json")).decode()))
            store.db.execute("INSERT INTO edges VALUES(?,?,?)", (request.attempt_id, parent.arm_id, child.arm_id))
            for outcome in evaluation.outcomes:
                store.db.execute("INSERT INTO test_outcomes VALUES(?,?,?)", (request.attempt_id, outcome.case_id,
                                 canonical(outcome.model_dump(mode="json")).decode()))
            store.db.execute("INSERT INTO snapshots VALUES(?,?,?)", (trial_id, state.iteration, canonical(snapshot).decode()))
            store.db.execute("UPDATE requests SET state='committed' WHERE id=?", (request.attempt_id,))
            state.committed_event = store.event(trial_id, "step_committed", event)
            store.checkpoint(state, "completed" if terminal else "running")
        fault("after_commit")
    with store.transaction():
        store.checkpoint(state, "completed" if state.stop_reason in ("visible_success", "checkpoints_covered") else "incomplete")
    return state
