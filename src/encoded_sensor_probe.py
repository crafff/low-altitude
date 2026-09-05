"""Six native encoded-fault plumbing cases with one unchanged untrained policy.

Run only through the controller's isolated launcher. All files are written under
LAB_RUN_DIR. The forward hook observes the original select_actions call; it
neither substitutes logits nor performs another forward or random draw.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
from dataclasses import asdict
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from encoded_sensor_faults import FaultPlan, Slot, load_config as load_sensor_config, transform
from paper_environment import PaperEnvironment, load_environment_config
from paper_rollout import constructed_scenario
from paper_train import select_actions
from perturbation_probe import (PhysicalTrace, canonical, model_digest, observations_packet,
                                physical_packet, scientific_summary, update_digest, write_json)
from shared_ppo import SharedActorCritic


CASES = ("plain", "emptyplan", "disabled", "missing", "abnormal", "falsified")
PLAN_CONTRACT = {"missing": {"own_column": 2, "sentinel": 2.0},
                 "abnormal": {"own_column": 2, "value": -.5},
                 "falsified": {"own_columns": [2, 3], "permutation": [1, 0]}}


def plans_for(case):
    if case in ("disabled", "missing"):
        return (FaultPlan("missing", (Slot("own", 2),)),)
    if case == "abnormal":
        return (FaultPlan("abnormal", (Slot("own", 2),), value=-.5),)
    if case == "falsified":
        return (FaultPlan("falsified", (Slot("own", 2), Slot("own", 3)), permutation=(1, 0)),)
    if case in ("plain", "emptyplan"):
        return ()
    raise ValueError("Unknown predeclared case")


def _check(checks, key, passed):
    checks[key] = checks.get(key, True) and bool(passed)
    if not passed:
        raise RuntimeError("Encoded sensor probe invariant failed: " + key)


def _deadline(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError("Global inner deadline reached between native decisions/cases")


def _rng_state(sampling):
    state = np.random.get_state()
    return (random.getstate(), state[0], state[1].tobytes(), state[2:],
            torch.get_rng_state().numpy().tobytes(), sampling.get_state().numpy().tobytes())


def _truth_packet(env):
    return canonical({"physical": physical_packet(env),
        "states": {acid: asdict(state) for acid, state in env.physical_states().items()},
        "accepted": {acid: env.actions.state_fields(acid) for acid in sorted(env.bs.traf.id)}})


def _tensor_packet(tensor):
    array = tensor.detach().cpu().contiguous().numpy()
    return {"dtype": array.dtype.str, "shape": list(array.shape), "bytes": array.tobytes().hex()}


class ForwardEvidence:
    """Passive checker of tensors already passed to the actual model call."""
    def __init__(self, checks, writer, sampling):
        self.checks, self.writer, self.sampling = checks, writer, sampling
        self.pending, self.calls, self.aircraft = None, 0, 0
        self.digest = hashlib.sha256()

    def expect(self, observations, time_s):
        _check(self.checks, "previous_forward_expectation_consumed", self.pending is None)
        self.pending = (time_s, observations_packet(observations), self.sampling.get_state().numpy().tobytes())

    def __call__(self, module, args, output):
        rng_before = _rng_state(self.sampling)
        _check(self.checks, "forward_has_one_declared_input", self.pending is not None)
        now, packets, sample_before = self.pending
        ids = sorted(packets)
        own, intruders, present, action_mask = args
        logits, values = output
        width = max(1, max(packet["intruders"]["shape"][0] for packet in packets.values()))
        _check(self.checks, "forward_shapes_and_device_match", tuple(own.shape) == (len(ids), 7)
            and tuple(intruders.shape) == (len(ids), width, 10)
            and tuple(present.shape) == (len(ids), width) and tuple(action_mask.shape) == (len(ids), 60)
            and tuple(logits.shape) == (len(ids), 60) and tuple(values.shape) == (len(ids),)
            and all(tensor.device.type == "cpu" for tensor in (*args, *output)))
        _check(self.checks, "forward_features_finite_and_original_dtype", own.dtype == intruders.dtype == torch.float32
            and present.dtype == action_mask.dtype == torch.bool
            and torch.isfinite(own).all() and torch.isfinite(intruders).all())
        for index, acid in enumerate(ids):
            packet = packets[acid]
            n = packet["intruders"]["shape"][0]
            expected_present = np.arange(width) < n
            _check(self.checks, "actual_policy_input_bytes_match_encoded", _tensor_packet(own[index]) == packet["own"]
                and _tensor_packet(intruders[index, :n]) == packet["intruders"]
                and _tensor_packet(action_mask[index]) == packet["action_mask"]
                and np.array_equal(present[index].detach().numpy(), expected_present)
                and bool((intruders[index, n:] == 0).all()))
        _check(self.checks, "legal_logits_and_values_finite", torch.isfinite(logits[action_mask]).all()
            and torch.isfinite(values).all() and action_mask.any(dim=1).all())
        _check(self.checks, "illegal_logits_keep_original_mask", torch.isneginf(logits[~action_mask]).all())
        _check(self.checks, "policy_forward_has_no_gradients", not module.training and not torch.is_grad_enabled()
            and all(not p.requires_grad and p.grad is None for p in module.parameters()))
        _check(self.checks, "forward_did_not_draw_sampling_rng", sample_before == self.sampling.get_state().numpy().tobytes())
        evidence = {"sim_time_s": now, "sorted_inferred_ids": ids,
                    "actual_inputs": [_tensor_packet(tensor) for tensor in args],
                    "logit_bits": _tensor_packet(logits), "values": values.detach().tolist(),
                    "legal_logit_min": float(logits[action_mask].min()),
                    "legal_logit_max": float(logits[action_mask].max())}
        update_digest(self.digest, evidence)
        self.writer.write(canonical(evidence) + "\n")
        self.calls += 1
        self.aircraft += len(ids)
        self.pending = None
        _check(self.checks, "passive_hook_consumes_no_rng", _rng_state(self.sampling) == rng_before)
        # Returning None preserves the original model output objects.


def run_case(env, model, cfg, case, scenario, output, deadline, initial_model):
    plans = plans_for(case)
    write_json(output / "effective_config.json", {"case": case, "scope": cfg["scope"],
        "enabled": case != "disabled", "plans": [asdict(plan) for plan in plans],
        "environment": env.cfg, "scenario_config": env.scenario_cfg, "action_config": env.action_cfg,
        "observation_config": env.observation_cfg, "policy_config": model.cfg,
        "model_seed": cfg["model_seed"], "sampling_seed": cfg["sampling_seed"]})
    write_json(output / "scenario.json", scenario)
    checks, inference_by_id = {}, Counter()
    counts = Counter(dict.fromkeys(("plans", "planned_entries", "applied_entries", "numerically_changed_entries",
                                    "bitwise_changed_entries", "decision_boundaries"), 0))
    clean_digest, encoded_digest, action_digest = (hashlib.sha256() for _ in range(3))
    result = {"case": case, "status": "error", "scope": cfg["scope"]}
    trace = evidence = None
    reset_completed = False
    sampling = torch.Generator(device="cpu").manual_seed(cfg["sampling_seed"])
    hook_members = list(model._forward_hooks.items())
    previous_callback, handle = env.on_physics_step, None
    try:
        _deadline(deadline)
        observations = env.reset(scenario["seed"], scenario=copy.deepcopy(scenario))
        reset_completed = True
        with ExitStack() as stack:
            trace = PhysicalTrace(env, None, output, stack)
            env.on_physics_step = trace.record
            fault_file = stack.enter_context((output / "faults.jsonl").open("x"))
            observation_file = stack.enter_context((output / "observations.jsonl").open("x"))
            decision_file = stack.enter_context((output / "decisions.jsonl").open("x"))
            evidence = ForwardEvidence(checks, stack.enter_context((output / "forward.jsonl").open("x")), sampling)
            handle = model.register_forward_hook(evidence)
            while not env.done:
                _deadline(deadline)
                now = float(env.bs.sim.simt)
                original_truth, clean = _truth_packet(env), observations_packet(observations)
                rng_before = _rng_state(sampling)
                encoded = {}
                for acid in sorted(observations):
                    if case == "plain":
                        encoded[acid] = observations[acid]
                    else:
                        changed = transform(observations[acid], plans, enabled=case != "disabled")
                        encoded[acid] = changed.observation
                        counts.update(asdict(changed.counts))
                        fault_file.write(canonical({"sim_time_s": now, "aircraft": acid,
                                                  "trace": changed.trace_dict()}) + "\n")
                        _check(checks, "transformed_arrays_are_independent_copies", all(
                            not np.shares_memory(encoded[acid][key], observations[acid][key])
                            for key in ("own", "intruders", "action_mask")))
                _check(checks, "transform_preserves_truth_native_and_accepted_fields", original_truth == _truth_packet(env))
                _check(checks, "transform_preserves_clean_arrays_masks_and_ids", canonical(clean) == canonical(observations_packet(observations)))
                _check(checks, "transform_consumes_no_global_or_sampling_rng", rng_before == _rng_state(sampling))
                _check(checks, "all_physical_ids_still_inferred", sorted(encoded) == sorted(env.bs.traf.id) == sorted(observations))
                packet = observations_packet(encoded)
                for acid in encoded:
                    expected_own = observations[acid]["own"].copy()
                    if case == "missing":
                        expected_own[2] = 2.0
                    elif case == "abnormal":
                        expected_own[2] = -.5
                    elif case == "falsified":
                        expected_own[[2, 3]] = observations[acid]["own"][[3, 2]]
                    _check(checks, "explicit_own_slot_values_and_only_selected_changes", encoded[acid]["own"].tobytes() == expected_own.tobytes()
                        and packet[acid]["intruders"] == clean[acid]["intruders"]
                        and packet[acid]["clipping_counts"] == clean[acid]["clipping_counts"])
                    _check(checks, "transformed_masks_and_row_ids_are_original", packet[acid]["action_mask"] == clean[acid]["action_mask"]
                        and packet[acid]["intruder_ids"] == clean[acid]["intruder_ids"]
                        and np.array_equal(encoded[acid]["action_mask"], env.actions.action_mask(acid)))
                if case in ("plain", "emptyplan", "disabled"):
                    _check(checks, "zero_case_input_bitwise_identity", canonical(packet) == canonical(clean))
                update_digest(clean_digest, [now, clean])
                update_digest(encoded_digest, [now, packet])
                observation_file.write(canonical({"sim_time_s": now, "clean_bits": clean, "policy_input_bits": packet}) + "\n")
                if encoded:
                    evidence.expect(encoded, now)
                calls_before = evidence.calls
                actions, log_probs, values = select_actions(model, encoded, sampling)
                _check(checks, "policy_call_preserves_encoded_and_clean_arrays", canonical(packet) == canonical(observations_packet(encoded))
                    and canonical(clean) == canonical(observations_packet(observations)))
                _check(checks, "exactly_one_original_forward_per_nonempty_decision", evidence.calls - calls_before == int(bool(encoded))
                    and evidence.pending is None)
                _check(checks, "actual_inference_and_action_ids_match", set(actions) == set(log_probs) == set(values) == set(encoded))
                _check(checks, "sampled_actions_legal_and_outputs_finite", all(encoded[acid]["action_mask"][action]
                    and math.isfinite(log_probs[acid]) and math.isfinite(values[acid]) for acid, action in actions.items()))
                inference_by_id.update(actions.keys())
                update_digest(action_digest, [now, actions])
                decision_file.write(canonical({"sim_time_s": now, "inferred_ids": sorted(actions), "actions": actions,
                                               "log_probs": log_probs, "values": values}) + "\n")
                observations, _, _, _ = env.step(actions)
                counts["decision_boundaries"] += 1
            summary = env.summary(include_flights=True)
            result.update(status="completed", environment_summary=summary)
            _check(checks, "complete_population_accounted", summary["completed_population"] and summary["planned"] == 2)
            _check(checks, "physics_trace_covers_native_lifetimes", trace.steps == summary["physics_steps"]
                and math.isclose(trace.alive_seconds, env.flight_seconds, rel_tol=0, abs_tol=1e-9)
                and all(trace.presence.get(r["id"], 0) == round(r["flight_seconds"] / env.dt) for r in summary["flights"]))
            _check(checks, "no_holding_or_missing_inference", sum(inference_by_id.values()) == summary["policy_decisions"]
                and all(inference_by_id[r["id"]] == r["policy_decisions"] for r in summary["flights"]))
            if case in ("missing", "abnormal", "falsified"):
                _check(checks, "explicit_corruption_reached_actual_inference", counts["numerically_changed_entries"] > 0
                    and counts["applied_entries"] > 0 and evidence.calls > 0)
            else:
                _check(checks, "zero_or_disabled_has_no_applied_changes", counts["applied_entries"] == 0
                    and counts["numerically_changed_entries"] == counts["bitwise_changed_entries"] == 0)
    except Exception as exc:
        result.update(status="cutoff" if isinstance(exc, TimeoutError) else "error", error=f"{type(exc).__name__}: {exc}",
                      traceback=traceback.format_exc())
    finally:
        if handle is not None:
            handle.remove()
        env.on_physics_step = previous_callback
        checks["forward_hook_restored"] = list(model._forward_hooks.items()) == hook_members
        checks["physical_callback_restored"] = env.on_physics_step is previous_callback
        checks["model_unchanged_and_no_gradients"] = model_digest(model) == initial_model and all(
            p.grad is None and not p.requires_grad for p in model.parameters())
        result.update(checks=checks, all_checks_passed=result["status"] == "completed" and all(checks.values()),
            fault_counts=dict(counts), inference_by_aircraft=dict(inference_by_id), held_aircraft_decisions=0,
            clean_observation_stream_sha256=clean_digest.hexdigest(), observation_stream_sha256=encoded_digest.hexdigest(),
            action_stream_sha256=action_digest.hexdigest(), model_sha256=model_digest(model),
            sampling_rng_sha256=hashlib.sha256(sampling.get_state().numpy().tobytes()).hexdigest())
        if trace is not None:
            result.update(physical_trajectory_sha256=trace.digest.hexdigest(), physical_steps=trace.steps, physical_rows=trace.rows)
        if evidence is not None:
            result.update(forward_stream_sha256=evidence.digest.hexdigest(), actual_forward_calls=evidence.calls,
                          actual_inference_aircraft_decisions=evidence.aircraft)
        result["case_reset_completed"] = reset_completed
        if reset_completed:
            write_json(output / "truth_risk_events.json", {"closed_events": env.tracker.events,
                "active_events": {level: list(active.values()) for level, active in env.tracker.active.items()},
                "unordered_pair_seconds": env.tracker.exposure,
                "scope": "Original environment truth tracker; active events are retained if this case was interrupted."})
            write_json(output / "flights.json", list(env.records.values()))
        write_json(output / "result.json", result)
    return result


def compare_cases(results):
    rows = {result["case"]: result for result in results}
    if set(rows) != set(CASES) or any(row["status"] != "completed" for row in rows.values()):
        return {"all_six_cases_completed_without_exception": False}
    checks, plain = {}, rows["plain"]
    for name in ("emptyplan", "disabled"):
        for key in ("clean_observation_stream_sha256", "observation_stream_sha256", "action_stream_sha256",
                    "physical_trajectory_sha256", "forward_stream_sha256", "sampling_rng_sha256"):
            checks[name + "_plain_" + key + "_equal"] = rows[name][key] == plain[key]
        checks[name + "_plain_scientific_summary_exact"] = canonical(scientific_summary(rows[name]["environment_summary"])) == canonical(scientific_summary(plain["environment_summary"]))
    # Nonzero faults have no trajectory-equality or risk-improvement criterion.
    return checks


def main(argv=None):
    started = time.perf_counter()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/encoded_sensor_probe.json")
    args = parser.parse_args(argv)
    cfg = json.loads(Path(args.config).read_text())
    if (cfg["schema"] != "bluesky.encoded-sensor-probe.v1" or tuple(cfg["cases"]) != CASES
            or cfg["explicit_plan_contract"] != PLAN_CONTRACT
            or (cfg["scenario_seed"], cfg["model_seed"], cfg["sampling_seed"]) != (920001, 930001, 940001)
            or cfg["inner_wall_seconds"] != 60.0 or cfg["environment_config"] != "configs/paper_environment_execution.json"):
        raise ValueError("Preserve the fixed six cases, seeds, plans, execution environment and inner budget")
    if version("bluesky-simulator") != cfg["bluesky_version"] or cfg["bluesky_version"] != "1.1.1":
        raise ValueError("This native probe requires BlueSky 1.1.1")
    env_cfg, parts = load_environment_config(cfg["environment_config"])
    if env_cfg["decision_seconds"] != 5 or parts["scenario"]["dt_seconds"] != .25:
        raise ValueError("Retain the original five-second decisions and quarter-second physics")
    policy_cfg = json.loads(Path(cfg["policy_config"]).read_text())
    sensor_cfg = load_sensor_config(cfg["sensor_config"])
    output = Path(os.environ["LAB_RUN_DIR"]).resolve(strict=True)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(cfg["model_seed"])
    model = SharedActorCritic(policy_cfg).eval().requires_grad_(False)
    initial_model = model_digest(model)
    scenario = constructed_scenario("crossing", "M100", cfg["scenario_seed"])
    env = PaperEnvironment(env_cfg, parts)
    source = Path(__file__).resolve().parent
    names = ("encoded_sensor_probe.py", "encoded_sensor_faults.py", "perturbation_probe.py", "paper_environment.py",
             "paper_actions.py", "paper_observation.py", "paper_rollout.py", "paper_train.py", "shared_ppo.py",
             "paper_performance.py", "bluesky_diagnostic.py", "nr_pilot.py", "route_completion.py")
    config_paths = [Path(args.config), Path(cfg["environment_config"]), Path(cfg["policy_config"]), Path(cfg["sensor_config"])]
    config_paths += [Path(env_cfg[key]) for key in ("scenario_config", "action_config", "observation_config", "types_config")]
    report = {"schema": cfg["schema"], "scope": cfg["scope"], "config": cfg, "sensor_contract": sensor_cfg,
        "model_initializations": 1, "model_initial_sha256": initial_model, "cases": [], "all_checks_passed": False,
        "versions": {name: version(name) for name in ("bluesky-simulator", "numpy", "torch")},
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names},
        "config_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in config_paths}}
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    with torch.no_grad():
        for case in CASES:
            case_output = output / case
            case_output.mkdir()
            if time.perf_counter() >= started + cfg["inner_wall_seconds"]:
                result = {"case": case, "status": "not_started", "error": "Inner budget exhausted", "all_checks_passed": False}
                write_json(case_output / "result.json", result)
            else:
                try:
                    result = run_case(env, model, cfg, case, scenario, case_output,
                                      started + cfg["inner_wall_seconds"], initial_model)
                except Exception as exc:
                    result = {"case": case, "status": "error", "error": f"{type(exc).__name__}: {exc}",
                              "traceback": traceback.format_exc(), "all_checks_passed": False}
                    write_json(case_output / "failure.json", result)
            report["cases"].append(result)
            (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
            print(canonical({"case": case, "status": result["status"], "all_checks_passed": result["all_checks_passed"]}), flush=True)
    report["comparison_checks"] = compare_cases(report["cases"])
    report["all_checks_passed"] = all(row["all_checks_passed"] for row in report["cases"]) and all(report["comparison_checks"].values())
    report["wall_seconds"] = time.perf_counter() - started
    (output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
