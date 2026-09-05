"""Bounded native validation of explicit ordinary-flyby refresh; no training."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
import traceback

import numpy as np
import torch

from navigation_sensitivity import refresh_reached
from paper_environment import PaperEnvironment, load_environment_config
from paper_scenarios import generate_scenario
from paper_train import _evaluate_case, _versions, aggregate_summaries
from route_observation_probe import CSVStream, model_identity, run_case
from shared_ppo import SharedActorCritic


SCOPE = "Native guidance reconstruction validation; development only; containment unresolved"
ROUTE_IDS = ("refresh-amzn-north-s0-l1", "refresh-amzn-north-s0-l2",
             "refresh-amzn-north-s2-l1", "refresh-amzn-north-s3-l1")
RUNTIME_KEYS = frozenset(("wall_seconds", "process_peak_rss_mib"))


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def scientific(value, *, route=False):
    """Exclude only named runtime fields and the route's intentional arm label."""
    if isinstance(value, dict):
        return {key: scientific(item, route=route) for key, item in value.items()
                if key not in RUNTIME_KEYS and not (route and key == "arm")}
    if isinstance(value, (list, tuple)):
        return [scientific(item, route=route) for item in value]
    return value


def differences(actual, expected, path="$", limit=100):
    """Exact structural/value comparison; retain count and bounded first details."""
    found, count = [], 0
    def add(where, a, b):
        nonlocal count
        count += 1
        if len(found) < limit:
            found.append({"path": where, "actual": a, "expected": b})
    def visit(a, b, where):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b)):
                if key not in a or key not in b:
                    add(where+"."+key, a.get(key, "<missing>"), b.get(key, "<missing>"))
                else:
                    visit(a[key], b[key], where+"."+key)
        elif isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                add(where+".length", len(a), len(b))
            for i, (left, right) in enumerate(zip(a, b)):
                visit(left, right, f"{where}[{i}]")
        elif canonical(a) != canonical(b):
            add(where, a, b)
    visit(actual, expected, path)
    return {"exact": count == 0, "mismatch_count": count, "first_mismatches": found,
            "mismatch_detail_limit": limit}


def file_identity(path):
    value = Path(path)
    return {"path": str(path), "bytes": value.stat().st_size,
            "sha256": hashlib.sha256(value.read_bytes()).hexdigest()}


def save(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")
    os.replace(temporary, path)


def read_zero(path, cases):
    matches = [(i, json.loads(line)) for i, line in enumerate(Path(path).read_text().splitlines(), 1)
               if line.strip() and json.loads(line).get("completed_episodes") == 0]
    if len(matches) != 1:
        raise ValueError("Reference must contain exactly one completed-episode-0 evaluation")
    line, row = matches[0]
    actual_cases = [{key: item[key] for key in ("seed", "corridor_count")} for item in row["cases"]]
    if actual_cases != cases or row["case_count"] != 12 or row["primary_policy"] != "sample":
        raise ValueError("Reference must retain all twelve original development cases and sampled primary policy")
    return row, line


def integrated_audit_ok(audit):
    return (audit["enabled"] and audit["wrapper_calls"] > 0
            and audit["wrapper_calls"] == audit["native_calls"] == audit["rng_checks"]
            and audit["context_entries"] == audit["context_exits"] > 0
            and not audit["context_active"] and audit["unsupported_calls"] == 0
            and all(audit[key] for key in ("adapter_rng_unchanged", "facade_identity_preserved",
                                          "wrapper_identity_preserved", "method_restored")))


class PopulationHarness:
    """Reuse _evaluate_case; install observers/external refresh only after reset.

    A passive inner observer independently counts calls to the native method.
    Both variants hash every post-physics state before terminal deletion. No
    observation construction, action sampling or reward computation is added.
    """
    def __init__(self, env, external):
        self.env, self.external = env, external
        self.stack = ExitStack()
        self.calls = self.physics_rows = self.resets = self.population_changes = 0
        self.sizes, self.previous_ids = set(), None
        self.native_digest, self.physics_digest = hashlib.sha256(), hashlib.sha256()
        self.external_audit = {}
        self.native_restored = self.callback_restored = False

    def __getattr__(self, name):
        return getattr(self.env, name)

    def reset(self, seed, *, scenario):
        if self.resets:
            raise RuntimeError("One harness owns exactly one completed population reset")
        if self.env.navigation_audit["context_active"]:
            raise RuntimeError("Reset encountered an installed integrated refresh")
        observations = self.env.reset(seed, scenario=scenario)
        self.resets += 1
        self.reset_audit = copy.deepcopy(self.env.navigation_audit)
        facade = self.env.bs.traf.actwp
        namespace = vars(facade)
        had_member, old_member = "reached" in namespace, namespace.get("reached")
        original = facade.reached
        def native(qdr, dist):
            ids = tuple(self.env.bs.traf.id)
            self.calls += 1
            self.sizes.add(len(ids))
            self.population_changes += int(self.previous_ids is not None and ids != self.previous_ids)
            self.previous_ids = ids
            self.native_digest.update(canonical([ids, np.asarray(qdr).tolist(), np.asarray(dist).tolist(),
                                                 np.asarray(facade.turndist).tolist()]).encode())
            return original(qdr, dist)
        def restore_native():
            if had_member:
                namespace["reached"] = old_member
            else:
                namespace.pop("reached", None)
            self.native_restored = facade.reached == original
        namespace["reached"] = native
        self.stack.callback(restore_native)
        old_callback = self.env.on_physics_step
        def record(env):
            traf = env.bs.traf
            self.physics_rows += 1
            self.physics_digest.update(canonical([float(env.bs.sim.simt), list(traf.id),
                *[np.asarray(getattr(traf, name)).tolist() for name in
                  ("lat", "lon", "alt", "tas", "gs", "hdg", "trk")],
                np.asarray(traf.actwp.turndist).tolist()]).encode())
            if old_callback is not None:
                old_callback(env)
        def restore_callback():
            self.env.on_physics_step = old_callback
            self.callback_restored = self.env.on_physics_step is old_callback
        self.env.on_physics_step = record
        self.stack.callback(restore_callback)
        if self.external:
            if self.env.ordinary_flyby_guidance != "native_cached":
                raise ValueError("External reference must never double-wrap integrated refresh")
            from bluesky.tools.aero import g0
            self.stack.enter_context(refresh_reached(self.env.bs, float(g0), self.external_audit))
        return observations

    def close(self):
        self.stack.close()

    def audit(self):
        integrated = copy.deepcopy(self.env.navigation_audit)
        active = self.external_audit if self.external else integrated
        passed = (self.calls == active.get("wrapper_calls") == active.get("native_calls") and self.calls > 0
                  and self.population_changes > 0 and len(self.sizes) > 1 and self.resets == 1
                  and self.reset_audit["context_entries"] == 0 and not self.reset_audit["context_active"]
                  and self.native_restored and self.callback_restored
                  and active.get("method_restored") and active.get("unsupported_calls") == 0
                  and (active.get("wrapper_member_survived") if self.external else integrated_audit_ok(integrated)))
        return {"all_checks_passed": bool(passed), "external": self.external,
                "actual_native_calls": self.calls, "physics_rows": self.physics_rows,
                "population_sizes_seen": sorted(self.sizes), "population_id_changes": self.population_changes,
                "resets": self.resets, "reset_audit": self.reset_audit,
                "native_method_restored": self.native_restored, "callback_restored": self.callback_restored,
                "native_call_stream_sha256": self.native_digest.hexdigest(),
                "physical_state_stream_sha256": self.physics_digest.hexdigest(),
                "integrated_audit": integrated, "external_audit": copy.deepcopy(self.external_audit)}


def physics_reference_comparison(actual_path, expected_path):
    def rows(path):
        with Path(path).open(newline="") as handle:
            result = list(csv.DictReader(handle))
        if {row["case_id"] for row in result} != set(ROUTE_IDS):
            raise ValueError("Physics CSV must contain exactly the four declared refresh cases")
        return [{key: value for key, value in row.items() if key != "arm"} for row in result]
    actual, expected = rows(actual_path), rows(expected_path)
    # Case labels are deliberately retained as the original refresh IDs; only
    # arm differs because run_case's native arm prevents external double-wrap.
    return differences(actual, expected)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-zero", required=True)
    parser.add_argument("--route-reference", required=True)
    parser.add_argument("--physics-reference", required=True)
    args = parser.parse_args(argv)
    if os.environ.get("LAB_RUN_DIR") != "/output" or Path.cwd() != Path("/workspace"):
        raise ValueError("Native validation requires the isolated lab launcher")
    output, started = Path("/output"), time.perf_counter()
    result = {"schema": "bluesky.navigation-refresh-probe-result.v1", "scope": SCOPE,
              "all_checks_passed": False, "default_off": {"cases": []}, "fixed_routes": [], "populations": []}
    try:
        cfg = json.loads(Path(args.config).read_text())
        if (cfg["schema"] != "bluesky.navigation-refresh-probe.v1" or cfg["scope"] != SCOPE
                or cfg["device"] != "cpu" or cfg["torch_num_threads"] != 1
                or not 0 < cfg["wall_seconds"] <= 400
                or not 0 < cfg["maximum_csv_bytes"] <= 33554432 or tuple(cfg["route_case_ids"]) != ROUTE_IDS):
            raise ValueError("Unexpected bounded refresh validation configuration")
        deadline = started+cfg["wall_seconds"]
        training = json.loads(Path(cfg["training_config"]).read_text())
        cases = [{"seed": 53001+i, "corridor_count": 3+i % 3} for i in range(12)]
        if training["development_cases"] != cases or training["training_seed"] != 61001 or training["dev_action_seed_base"] != 810000:
            raise ValueError("Original initialization/development seeds differ")
        reference, line = read_zero(args.reference_zero, cases)
        route_reference = json.loads(Path(args.route_reference).read_text())
        if file_identity(args.route_reference)["sha256"] != cfg["route_reference_sha256"]:
            raise ValueError("Route reference differs from the audited fifty-case result")
        if file_identity(args.physics_reference)["sha256"] != cfg["physics_reference_sha256"]:
            raise ValueError("Physics reference differs from the four selected audited traces")
        reference_routes = {row["case"]["id"]: row for row in route_reference["cases"]}
        off_cfg, parts = load_environment_config(training["environment_config"])
        on_cfg, on_parts = load_environment_config(cfg["refresh_environment_config"])
        if (off_cfg.get("ordinary_flyby_guidance", "native_cached") != "native_cached"
                or on_cfg["ordinary_flyby_guidance"] != "current_state_refresh" or parts != on_parts
                or {k:v for k,v in off_cfg.items() if k != "reconstruction_choices"} !=
                   {k:v for k,v in on_cfg.items() if k not in ("ordinary_flyby_guidance", "reconstruction_choices")}):
            raise ValueError("On/off environment difference extends beyond the declared guidance option")
        ppo_cfg = json.loads(Path(training["ppo_config"]).read_text())
        route_cfg = json.loads(Path(cfg["route_study_config"]).read_text())
        if route_cfg["model_seed"] != 960001 or route_cfg["seed"] != 966001:
            raise ValueError("Audited route initialization changed")
        paths = [args.config, args.reference_zero, args.route_reference, args.physics_reference,
                 cfg["training_config"], training["environment_config"], cfg["refresh_environment_config"],
                 training["ppo_config"], cfg["route_study_config"], off_cfg["types_config"],
                 *[off_cfg[f"{key}_config"] for key in ("scenario", "action", "observation")],
                 "src/navigation_refresh_probe.py", "src/navigation_sensitivity.py", "src/route_observation_probe.py",
                 "src/route_action_study.py"]
        result.update(config=cfg, reference_zero_line=line, provenance=[file_identity(path) for path in dict.fromkeys(paths)],
                      versions=_versions(), effective_training=training, effective_off_environment=off_cfg,
                      effective_on_environment=on_cfg, effective_parts=parts,
                      comparison_exclusions={"summary": sorted(RUNTIME_KEYS), "fixed_route_and_csv": ["arm"]})
        if result["versions"]["bluesky"] != "1.1.1":
            raise ValueError("Native version differs from the audited BlueSky 1.1.1")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        random.seed(61001)
        np.random.seed(61001)
        torch.manual_seed(61001)
        model = SharedActorCritic(ppo_cfg).eval().requires_grad_(False)
        before_model = model_identity(model)
        result["model_before_sha256"] = before_model
        startup = (random.getstate(), np.random.get_state(), torch.random.get_rng_state())
        env_off = PaperEnvironment(off_cfg, parts)
        random.setstate(startup[0])
        np.random.set_state(startup[1])
        torch.random.set_rng_state(startup[2])
        scenarios = [generate_scenario(dict(parts["scenario"], corridor_counts=[case["corridor_count"]]),
                                       case["seed"], list(env_off.types)) for case in cases]
        for case, scenario, expected in zip(cases, scenarios, reference["cases"]):
            row = dict(case, action_seed=810000+case["seed"])
            result["default_off"]["cases"].append(row)
            for policy in ("nr", "sample"):
                summary = _evaluate_case(env_off, model, scenario, policy, row["action_seed"], deadline)
                row[policy] = summary
                row[policy+"_comparison"] = differences(scientific(summary), scientific(expected[policy]))
                row[policy+"_audit"] = copy.deepcopy(env_off.navigation_audit)
                save(output/"result.json", result)
            print(canonical({"phase": "default_off", "seed": case["seed"],
                             "exact": all(row[p+"_comparison"]["exact"] for p in ("nr", "sample"))}), flush=True)
        aggregate = {policy: aggregate_summaries([row[policy] for row in result["default_off"]["cases"]])
                     for policy in ("nr", "sample")}
        result["default_off"].update(aggregate=aggregate, aggregate_comparison=differences(aggregate, reference["aggregate"]))
        env_on = PaperEnvironment(on_cfg, on_parts, bs=env_off.bs)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(route_cfg["model_seed"])
            route_model = SharedActorCritic(ppo_cfg).eval().requires_grad_(False)
        if model_identity(route_model) != route_reference["model_sha256"]:
            raise ValueError("Route model initialization does not match the audited reference")
        budget = {"used": 0, "maximum": cfg["maximum_csv_bytes"]}
        physics = CSVStream(output/"physics.csv", budget)
        unused_refresh = CSVStream(output/"unused_external_refresh.csv", budget)
        try:
            with torch.no_grad():
                for case_id in ROUTE_IDS:
                    expected = reference_routes[case_id]
                    case = dict(expected["case"], arm="native")
                    actual = run_case(env_on, route_model, route_cfg, case, physics, unused_refresh, output/case_id, deadline)
                    fields = ("geometry", "status", "initial_observation", "native_type_envelope", "pre_turn",
                              "environment_summary", "checks", "all_checks_passed", "repeated_action_count",
                              "selected_action", "lane_completed", "trace")
                    comparison = differences(scientific({k:actual.get(k) for k in fields}, route=True),
                                             scientific({k:expected[k] for k in fields}, route=True))
                    result["fixed_routes"].append({"case_id": case_id, "actual": actual, "comparison": comparison,
                        "integrated_audit": copy.deepcopy(env_on.navigation_audit),
                        "integrated_audit_passed": integrated_audit_ok(env_on.navigation_audit)})
                    save(output/"result.json", result)
        finally:
            physics.close()
            unused_refresh.close()
        result["fixed_physics_comparison"] = physics_reference_comparison(output/"physics.csv", args.physics_reference)
        for policy in ("nr", "sample"):
            paired = {"seed": 53001, "policy": policy}
            for name, env, external in (("integrated", env_on, False), ("external", env_off, True)):
                harness = PopulationHarness(env, external)
                try:
                    summary = _evaluate_case(harness, model, scenarios[0], policy, 863001, deadline)
                    flights = env.summary(include_flights=True)["flights"]
                finally:
                    harness.close()
                paired[name] = {"summary": summary, "flights": flights, "audit": harness.audit()}
            paired["comparison"] = differences(scientific({key:paired["integrated"][key] for key in ("summary", "flights")}),
                                                scientific({key:paired["external"][key] for key in ("summary", "flights")}))
            paired["stream_comparison"] = differences(
                {key:paired["integrated"]["audit"][key] for key in ("native_call_stream_sha256", "physical_state_stream_sha256", "physics_rows", "actual_native_calls")},
                {key:paired["external"]["audit"][key] for key in ("native_call_stream_sha256", "physical_state_stream_sha256", "physics_rows", "actual_native_calls")})
            result["populations"].append(paired)
            save(output/"result.json", result)
        result["model_before_sha256"], result["model_after_sha256"] = before_model, model_identity(model)
        checks = {"default_off_all_cases_exact": all(row[p+"_comparison"]["exact"] for row in result["default_off"]["cases"] for p in ("nr", "sample")),
            "default_off_never_wrapped": all(row[p+"_audit"]["context_entries"] == 0 for row in result["default_off"]["cases"] for p in ("nr", "sample")),
            "default_off_aggregate_exact": result["default_off"]["aggregate_comparison"]["exact"],
            "four_routes_exact": len(result["fixed_routes"]) == 4 and all(row["comparison"]["exact"] and row["integrated_audit_passed"] for row in result["fixed_routes"]),
            "all_fixed_physics_rows_exact": result["fixed_physics_comparison"]["exact"],
            "population_outputs_and_streams_exact": len(result["populations"]) == 2 and all(row["comparison"]["exact"] and row["stream_comparison"]["exact"] for row in result["populations"]),
            "population_native_once_resize_reset_restoration": all(row[name]["audit"]["all_checks_passed"] for row in result["populations"] for name in ("integrated", "external")),
            "model_unchanged_no_gradients": model_identity(model) == before_model and all(p.grad is None for p in model.parameters())}
        result.update(checks=checks, all_checks_passed=all(checks.values()))
    except Exception as exc:
        result.update(error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
    finally:
        result["wall_seconds"] = time.perf_counter()-started
        save(output/"result.json", result)
    print(canonical({"all_checks_passed": result["all_checks_passed"], "wall_seconds": result["wall_seconds"]}), flush=True)
    return 0 if result["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
