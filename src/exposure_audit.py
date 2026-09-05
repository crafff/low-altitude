"""Read-only NR instrumentation for raw pair exposure and denominator audits.

Pure counters import no simulator. The CLI uses the unchanged PaperEnvironment
path and writes artifacts only to LAB_RUN_DIR, under the controller's launcher.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager
import copy
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import platform
import traceback

from bluesky_diagnostic import FT, distance_m, risk_flags


LEVELS = ("potential", "lowc", "nmac")
CASES = tuple((53001 + i, 3 + i % 3) for i in range(12))
EXCLUDED_REFERENCE_FIELDS = ("wall_seconds", "process_peak_rss_mib")
OBSERVATION_FIELDS = ("record_kind", "scope", "sim_time_s", "frame_index", "ownship_entries",
                      "intruder_entries", "nonempty", "done")
PHYSICS_FIELDS = ("step_index", "t0_s", "dt_s", "active_aircraft", "potential_pairs", "lowc_pairs",
                  "nmac_pairs", "aircraft_seconds", "airspace_seconds", "occupied_airspace_seconds")


def normalized_exposure(raw):
    """Derive rates from raw sums. None denotes an absent denominator."""
    hours = {name: raw[name + "_seconds"] / 3600.0
             for name in ("aircraft", "airspace", "occupied_airspace")}
    exposure = {}
    for level in LEVELS:
        unordered = raw[level + "_pair_seconds"]
        values = {"unordered_pair_seconds": unordered, "directed_pair_seconds": 2 * unordered,
                  "continuous_unordered_event_count": raw[level + "_event_count"]}
        for denominator, value in hours.items():
            values["unordered_seconds_per_" + denominator + "_hour"] = unordered / value if value else None
            values["directed_seconds_per_" + denominator + "_hour"] = 2 * unordered / value if value else None
        exposure[level] = values
    potential = raw["potential_pair_seconds"]
    return {"raw": dict(raw), "hours": hours, "exposure": exposure,
            "exposure_ratios": {level + "_over_potential": raw[level + "_pair_seconds"] / potential
                                if potential else None for level in ("lowc", "nmac")},
            "ratio_scope": "Ratios of sampled pair-duration exposure; not independent-event probabilities."}


def aggregate_exposure(rows):
    """Pool raw numerators/denominators before dividing, including idle time."""
    if not rows:
        raise ValueError("Cannot aggregate an empty declared set")
    keys = set(rows[0]["raw"])
    if any(set(row["raw"]) != keys for row in rows):
        raise ValueError("Exposure rows use different raw fields")
    return normalized_exposure({key: sum(row["raw"][key] for row in rows) for key in sorted(keys)})


class ExposureCounter:
    """Integrate interval-start truth, retaining every sampled continuous pair event."""
    def __init__(self, risk_cfg, radius_ft=6000.0):
        if radius_ft != 6000.0:
            raise ValueError("This audit fixes the horizontal radius to 6000 ft")
        self.cfg, self.radius_m = dict(risk_cfg), radius_ft * FT
        self.raw = dict.fromkeys(("aircraft_seconds", "airspace_seconds", "occupied_airspace_seconds"), 0.0)
        self.raw["physics_steps"] = 0
        for level in LEVELS:
            self.raw[level + "_pair_seconds"] = 0.0
            self.raw[level + "_event_count"] = 0
        self.active = {level: {} for level in LEVELS}
        self.events, self.last_end = [], None

    def _close(self, level, pair, time_s, reason):
        self.events.append(self.active[level].pop(pair) | {"end_s": time_s, "end_reason": reason})

    def observe(self, time_s, states, dt):
        time_s, dt = float(time_s), float(dt)
        if not math.isfinite(time_s) or not math.isfinite(dt) or time_s < 0 or dt <= 0:
            raise ValueError("Exposure needs a finite nonnegative time and positive interval")
        if self.last_end is not None and not math.isclose(time_s, self.last_end, rel_tol=0, abs_tol=1e-8):
            raise ValueError("Record empty intervals explicitly; do not omit simulation time gaps")
        current = {level: set() for level in LEVELS}
        ids = sorted(states)
        for index, aid in enumerate(ids):
            for bid in ids[index + 1:]:
                a, b = states[aid], states[bid]
                distance = distance_m(a, b)
                if distance <= self.radius_m:
                    current["potential"].add((aid, bid))
                lowc, nmac = risk_flags(distance, a[2] - b[2], self.cfg)
                if lowc:
                    current["lowc"].add((aid, bid))
                if nmac:
                    current["nmac"].add((aid, bid))
        if not current["nmac"] <= current["lowc"] <= current["potential"]:
            raise ValueError("Physical safety layers are not nested inside the observation radius")
        for level, pairs in current.items():
            for pair in sorted(set(self.active[level]) - pairs):
                self._close(level, pair, time_s, "not_present_or_separated_at_next_interval_start")
            for pair in sorted(pairs):
                if pair not in self.active[level]:
                    self.active[level][pair] = {"level": level, "pair": list(pair), "start_s": time_s,
                                                "pair_seconds": 0.0}
                    self.raw[level + "_event_count"] += 1
                self.active[level][pair]["pair_seconds"] += dt
            self.raw[level + "_pair_seconds"] += len(pairs) * dt
        self.raw["physics_steps"] += 1
        self.raw["aircraft_seconds"] += len(states) * dt
        self.raw["airspace_seconds"] += dt
        self.raw["occupied_airspace_seconds"] += bool(states) * dt
        self.last_end = time_s + dt
        return {"step_index": self.raw["physics_steps"], "t0_s": time_s, "dt_s": dt,
                "active_aircraft": len(states), **{level + "_pairs": len(current[level]) for level in LEVELS},
                **{key: self.raw[key] for key in ("aircraft_seconds", "airspace_seconds", "occupied_airspace_seconds")}}

    def finish(self):
        for level in LEVELS:
            for pair in sorted(self.active[level]):
                self._close(level, pair, self.last_end, "last_integrated_interval_end")

    def summary(self):
        return normalized_exposure(self.raw)


def _entry_counts(observations):
    return len(observations), sum(len(row["intruder_ids"]) for row in observations.values())


class ObservationCounts:
    def __init__(self, writer=None):
        self.writer, self.phase = writer, "reset"
        self.constructed, self.returned, self.terminal_payload = {}, {}, Counter()

    def _record(self, category, scope, observations, time_s, done):
        group = self.constructed if category == "constructed" else self.returned
        counts = group.setdefault(scope, Counter())
        ownships, intruders = _entry_counts(observations)
        counts.update(frames=1, ownship_entries=ownships, intruder_entries=intruders,
                      nonempty_frames=int(bool(ownships)))
        if self.writer is not None:
            self.writer.writerow({"record_kind": category, "scope": scope, "sim_time_s": time_s,
                "frame_index": counts["frames"], "ownship_entries": ownships,
                "intruder_entries": intruders, "nonempty": bool(ownships), "done": bool(done)})

    def constructed_frame(self, env, observations):
        terminal_preview = any(env.records[acid]["status"] != "active" for acid in observations)
        scope = "terminal_preview" if terminal_preview else "reset" if self.phase == "reset" else "step_return"
        self._record("constructed", scope, observations, float(env.bs.sim.simt), env.done)

    def returned_frame(self, observations, time_s, *, reset=False, done=False, terminal_observations=None):
        self._record("returned", "reset" if reset else "step_return", observations, time_s, done)
        if terminal_observations is not None:
            ownships, intruders = _entry_counts(terminal_observations)
            self.terminal_payload.update(step_payloads=1, nonempty_payloads=int(bool(ownships)),
                                         ownship_entries=ownships, intruder_entries=intruders)

    def summary(self):
        def group(values):
            total = Counter()
            for value in values.values():
                total.update(value)
            return {"by_scope": {key: dict(value) for key, value in values.items()}, "total": dict(total)}
        return {"actual_policy_inference_calls": 0, "constructed": group(self.constructed),
                "returned_decision_frames": group(self.returned),
                "returned_terminal_payload": dict(self.terminal_payload),
                "scope": "Construction counts include full terminal previews before deletion. Returned frames include reset and the final empty frame. Terminal payloads are separate subsets. None of these counts measures inference or safety exposure."}


@contextmanager
def instrument_environment(env, exposure, observations, audit, physics_writer=None):
    """Observe two call facades; invoke each original once and restore in finally."""
    sim = env.bs.sim
    originals = [(sim, "step", sim.step), (env, "observations", env.observations)]
    members = [(obj, name, name in vars(obj), vars(obj).get(name)) for obj, name, _ in originals]
    native_step, construct = originals[0][2], originals[1][2]
    audit.update(step_wrapper_calls=0, native_step_calls=0, observation_wrapper_calls=0,
                 original_observation_calls=0, restored=False)

    def step(*args, **kwargs):
        audit["step_wrapper_calls"] += 1
        if env.bs.sim is not sim:
            raise RuntimeError("Simulation facade changed during exposure audit")
        before, t0 = env._positions(), float(sim.simt)
        audit["native_step_calls"] += 1
        result = native_step(*args, **kwargs)
        dt = float(sim.simt) - t0
        if not math.isclose(dt, env.dt, rel_tol=0, abs_tol=1e-8):
            raise RuntimeError("Audit native timestep differs from environment configuration")
        row = exposure.observe(t0, before, dt)
        if physics_writer is not None:
            physics_writer.writerow(row)
        return result

    def observe(*args, **kwargs):
        audit["observation_wrapper_calls"] += 1
        audit["original_observation_calls"] += 1
        result = construct(*args, **kwargs)
        observations.constructed_frame(env, result)
        return result

    try:
        vars(sim)["step"], vars(env)["observations"] = step, observe
        yield
    finally:
        audit["wrapper_entries_survived"] = vars(sim).get("step") is step and vars(env).get("observations") is observe
        for obj, name, existed, member in reversed(members):
            if existed:
                vars(obj)[name] = member
            else:
                vars(obj).pop(name, None)
        audit["restored"] = all(getattr(obj, name) == original for obj, name, original in originals)
        audit["original_instance_namespace_restored"] = all(
            (name in vars(obj)) == existed and (not existed or vars(obj)[name] is member)
            for obj, name, existed, member in members)


def compare_reference(actual, expected):
    """Exact comparison of original summary fields, without wall/RSS noise."""
    differences = []
    for key in sorted(set(actual) | set(expected)):
        if key not in EXCLUDED_REFERENCE_FIELDS and (key not in actual or key not in expected or actual[key] != expected[key]):
            differences.append({"field": key, "actual": actual.get(key), "reference": expected.get(key)})
    return {"exact_scientific_match": not differences, "excluded_fields": list(EXCLUDED_REFERENCE_FIELDS),
            "compared_fields": sorted((set(actual) | set(expected)) - set(EXCLUDED_REFERENCE_FIELDS)),
            "differences": differences}


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if (cfg["schema"] != "bluesky.exposure-audit.v1" or cfg["potential_radius_ft"] != 6000
            or cfg["physics_dt_seconds"] != .25 or cfg["decision_seconds"] != 5
            or cfg["reference_line"] != 1
            or cfg["reference_input"] != "checkpoints/execution-pilot-20260905/development.jsonl"
            or tuple((c["seed"], c["corridor_count"]) for c in cfg["development_cases"]) != CASES
            or cfg["environment_config"] != "configs/paper_environment_execution.json"):
        raise ValueError("Keep the fixed NR execution configuration, radius and twelve development cases")
    return cfg


def load_reference(path):
    with Path(path).open() as handle:
        first = handle.readline()
    data = json.loads(first)
    if tuple((case["seed"], case["corridor_count"]) for case in data["cases"]) != CASES:
        raise ValueError("The first reference line must contain the original twelve development cases")
    if any(case["nr"]["policy"] != "nr" or case["nr"]["policy_decisions"] != 0 for case in data["cases"]):
        raise ValueError("Reference must contain NR with zero policy decisions")
    return {case["seed"]: case["nr"] for case in data["cases"]}, {
        "path": str(path), "line": 1, "file_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "line_sha256": hashlib.sha256(first.encode()).hexdigest(), "phase": data.get("phase")}


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def run_case(env, case, scenario, reference, cfg, output, identities):
    output.mkdir()
    _write(output / "scenario.json", scenario)
    exposure = ExposureCounter(env.scenario_cfg, cfg["potential_radius_ft"])
    instrumentation, result = {}, dict(case, status="error", source_identity=identities,
        physics_dt_seconds=env.dt, decision_seconds=env.decision_dt, potential_radius_ft=6000.0,
        aircraft_type_counts=dict(Counter(f["type"] for f in scenario["flights"])))
    with (output / "physics_exposure.csv").open("w", newline="") as physics_file, (output / "observation_counts.csv").open("w", newline="") as observation_file:
        physics_writer = csv.DictWriter(physics_file, fieldnames=PHYSICS_FIELDS)
        observation_writer = csv.DictWriter(observation_file, fieldnames=OBSERVATION_FIELDS)
        physics_writer.writeheader()
        observation_writer.writeheader()
        observations = ObservationCounts(observation_writer)
        try:
            with instrument_environment(env, exposure, observations, instrumentation, physics_writer):
                frame = env.reset(case["seed"], scenario=copy.deepcopy(scenario))
                observations.returned_frame(frame, float(env.bs.sim.simt), reset=True, done=env.done)
                observations.phase = "step"
                while not env.done:
                    frame, _, _, info = env.step(None)
                    observations.returned_frame(frame, info["sim_time_s"], done=info["done"],
                                                 terminal_observations=info["terminal_observations"])
            summary = dict(env.summary(), policy="nr", action_seed=None, action_histogram=[0] * 60)
            result.update(status="completed", nr=summary, reference_comparison=compare_reference(summary, reference))
            _write(output / "native_events.json", env.tracker.events)
            _write(output / "flights.json", list(env.records.values()))
            result["checks"] = {
                "aircraft_seconds_equal_env_flight_seconds": exposure.raw["aircraft_seconds"] == env.flight_seconds,
                "aircraft_seconds_equal_flight_record_sum": exposure.raw["aircraft_seconds"] == sum(r["flight_seconds"] for r in env.records.values()),
                "airspace_seconds_equal_env_sim_seconds": exposure.raw["airspace_seconds"] == summary["sim_seconds"],
                "occupied_time_within_complete_airspace_time": 0 <= exposure.raw["occupied_airspace_seconds"] <= exposure.raw["airspace_seconds"],
                "physics_step_counts_match": exposure.raw["physics_steps"] == summary["physics_steps"] == instrumentation["native_step_calls"],
                "native_step_called_once_per_wrapper": instrumentation["step_wrapper_calls"] == instrumentation["native_step_calls"],
                "original_observations_called_once_per_wrapper": instrumentation["observation_wrapper_calls"] == instrumentation["original_observation_calls"],
                "returned_frames_equal_steps_plus_reset": observations.summary()["returned_decision_frames"]["total"]["frames"] == summary["decision_steps"] + 1,
                "constructed_reset_and_step_returns_match": observations.constructed.get("reset", {}).get("frames", 0) == 1 and observations.constructed.get("step_return", {}).get("frames", 0) == summary["decision_steps"],
                "zero_nr_policy_inference_and_actions": summary["policy_decisions"] == summary["changed_instructions"] == 0,
                "exposure_layers_nested": exposure.raw["nmac_pair_seconds"] <= exposure.raw["lowc_pair_seconds"] <= exposure.raw["potential_pair_seconds"],
                "exact_reference_nr": result["reference_comparison"]["exact_scientific_match"],
                "wrappers_restored": instrumentation["restored"] and instrumentation["original_instance_namespace_restored"] and instrumentation["wrapper_entries_survived"],
            }
            for level in ("lowc", "nmac"):
                native = summary["risk"][level]
                result["checks"][level + "_raw_and_events_match_environment"] = (
                    native["unordered_pair_seconds"] == exposure.raw[level + "_pair_seconds"]
                    and native["directed_pair_seconds"] == 2 * exposure.raw[level + "_pair_seconds"]
                    and native["event_count"] == exposure.raw[level + "_event_count"])
            result["checks"]["all_directed_exposures_equal_twice_unordered"] = all(
                value["directed_pair_seconds"] == 2 * value["unordered_pair_seconds"]
                for value in exposure.summary()["exposure"].values())
        except Exception as exc:
            result.update(status="error", error_type=type(exc).__name__, error=str(exc), traceback=traceback.format_exc())
        finally:
            exposure.finish()
            result.update(metrics=exposure.summary(), observation_counts=observations.summary(), instrumentation=instrumentation)
            result["all_checks_passed"] = result["status"] == "completed" and all(result["checks"].values())
            _write(output / "sampled_events.json", exposure.events)
            _write(output / "result.json", result)
    return result


def _identities(config_path, cfg, env_cfg):
    source = Path(__file__).resolve().parent
    names = ("exposure_audit.py", "paper_environment.py", "paper_actions.py", "paper_observation.py",
             "paper_scenarios.py", "paper_performance.py", "nr_pilot.py", "bluesky_diagnostic.py", "route_completion.py")
    paths = [Path(config_path), Path(cfg["environment_config"])] + [Path(env_cfg[key]) for key in
             ("scenario_config", "action_config", "observation_config", "types_config")]
    import bluesky
    native = Path(bluesky.__file__).resolve().parent
    native_names = ("simulation/simulation.py", "traffic/traffic.py", "traffic/autopilot.py", "traffic/activewpdata.py")
    return {"python": platform.python_version(), "bluesky": version("bluesky-simulator"),
            "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names},
            "native_source_sha256": {name: hashlib.sha256((native / name).read_bytes()).hexdigest() for name in native_names},
            "config_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/exposure_audit.json")
    parser.add_argument("--input", required=True, help="Controller-declared development.jsonl; only its first line is used")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if Path(args.input).as_posix() != cfg["reference_input"]:
        parser.error("--input must name the explicitly configured reference, with no run discovery")
    reference, reference_identity = load_reference(args.input)
    output = Path(os.environ["LAB_RUN_DIR"])
    from paper_environment import PaperEnvironment, load_environment_config
    from paper_scenarios import generate_scenario
    env_cfg, parts = load_environment_config(cfg["environment_config"])
    if (parts["scenario"]["dt_seconds"] != cfg["physics_dt_seconds"]
            or env_cfg["decision_seconds"] != cfg["decision_seconds"]
            or parts["observation"]["observation_radius_ft"] != cfg["potential_radius_ft"]):
        raise ValueError("Environment and audit clocks/radius differ")
    env = PaperEnvironment(env_cfg, parts)
    if len(env.types) != 12:
        raise ValueError("Keep the original twelve-type environment")
    identities = _identities(args.config, cfg, env_cfg)
    report = {"schema": cfg["schema"], "scope": cfg["scope"], "config": cfg,
              "environment_config": env_cfg, "effective_parts": parts, "aircraft_types": env.types,
              "reference": reference_identity, "source_identity": identities, "cases": [], "all_checks_passed": False}
    for case in cfg["development_cases"]:
        scenario = generate_scenario(dict(parts["scenario"], corridor_counts=[case["corridor_count"]]),
                                     case["seed"], list(env.types))
        result = run_case(env, case, scenario, reference[case["seed"]], cfg, output / str(case["seed"]), identities)
        report["cases"].append(result)
        _write(output / "result.json", report)
        print(json.dumps({"seed": case["seed"], "status": result["status"], "all_checks_passed": result["all_checks_passed"]}), flush=True)
    report["aggregates"] = {}
    for name, selected in (("all_twelve", report["cases"]),
                           ("existing_five_corridor_four", [r for r in report["cases"] if r["corridor_count"] == 5])):
        complete = all(row["status"] == "completed" for row in selected)
        report["aggregates"][name] = {"seeds": [row["seed"] for row in selected], "complete": complete,
            "all_reference_matches": complete and all(row["reference_comparison"]["exact_scientific_match"] for row in selected),
            "metrics": aggregate_exposure([row["metrics"] for row in selected]) if complete else None,
            "scope": "All declared cases required; partial failures are retained individually and never pooled as complete exposure."}
        if complete:
            additive = ("planned", "completed", "failed_timeout", "failed_route_exhausted", "outside_exit_crossings",
                        "path_length_m", "outside_corridor_aircraft_seconds", "outside_corridor_flights",
                        "outside_altitude_aircraft_seconds", "changed_instructions", "policy_decisions", "return_sum")
            report["aggregates"][name]["native_population_totals"] = {
                key: sum(row["nr"][key] for row in selected) for key in additive}
            report["aggregates"][name]["native_population_totals"]["max_centerline_distance_m"] = max(
                row["nr"]["max_centerline_distance_m"] for row in selected)
            report["aggregates"][name]["observation_counts"] = {
                group: {key: sum(row["observation_counts"][group]["total"].get(key, 0) for row in selected)
                        for key in ("frames", "ownship_entries", "intruder_entries", "nonempty_frames")}
                for group in ("constructed", "returned_decision_frames")}
            report["aggregates"][name]["observation_counts"]["actual_policy_inference_calls"] = 0
    report["all_checks_passed"] = all(row["all_checks_passed"] for row in report["cases"])
    _write(output / "result.json", report)
    return 0 if report["all_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
