"""Independent public-speed and ordinary-flyby refresh sensitivity diagnostics.

Author source is inspected only during implementation, never fetched/imported
by this entry point. Three variants run serially in fresh Python processes under
one outer lab launcher; the native performance table remains immutable within
each process. All artifacts are written below LAB_RUN_DIR.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import csv
import hashlib
from importlib.metadata import version
import inspect
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np

from paper_environment import PaperEnvironment, load_environment_config
from paper_performance import load_types
from route_action_study import load_study_config, run_case, select_cases


VARIANTS = ("baseline", "amzn_public_44mps", "refresh_flyby_turn_distance")


def load_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg["schema"] != "bluesky.navigation-sensitivity.v1" or tuple(cfg["variants"]) != VARIANTS:
        raise ValueError("Expected the three independent navigation-sensitivity.v1 variants")
    timeout = cfg["variant_timeout_seconds"]
    if isinstance(timeout, bool) or not isinstance(timeout, (float, int)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("variant_timeout_seconds must be finite and positive")
    return cfg


def select_variants(requested):
    names = list(VARIANTS) if requested is None else requested.split(",")
    if not names or len(names) != len(set(names)) or any(name not in VARIANTS for name in names):
        raise ValueError("--variants must select distinct predeclared variants; no combined axis exists")
    return names


def validate_type_axis(baseline_path, alternative_path):
    baseline, alternative = load_types(baseline_path), load_types(alternative_path)
    if len(baseline) != 12 or set(baseline) != set(alternative) or "Amzn" not in baseline:
        raise ValueError("Both tables must contain the same twelve types")
    for name in baseline:
        if name != "Amzn" and baseline[name] != alternative[name]:
            raise ValueError(f"The speed variant changed another type: {name}")
    base, alt = baseline["Amzn"], alternative["Amzn"]
    changed = {key for key in base if base[key] != alt[key]}
    if changed != {"maximum_tas_mps", "nominal_tas_mps"}:
        raise ValueError("Only Amzn maximum and derived nominal TAS may change")
    for actual, expected in ((base["maximum_tas_mps"], 196 * 1852 / 3600),
                             (alt["maximum_tas_mps"], 44), (alt["nominal_tas_mps"], 35.2)):
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-10):
            raise ValueError("Amzn speed axis does not match the declared values")
    return baseline, alternative


def refreshed_flyby_distance(tas_mps, bank_rad, qdr_deg, next_qdr_deg, gravity_mps2):
    """Pure ordinary-flyby formula; return distance, radius, effective next qdr.

    This is scoped to the inspected old ordinary-flyby branch. Current TAS is
    deliberate; current cached turnspd can be a computed ordinary-flyby value.
    No flyturn branch, bank change or undocumented four-radius cap is added.
    """
    tas, bank, qdr, next_qdr = np.broadcast_arrays(*(
        np.asarray(value, dtype=float) for value in (tas_mps, bank_rad, qdr_deg, next_qdr_deg)))
    if not all(np.isfinite(value).all() for value in (tas, bank, qdr, next_qdr)) or np.any(tas < 0):
        raise ValueError("Flyby inputs must be finite with nonnegative TAS")
    if not math.isfinite(gravity_mps2) or gravity_mps2 <= 0:
        raise ValueError("Gravity must be finite and positive")
    following = np.where(next_qdr < -900.0, qdr, next_qdr)
    difference = (qdr % 360.0 - following % 360.0 + 180.0) % 360.0 - 180.0
    radius = tas * tas / (gravity_mps2 * np.maximum(.01, np.tan(bank)))
    distance = np.abs(radius * np.tan(np.radians(.5 * np.abs(difference))))
    if not np.isfinite(distance).all():
        raise ValueError("Flyby refresh produced a nonfinite distance")
    return distance, radius, following


@contextmanager
def refresh_reached(bs, gravity_mps2, audit, writer=None):
    """Wrap the actual call facade, including BlueSky's cached-method Proxy."""
    facade = bs.traf.actwp
    namespace = vars(facade)
    had_member, old_member = "reached" in namespace, namespace.get("reached")
    original = facade.reached
    audit.update(enabled=True, wrapper_calls=0, native_calls=0, flyby_entries_refreshed=0,
                 unsupported_calls=0, max_abs_cache_change_m=0.0, method_restored=False,
                 facade_class=f"{type(facade).__module__}.{type(facade).__qualname__}")

    def wrapped(qdr, dist):
        audit["wrapper_calls"] += 1
        if bs.traf.actwp is not facade:
            raise RuntimeError("ActiveWaypoint facade changed while refresh was installed")
        if any(np.asarray(getattr(facade, name), dtype=bool).any()
               for name in ("flyturn", "turnfromlastwp", "turntonextwp")):
            audit["unsupported_calls"] += 1
            raise RuntimeError("Flyturn and flyturn-transition modes are unsupported by this sensitivity")
        qdr_array, dist_array = np.asarray(qdr, dtype=float), np.asarray(dist, dtype=float)
        tas = np.asarray(bs.traf.tas, dtype=float)
        bank = np.asarray(bs.traf.ap.bankdef, dtype=float)
        next_qdr = np.asarray(facade.next_qdr, dtype=float)
        before = np.array(facade.turndist, dtype=float, copy=True)
        flyby = np.asarray(facade.flyby, dtype=bool)
        arrays = (qdr_array, dist_array, tas, bank, next_qdr, before, flyby)
        if any(array.shape != (len(bs.traf.id),) for array in arrays):
            raise RuntimeError("Native flyby arrays differ from traffic IDs")
        refreshed, radius, following = refreshed_flyby_distance(tas, bank, qdr_array, next_qdr, gravity_mps2)
        # This is the only pre-native mutation: leave flyover handling to native reached.
        facade.turndist[flyby] = refreshed[flyby]
        audit["flyby_entries_refreshed"] += int(flyby.sum())
        if flyby.any():
            audit["max_abs_cache_change_m"] = max(audit["max_abs_cache_change_m"],
                                                  float(np.max(np.abs(refreshed[flyby] - before[flyby]))))
        audit["native_calls"] += 1
        result = original(qdr, dist)  # Exactly once, with the original arguments and result.
        if writer is not None:
            reached_indices = set(int(index) for index in result)
            for i, acid in enumerate(bs.traf.id):
                writer.writerow({"call_index": audit["wrapper_calls"], "sim_time_s": float(bs.sim.simt),
                    "aircraft": acid, "qdr_deg": float(qdr_array[i]), "next_qdr_deg": float(next_qdr[i]),
                    "effective_next_qdr_deg": float(following[i]), "distance_to_waypoint_m": float(dist_array[i]),
                    "actual_tas_mps": float(tas[i]), "bank_default_deg": math.degrees(float(bank[i])),
                    "cached_turndist_before_m": float(before[i]), "computed_radius_m": float(radius[i]),
                    "refreshed_turndist_m": float(refreshed[i]) if flyby[i] else None,
                    "ordinary_flyby_refreshed": bool(flyby[i]), "native_reached": i in reached_indices})
        return result

    namespace["reached"] = wrapped
    try:
        yield audit
    finally:
        audit["wrapper_member_survived"] = namespace.get("reached") is wrapped
        if had_member:
            namespace["reached"] = old_member
        else:
            namespace.pop("reached", None)
        audit["method_restored"] = facade.reached == original


REFRESH_FIELDS = ("call_index", "sim_time_s", "aircraft", "qdr_deg", "next_qdr_deg",
                  "effective_next_qdr_deg", "distance_to_waypoint_m", "actual_tas_mps", "bank_default_deg",
                  "cached_turndist_before_m", "computed_radius_m", "refreshed_turndist_m",
                  "ordinary_flyby_refreshed", "native_reached")


def _identity(path):
    path = Path(path)
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _write_json(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _speed_summary(path, envelope):
    if not path.exists():
        return {"expected_envelope": envelope, "sample_count": 0}
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"expected_envelope": envelope, "sample_count": 0}
    fields = ("sim_time_s", "tas_mps", "ground_speed_mps", "nominal_speed_mps", "target_speed_mps", "bank_default_deg")
    first, last = ({key: float(row[key]) for key in fields} for row in (rows[0], rows[-1]))
    return {"expected_envelope": envelope, "sample_count": len(rows), "initial_sample": first, "last_sample": last,
            "actual_tas_min_mps": min(float(row["tas_mps"]) for row in rows),
            "actual_tas_max_mps": max(float(row["tas_mps"]) for row in rows),
            "initial_tas_matches_nominal": math.isclose(first["tas_mps"], envelope["nominal_tas_mps"], abs_tol=1e-3),
            "recorded_nominal_matches_table": all(math.isclose(float(row["nominal_speed_mps"]), envelope["nominal_tas_mps"], abs_tol=1e-10) for row in rows)}


def run_variant(cfg, config_path, variant, requested_cases, output):
    study = load_study_config(cfg["study_config"])
    cases = select_cases(study, requested_cases)
    environment, parts = load_environment_config(study["environment_config"])
    baseline, alternative = validate_type_axis(cfg["baseline_types_config"], cfg["public_amzn_types_config"])
    if environment["types_config"] != cfg["baseline_types_config"]:
        raise ValueError("Shared environment is not using the declared baseline type table")
    if variant == "amzn_public_44mps":
        environment = dict(environment, types_config=cfg["public_amzn_types_config"])
    selected_types = alternative if variant == "amzn_public_44mps" else baseline
    if version("bluesky-simulator") != cfg["expected_bluesky_version"]:
        raise ValueError("Native BlueSky version differs from the audited sensitivity target")
    paths = [config_path, cfg["study_config"], study["environment_config"], cfg["baseline_types_config"],
             cfg["public_amzn_types_config"], *(environment[f"{name}_config"] for name in ("scenario", "action", "observation")),
             "src/navigation_sensitivity.py", "src/paper_environment.py", "src/paper_actions.py",
             "src/paper_performance.py", "src/route_action_study.py"]
    result = {"schema": "bluesky.navigation-sensitivity-result.v1", "variant": variant,
              "scope": "One-axis development sensitivity, not the identified 2026 author implementation",
              "config": cfg, "study_config": study, "effective_environment": environment, "effective_parts": parts,
              "effective_types_si": selected_types, "source_identity": [_identity(path) for path in dict.fromkeys(paths)],
              "public_source_identity": cfg["public_sources"], "bluesky_version": version("bluesky-simulator"),
              "python_version": sys.version, "numpy_version": np.__version__,
              "selected_cases": [case["id"] for case in cases], "results": []}
    _write_json(output / "effective_config.json", {key: value for key, value in result.items() if key != "results"})
    try:
        env = PaperEnvironment(environment, parts)
        from bluesky.core.entity import getproxied
        from bluesky.traffic.route import Route
        from bluesky.tools.aero import g0
        native_classes = [type(getproxied(obj)) for obj in (env.bs.traf, env.bs.traf.ap, env.bs.traf.actwp)] + [Route]
        result["native_source_identity"] = [_identity(inspect.getsourcefile(cls)) for cls in native_classes]
    except Exception as exc:
        result["initialization_error"] = {"type": type(exc).__name__, "message": str(exc)}
        (output / "error.txt").write_text(traceback.format_exc())
        _write_json(output / "result.json", result)
        return 1
    for case in cases:
        case_output = output / case["id"]
        audit = {"enabled": False, "wrapper_calls": 0, "native_calls": 0, "method_restored": True}
        refresh_handle = None
        try:
            if variant == "refresh_flyby_turn_distance":
                refresh_handle = (output / f"{case['id']}-reached_refresh.csv").open("x", newline="", buffering=1)
                writer = csv.DictWriter(refresh_handle, fieldnames=REFRESH_FIELDS)
                writer.writeheader()
                context = refresh_reached(env.bs, float(g0), audit, writer)
            else:
                context = nullcontext()
            with context:
                raw = run_case(env, study, case, case_output)
        except Exception as exc:
            raw = {"case": case, "status": "error", "error": {"type": type(exc).__name__, "message": str(exc)}}
            (output / f"{case['id']}-error.txt").write_text(traceback.format_exc())
        finally:
            if refresh_handle is not None:
                refresh_handle.close()
        speed = _speed_summary(case_output / "physics.csv", selected_types[case["aircraft_type"]])
        hook_ok = not audit["enabled"] or (audit["wrapper_calls"] > 0 and audit["native_calls"] == audit["wrapper_calls"]
                                            and audit["method_restored"] and audit.get("wrapper_member_survived", False))
        sensitivity = {"variant": variant, "refresh_audit": audit, "speed": speed,
                       "hook_execution_verified": hook_ok,
                       "source_identity": result["source_identity"], "native_source_identity": result["native_source_identity"]}
        if case_output.is_dir():
            _write_json(case_output / "sensitivity.json", sensitivity)
        result["results"].append(dict(raw, sensitivity=sensitivity))
        print(json.dumps({"variant": variant, "case": case["id"], "status": raw["status"],
                          "hook_execution_verified": hook_ok, "initial_speed": speed.get("initial_sample")}), flush=True)
    result["all_cases_terminal"] = all(case["status"] in ("arrived", "flight_timeout") for case in result["results"])
    result["all_arrived"] = all(case["status"] == "arrived" for case in result["results"])
    result["hook_execution_verified"] = all(case["sensitivity"]["hook_execution_verified"] for case in result["results"])
    _write_json(output / "result.json", result)
    return 0 if result["all_cases_terminal"] and result["hook_execution_verified"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/navigation_sensitivity.json")
    parser.add_argument("--variants", help="Comma-separated subset of the three independent variants")
    parser.add_argument("--cases", help="Same comma-separated case subset for every selected variant")
    parser.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = load_config(args.config)
    variants = select_variants(args.variants)
    select_cases(load_study_config(cfg["study_config"]), args.cases)
    output = Path(os.environ["LAB_RUN_DIR"])
    if not output.is_absolute() or not output.is_dir():
        raise ValueError("LAB_RUN_DIR must be the launcher's existing absolute output directory")
    if args._worker:
        if len(variants) != 1:
            raise ValueError("Each fresh worker runs exactly one variant")
        return run_variant(cfg, args.config, variants[0], args.cases, output)
    result = {"schema": "bluesky.navigation-sensitivity-suite.v1", "config": cfg,
              "source_identity": _identity(args.config), "variants": variants, "cases": args.cases,
              "execution": "Sequential fresh Python children inside the same isolated launcher; no combined factor", "results": []}
    _write_json(output / "suite_config.json", result)
    for variant in variants:
        destination = output / variant
        destination.mkdir()
        command = [sys.executable, "-B", "-m", "navigation_sensitivity", "--config", args.config,
                   "--variants", variant, "--_worker"]
        if args.cases is not None:
            command.extend(("--cases", args.cases))
        record = {"variant": variant, "command": command}
        with (destination / "worker.log").open("x") as log:
            try:
                process = subprocess.run(command, env=dict(os.environ, LAB_RUN_DIR=str(destination)),
                                         stdout=log, stderr=subprocess.STDOUT, timeout=cfg["variant_timeout_seconds"], check=False)
                record["returncode"] = process.returncode
            except subprocess.TimeoutExpired:
                record.update(returncode=None, worker_timeout=True)
        if (destination / "result.json").is_file():
            record["result"] = json.loads((destination / "result.json").read_text())
        else:
            record["result_missing"] = True
        result["results"].append(record)
        print(json.dumps({"variant": variant, "returncode": record["returncode"],
                          "worker_timeout": record.get("worker_timeout", False),
                          "result_missing": record.get("result_missing", False)}), flush=True)
    result["all_workers_completed"] = all(record["returncode"] == 0 and not record.get("result_missing") for record in result["results"])
    _write_json(output / "result.json", result)
    return 0 if result["all_workers_completed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
