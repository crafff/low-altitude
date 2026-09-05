"""Seven predeclared, single-aircraft native route/action diagnostics.

Uses PaperEnvironment without changing guidance, physics or action handling.
The controller executes this entry point through tools/lab.py; all study
artifacts are children of LAB_RUN_DIR. This is an execution-defect study,
not a learned-policy evaluation or a corridor-containment correction.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import traceback

from bluesky_diagnostic import FT, distance_m
from paper_actions import RouteGeometry, decode_action
from paper_environment import PaperEnvironment, load_environment_config
from paper_scenarios import _bearing, _direct


CASE_SPECS = (
    ("mavic_center", "Mavic", 37, "at_start"),
    ("mavic_inner", "Mavic", 36, "at_start"),
    ("mavic_outer", "Mavic", 38, "at_start"),
    ("amzn_center", "Amzn", 37, "at_start"),
    ("amzn_inner", "Amzn", 36, "at_start"),
    ("amzn_outer", "Amzn", 38, "at_start"),
    ("amzn_late_after_native_advance", "Amzn", 36, "late_after_native_advance"),
)


def load_study_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg["schema"] != "bluesky.route-action-study.v1":
        raise ValueError("Expected route-action-study.v1 configuration")
    if (cfg["origin_lat_lon_deg"] != [0.0, 0.0]
            or cfg["route_length_m"] != 9260.0
            or cfg["leg_bearings_deg"] != [90.0, 0.0]
            or cfg["earth_radius_m"] != 6371000.0):
        raise ValueError("This study requires two equal 4630 m east/north legs at the equator")
    actual = tuple((c["id"], c["aircraft_type"], c["action"], c["trigger"]) for c in cfg["cases"])
    if actual != CASE_SPECS:
        raise ValueError("Keep the seven predeclared cases; --cases selects a subset")
    return cfg


def select_cases(cfg, requested=None):
    names = [c["id"] for c in cfg["cases"]] if requested is None else requested.split(",")
    known = {c["id"]: c for c in cfg["cases"]}
    if not names or len(names) != len(set(names)) or any(name not in known for name in names):
        raise ValueError("--cases must contain distinct predeclared case IDs separated by commas")
    return [known[name] for name in names]


def constructed_scenario(cfg, case):
    """Exact spherical equal legs; no traffic/scenario sampling or dynamics."""
    points = [list(cfg["origin_lat_lon_deg"])]
    for bearing in cfg["leg_bearings_deg"]:
        points.append(_direct(points[-1], bearing, cfg["route_length_m"] / 2,
                              cfg["earth_radius_m"]))
    return {"seed": cfg["seed"], "scope": "constructed route/action execution-defect diagnostic",
            "corridors": [{"id": "C0", "waypoints_lat_lon_deg": points}],
            "flights": [{"id": "F0", "corridor_id": "C0", "type": case["aircraft_type"],
                         "scheduled_entry_s": 0.0}]}


def geometry_summary(scenario, action_cfg):
    points = scenario["corridors"][0]["waypoints_lat_lon_deg"]
    geometry = RouteGeometry(points)
    incoming = (_bearing(points[1], points[0]) + 180.0) % 360.0
    outgoing = _bearing(points[1], points[2])
    turn = (outgoing - incoming + 180.0) % 360.0 - 180.0
    plans = {}
    for label, action in (("center", 37), ("inner", 36), ("outer", 38)):
        speed, altitude, lane = decode_action(action)
        if (speed, altitude) != (2, 2):
            raise ValueError("Diagnostic actions must retain nominal speed and altitude")
        offset = action_cfg["lane_fractions"][lane] * action_cfg["corridor_width_ft"] * FT
        plans[label] = {"action": action, "target_lane_m": offset,
                        "offset_vertices_lat_lon_deg": [geometry.to_latlon(p) for p in geometry.offset(offset)]}
    if not math.isclose(turn, -90.0, abs_tol=1e-7) or not plans["inner"]["target_lane_m"] < 0 < plans["outer"]["target_lane_m"]:
        raise ValueError("Inner/outer action sign does not match the constructed left turn")
    lengths = [distance_m(a, b) for a, b in zip(points, points[1:])]
    if any(not math.isclose(length, 4630.0, abs_tol=1e-6) for length in lengths):
        raise ValueError("Constructed geodesic leg length differs from 4630 m")
    return {"nominal_waypoints_lat_lon_deg": points, "great_circle_leg_lengths_m": lengths,
            "great_circle_total_length_m": sum(lengths), "signed_turn_deg": turn,
            "incoming_course_at_corner_deg": incoming, "outgoing_course_at_corner_deg": outgoing,
            "lane_sign": "Positive is right of travel; negative is inside this east-to-north left turn.",
            "parallel_plan_geometry": plans}


def requested_action(case, nominal_waypoint_index, already_triggered):
    """Called only at environment decision boundaries; never delays a command."""
    trigger = already_triggered or case["trigger"] == "at_start" or nominal_waypoint_index >= 2
    return (case["action"] if trigger else 37), trigger


def _native_plan(route, fields):
    indices = {p["name"]: p["nominal_index"] for p in fields["native_route_plan"]}
    return [{"name": name, "latitude_deg": float(route.wplat[j]),
             "longitude_deg": float(route.wplon[j]), "altitude_m": float(route.wpalt[j]),
             "speed_constraint_mps": float(route.wpspd[j]),
             "flyby": bool(route.wpflyby[j]), "flyturn": bool(route.wpflyturn[j]),
             "nominal_index": indices.get(name)} for j, name in enumerate(route.wpname)]


def _sample(env, case, row_index, decision_time, proposed_action, previous):
    from bluesky.tools.aero import g0

    traf, acid = env.bs.traf, "F0"
    i = traf.id2idx(acid)
    if i < 0 or traf.ntraf != 1:
        raise RuntimeError("Study requires exactly its one planned aircraft until terminal deletion")
    route = traf.ap.route[i]
    active = int(route.iactwp)
    if not 0 <= active < len(route.wpname):
        raise RuntimeError("Native active waypoint is invalid")
    fields = env.actions.state_fields(acid)
    position = (float(traf.lat[i]), float(traf.lon[i]))
    nominal = env.scenario["corridors"][0]["waypoints_lat_lon_deg"]
    time = float(env.bs.sim.simt)
    endpoint = distance_m(position, fields["destination_waypoint"])
    arrived = bool(fields["final_nominal_active"] and endpoint <= env.scenario_cfg["arrival_radius_m"])
    timeout = time - env.records[acid]["actual_entry_s"] >= env.scenario_cfg["per_flight_timeout_seconds"] - 1e-8
    bankdef, turnphi = float(traf.ap.bankdef[i]), float(traf.ap.turnphi[i])
    bank = turnphi if turnphi > float(traf.eps[i]) ** 2 else bankdef
    capture = next((p for p in fields["native_route_plan"] if p["nominal_index"] is None), None)
    row = {
        "row_index": row_index, "sample_phase": "initial_before_action" if previous is None else "post_physics_post_action_update",
        "sim_time_s": time, "case_id": case["id"], "aircraft": acid,
        "decision_time_s": decision_time, "proposed_action_for_interval": proposed_action,
        "lat_deg": position[0], "lon_deg": position[1], "alt_m": float(traf.alt[i]),
        "tas_mps": float(traf.tas[i]), "ground_speed_mps": float(traf.gs[i]),
        "actual_hdg_deg": float(traf.hdg[i]), "actual_track_deg": float(traf.trk[i]),
        "target_hdg_deg": float(traf.aporasas.hdg[i]) % 360.0,
        "target_track_deg": float(traf.ap.trk[i]) % 360.0,
        "bank_default_deg": math.degrees(bankdef), "bank_turnphi_deg": math.degrees(turnphi),
        "bank_limit_deg": math.degrees(bank), "observed_heading_rate_deg_s": None,
        "equivalent_kinematic_bank_deg": None, "heading_rate_limited": bool(traf.swhdgsel[i]),
        "active_waypoint_index": active, "active_waypoint_name": route.wpname[active],
        "active_waypoint_lat_deg": float(traf.actwp.lat[i]), "active_waypoint_lon_deg": float(traf.actwp.lon[i]),
        "active_waypoint_distance_m": distance_m(position, (float(traf.actwp.lat[i]), float(traf.actwp.lon[i]))),
        "guidance_distance_before_motion_m": float(traf.ap.dist2wp[i]),
        "turn_initiation_distance_m": float(traf.actwp.turndist[i]),
        "native_turn_radius_m": float(traf.actwp.turnrad[i]),
        "active_leg_direction_deg": float(traf.actwp.curlegdir[i]),
        "flyby": bool(traf.actwp.flyby[i]), "flyturn": bool(traf.actwp.flyturn[i]),
        "lnav": bool(traf.swlnav[i]), "vnav": bool(traf.swvnav[i]),
        "nominal_corner_distance_m": distance_m(position, nominal[1]),
        "nominal_endpoint_distance_m": distance_m(position, nominal[-1]),
        "accepted_endpoint_distance_m": endpoint,
        "outside_corridor": fields["centerline_distance_m"] > env.scenario_cfg["corridor_width_ft"] * FT / 2,
        "capture_waypoint_name": None if capture is None else capture["name"],
        "capture_waypoint_distance_m": None if capture is None else distance_m(position, (capture["latitude_deg"], capture["longitude_deg"])),
        "capture_waypoint_active": capture is not None and capture["name"] == route.wpname[active],
        "controller_route_plan_json": json.dumps(fields["native_route_plan"], separators=(",", ":"), allow_nan=False),
        "native_route_plan_json": json.dumps(_native_plan(route, fields), separators=(",", ":"), allow_nan=False),
        "record_status_before_terminal_check": env.records[acid]["status"],
        "arrival_guard": arrived, "timeout_guard": timeout,
        "terminal_reason_from_guards": "arrived" if arrived else "flight_timeout" if timeout else None,
        **{name: fields[name] for name in (
            "accepted_action_index", "target_speed_mps", "nominal_speed_mps", "target_alt_m", "target_lane_m",
            "altitude_active", "lane_active", "nominal_waypoint_index", "nominal_course_deg",
            "actual_cross_track_m", "lane_error_m", "lane_track_error_deg", "on_parallel_segment",
            "centerline_distance_m", "capture_beyond_leg_end", "final_nominal_active")},
        **{name: fields["command_counts"][name] for name in (
            "lane_commands", "lane_captures", "route_rebuilds", "capture_beyond_leg_end_commands")},
    }
    if previous is not None:
        dt = time - previous["sim_time_s"]
        if not math.isclose(dt, env.dt, abs_tol=1e-8):
            raise RuntimeError(f"Trace did not receive every physics step: {dt}")
        rate = ((row["actual_hdg_deg"] - previous["actual_hdg_deg"] + 180.0) % 360.0 - 180.0) / dt
        row["observed_heading_rate_deg_s"] = rate
        row["equivalent_kinematic_bank_deg"] = math.degrees(math.atan(math.radians(rate) * row["tas_mps"] / g0))
    if any(isinstance(value, float) and not math.isfinite(value) for value in row.values()):
        raise RuntimeError("Nonfinite native trace value")
    return row


def _context(row):
    return {key: value for key, value in row.items() if not key.endswith("_json")}


class Trace:
    def __init__(self, handle, case):
        self.handle, self.case = handle, case
        self.writer = self.last = self.peak = self.first_native_advance = None
        self.decision_time = self.proposed_action = None
        self.rows = 0
        self.outside_seconds = self.lock_seconds = 0.0
        self.minimum_corner_distance_m = float("inf")
        self.max_abs_lane_error_m = 0.0
        self.route_versions, self.waypoint_switches, self.lock_completions = [], [], []
        self.captures = {}
        self.first_after_decision = {}

    def record(self, env):
        row = _sample(env, self.case, self.rows, self.decision_time, self.proposed_action, self.last)
        if self.writer is None:
            self.writer = csv.DictWriter(self.handle, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)
        self.handle.flush()  # A resource interruption still leaves the latest complete sample.
        self.rows += 1
        if self.decision_time is not None and self.decision_time not in self.first_after_decision:
            self.first_after_decision[self.decision_time] = _context(row)
        self.minimum_corner_distance_m = min(self.minimum_corner_distance_m, row["nominal_corner_distance_m"])
        self.max_abs_lane_error_m = max(self.max_abs_lane_error_m, abs(row["lane_error_m"]))
        if self.peak is None or row["centerline_distance_m"] > self.peak["centerline_distance_m"]:
            self.peak = _context(row)
        if self.first_native_advance is None and row["nominal_waypoint_index"] >= 2:
            self.first_native_advance = _context(row)
        if self.last is None or row["native_route_plan_json"] != self.last["native_route_plan_json"]:
            self.route_versions.append({"first_observed_time_s": row["sim_time_s"],
                "decision_time_s": self.decision_time,
                "native_plan": json.loads(row["native_route_plan_json"]),
                "controller_plan": json.loads(row["controller_route_plan_json"]), "first_row": _context(row)})
        if self.last is not None:
            dt = row["sim_time_s"] - self.last["sim_time_s"]
            self.outside_seconds += dt * row["outside_corridor"]
            self.lock_seconds += dt * row["lane_active"]
            if row["active_waypoint_name"] != self.last["active_waypoint_name"]:
                self.waypoint_switches.append({"before": _context(self.last), "after": _context(row)})
            if row["lane_captures"] > self.last["lane_captures"]:
                self.lock_completions.append(_context(row))
        if row["capture_waypoint_name"] is not None:
            capture = self.captures.setdefault(row["capture_waypoint_name"], {
                "first_observed_time_s": row["sim_time_s"], "first_observed_distance_m": row["capture_waypoint_distance_m"],
                "minimum_sampled_distance_m": row["capture_waypoint_distance_m"],
                "first_active_sample_s": None, "last_active_sample_s": None})
            capture["minimum_sampled_distance_m"] = min(capture["minimum_sampled_distance_m"], row["capture_waypoint_distance_m"])
            if row["capture_waypoint_active"]:
                if capture["first_active_sample_s"] is None:
                    capture["first_active_sample_s"] = row["sim_time_s"]
                capture["last_active_sample_s"] = row["sim_time_s"]
        self.last = row

    def summary(self):
        return {"trace_rows": self.rows, "physics_samples": max(0, self.rows - 1),
                "peak_deviation_row": self.peak, "terminal_sample": None if self.last is None else _context(self.last),
                "post_step_outside_corridor_seconds": self.outside_seconds,
                "post_step_lane_lock_seconds": self.lock_seconds, "max_abs_lane_error_m": self.max_abs_lane_error_m,
                "minimum_corner_distance_m": None if self.last is None else self.minimum_corner_distance_m,
                "first_native_advance": self.first_native_advance,
                "lane_lock_completions": self.lock_completions, "capture_waypoints": self.captures,
                "waypoint_switches": self.waypoint_switches, "route_plan_versions": self.route_versions}


def _write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def run_case(env, cfg, case, output):
    output.mkdir()
    scenario = constructed_scenario(cfg, case)
    geometry = geometry_summary(scenario, env.action_cfg)
    _write_json(output / "scenario.json", scenario)
    decisions, command = [], None
    result = {"case": case, "geometry": geometry, "status": "not_started"}
    with (output / "physics.csv").open("x", newline="") as handle:
        trace = Trace(handle, case)
        try:
            env.on_physics_step = None
            observations = env.reset(cfg["seed"], scenario=scenario)
            trace.record(env)
            env.on_physics_step = trace.record
            triggered = False
            while not env.done:
                row = trace.last
                action, now_triggered = requested_action(case, row["nominal_waypoint_index"], triggered)
                if set(observations) != {"F0"} or not observations["F0"]["action_mask"][action]:
                    raise RuntimeError("Scripted action is unavailable; no replacement action is allowed")
                context = _context(row)
                context.update(proposed_action=action, trigger_satisfied=now_triggered,
                               decision_time_s=float(env.bs.sim.simt))
                if now_triggered and not triggered:
                    command = {"requested_action": action, "before_apply": context,
                               "first_post_apply_sample": None}
                decisions.append(context)
                triggered = now_triggered
                trace.decision_time, trace.proposed_action = float(env.bs.sim.simt), action
                observations, _, _, _ = env.step({"F0": action})
                if command is not None and command["first_post_apply_sample"] is None:
                    # Include unchanged center commands, not just route rebuilds;
                    # this is the first physics sample, not the interval end.
                    command["first_post_apply_sample"] = trace.first_after_decision.get(trace.decision_time)
            summary = env.summary(include_flights=True)
            flight = summary["flights"][0]
            result.update(status=flight["status"], arrived=flight["status"] == "arrived",
                          timed_out=flight["status"] == "flight_timeout", environment=summary,
                          command_triggered=triggered,
                          lane_completion_required=case["action"] != 37,
                          lane_completed=bool(flight["action_execution"]["command_counts"]["lane_captures"]),
                          terminal_guards_match_record=trace.last["terminal_reason_from_guards"] == flight["status"])
        except Exception as exc:
            result.update(status="error", arrived=False, timed_out=False,
                          error={"type": type(exc).__name__, "message": str(exc)})
            (output / "error.txt").write_text(traceback.format_exc())
        finally:
            env.on_physics_step = None
            result.update(trace=trace.summary(), command=command)
            _write_json(output / "decisions.json", decisions)
            _write_json(output / "result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/route_action_study.json")
    parser.add_argument("--cases", help="Comma-separated subset of predeclared case IDs")
    args = parser.parse_args()
    cfg = load_study_config(args.config)
    cases = select_cases(cfg, args.cases)
    environment_cfg, parts = load_environment_config(cfg["environment_config"])
    if (environment_cfg["decision_seconds"] != 5.0 or parts["scenario"]["dt_seconds"] != .25
            or parts["scenario"]["altitude_ft"] != 350.0 or parts["action"]["nominal_altitude_ft"] != 350.0):
        raise ValueError("Study must use the shared 5 s/0.25 s, 350 ft baseline environment")
    output = Path(os.environ["LAB_RUN_DIR"])
    if not output.is_absolute() or not output.is_dir():
        raise ValueError("LAB_RUN_DIR must be the launcher's existing absolute output directory")
    result = {"scope": "native route/action execution-defect diagnostic; no fixes or policy training",
              "config": cfg, "environment_config": environment_cfg, "effective_parts": parts,
              "selected_cases": [case["id"] for case in cases], "results": [],
              "trace_semantics": {
                  "sampling": "Initial pre-action state then each 0.25 s after native motion and ActionController.update, before environment terminal deletion.",
                  "bank": "Native bank_limit selects turnphi or bankdef; BlueSky has no measured roll state. Signed equivalent bank is inferred from sampled heading rate.",
                  "commands": "Decision context is immediately before env.step applies the action. Plan versions first appear at the next physics sample and can already have advanced past CAP.",
                  "arrival": "Trace terminal guards are previews; confirmed arrival/timeout is the environment flight record. Boundary arrival takes precedence over timeout, as in the environment.",
                  "deviation": "Post-step distance to the full nominal polyline and signed current-leg lane error answer different questions; both are retained.",
                  "failures": "Timeouts, width violations, unresolved locks and runtime exceptions remain in results. Successful process completion does not mean corridor containment."}}
    _write_json(output / "effective_config.json", {k: result[k] for k in ("config", "environment_config", "effective_parts", "selected_cases", "trace_semantics")})
    try:
        env = PaperEnvironment(environment_cfg, parts)
    except Exception as exc:
        result["initialization_error"] = {"type": type(exc).__name__, "message": str(exc)}
        (output / "error.txt").write_text(traceback.format_exc())
        _write_json(output / "result.json", result)
        return 1
    for case in cases:
        case_result = run_case(env, cfg, case, output / case["id"])
        result["results"].append(case_result)
        print(json.dumps({"case": case["id"], "status": case_result["status"],
                          "command_triggered": case_result.get("command_triggered"),
                          "lane_completed": case_result.get("lane_completed"),
                          "peak_deviation_row": case_result["trace"]["peak_deviation_row"]}, allow_nan=False), flush=True)
    result["all_cases_terminal"] = all(r["status"] in ("arrived", "flight_timeout") for r in result["results"])
    result["all_arrived"] = all(r["status"] == "arrived" for r in result["results"])
    _write_json(output / "result.json", result)
    return 0 if result["all_cases_terminal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
