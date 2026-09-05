"""Stateful, reproducible perception perturbations; no simulator/action mutation.

ObservationPerturbation(cfg, seed=...).snapshot(states, time_s) samples the full
fleet once per five-second boundary. Every observer uses the same sensed state.
The frozen snapshot can encode available aircraft through the existing observe;
handling missing ownships in a policy/trajectory driver is deliberately external.
"""
from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass, fields, replace
import hashlib
import json
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np

from paper_observation import AircraftState, observe


STAT_KEYS = (
    "decision_snapshots", "eligible_aircraft_decisions", "none_selections",
    "position_triggers", "communication_triggers", "communication_retriggers",
    "blackout_segments_started", "blackout_segments_continued_at_expiry",
    "currently_unavailable", "categorical_draws", "position_noise_vectors",
    "communication_duration_draws",
)
STATE_FIELDS = tuple(field.name for field in fields(AircraftState))


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _primitive_copy(value):
    try:
        return json.loads(_canonical(value))
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Expected finite primitive JSON data") from exc


def _fingerprint(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def _seed(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("seed must be an explicit nonnegative integer")
    return value


def validate_config(cfg):
    cfg = _primitive_copy(cfg)
    if cfg["schema"] != "paper.observation-perturbation.v1" or cfg["rng_bit_generator"] != "PCG64":
        raise ValueError("Expected observation-perturbation.v1 with PCG64")
    if cfg["decision_seconds"] != 5.0:
        raise ValueError("This paper perception interface uses five-second decisions")
    p = _number(cfg["position_probability"], "position_probability")
    q = _number(cfg["communication_probability"], "communication_probability")
    if p < 0 or q < 0 or p + q > 1:
        raise ValueError("Position and communication probabilities must be nonnegative and sum to <=1")
    if _number(cfg["position_sigma_m"], "position_sigma_m") < 0:
        raise ValueError("Position sigma must be nonnegative")
    if (cfg["communication_duration_seconds"] != [5.0, 10.0, 15.0]
            or cfg["communication_duration_probabilities"] != [.25, .5, .25]):
        raise ValueError("Retain paper communication durations and probabilities")
    if _number(cfg["earth_radius_m"], "earth_radius_m") <= 0:
        raise ValueError("Earth radius must be positive")
    if not 0 < _number(cfg["pole_guard_degrees"], "pole_guard_degrees") < 1:
        raise ValueError("Pole guard must be positive and less than one degree")
    return cfg


def load_config(path):
    return validate_config(json.loads(Path(path).read_text()))


def displace_enu(state, east_m, north_m, *, earth_radius_m=6371000.0, pole_guard_degrees=1e-6):
    """Map an EN tangent vector to a spherical geodesic, changing only lat/lon."""
    east, north = _number(east_m, "east_m"), _number(north_m, "north_m")
    if east == 0.0 and north == 0.0:
        return state
    radius = _number(earth_radius_m, "earth_radius_m")
    guard = _number(pole_guard_degrees, "pole_guard_degrees")
    if radius <= 0 or not 0 < guard < 1:
        raise ValueError("Invalid spherical displacement parameters")
    if abs(state.lat_deg) >= 90.0 - guard:
        raise ValueError("Nonzero EN displacement has an ambiguous polar frame")
    lat, lon = math.radians(state.lat_deg), math.radians(state.lon_deg)
    arc, bearing = math.hypot(east, north) / radius, math.atan2(east, north)
    if not math.isfinite(arc) or arc >= math.pi:
        raise ValueError("Position error exceeds the local unambiguous spherical displacement domain")
    displaced_lat = math.asin(max(-1.0, min(1.0,
        math.sin(lat) * math.cos(arc) + math.cos(lat) * math.sin(arc) * math.cos(bearing))))
    displaced_lon = lon + math.atan2(math.sin(bearing) * math.sin(arc) * math.cos(lat),
                                     math.cos(arc) - math.sin(lat) * math.sin(displaced_lat))
    latitude = math.degrees(displaced_lat)
    longitude = (math.degrees(displaced_lon) + 180.0) % 360.0 - 180.0
    if abs(latitude) >= 90.0 - guard:
        raise ValueError("Displaced position lies in the ambiguous polar guard")
    return replace(state, lat_deg=latitude, lon_deg=longitude)


def _record(state):
    return {name: list(getattr(state, name)) if name in ("previous_waypoint", "next_waypoint")
            else getattr(state, name) for name in STATE_FIELDS}


def _ordered_states(states):
    if not isinstance(states, Mapping):
        raise ValueError("states must map stable aircraft IDs to AircraftState")
    for acid, state in states.items():
        if not isinstance(state, AircraftState) or acid != state.acid:
            raise ValueError("Each mapping key must match AircraftState.acid")
    return {acid: states[acid] for acid in sorted(states)}


def _decode_records(records):
    if not isinstance(records, list):
        raise ValueError("Cached aircraft records must be a list")
    result = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != set(STATE_FIELDS):
            raise ValueError("Cached AircraftState fields differ from the current API")
        state = AircraftState(**record)
        if state.acid in result:
            raise ValueError("Duplicate cached aircraft ID")
        result[state.acid] = state
    if list(result) != sorted(result):
        raise ValueError("Cached aircraft IDs are not sorted")
    return result


def _decision_time(time_s):
    time = _number(time_s, "time_s")
    if time < 0 or time % 5.0 != 0:
        raise ValueError("Decision time must be a nonnegative multiple of five seconds")
    return time


def _copy_rng(rng):
    other = np.random.Generator(np.random.PCG64(0))
    other.bit_generator.state = copy.deepcopy(rng.bit_generator.state)
    return other


def _available(intervals, acid, time):
    return not any(start <= time < end for start, end in intervals.get(acid, ()))


@dataclass(frozen=True)
class PerturbationSnapshot:
    time_s: float
    sensed_states: Mapping[str, AircraftState]
    available_ids: tuple[str, ...]
    events: tuple[Mapping, ...]
    until: Mapping[str, float]
    statistics: Mapping[str, int]

    @property
    def visible_states(self):
        return MappingProxyType({acid: self.sensed_states[acid] for acid in self.available_ids})

    def encode(self, observation_cfg):
        """Encode available aircraft only; absent ownship control is external."""
        return observe(self.visible_states, observation_cfg)


def _snapshot(time, sensed, available, events, until, statistics):
    return PerturbationSnapshot(time, MappingProxyType(dict(sensed)), tuple(available),
                                tuple(MappingProxyType(dict(event)) for event in events),
                                MappingProxyType(dict(sorted(until.items()))),
                                MappingProxyType(dict(statistics)))


class ObservationPerturbation:
    def __init__(self, cfg, *, seed):
        self._cfg = validate_config(cfg)
        self._config_fingerprint = _fingerprint(self._cfg)
        self.reset(seed=seed)

    def reset(self, *, seed):
        """Start a new episode at t=0; never seed or draw from external RNGs."""
        self._seed = _seed(seed)
        self._rng = np.random.Generator(np.random.PCG64(self._seed))
        self._time = self._cache = self._truth_records = self._input_fingerprint = None
        self._until, self._intervals = {}, {}
        self._stats = dict.fromkeys(STAT_KEYS, 0)

    @property
    def statistics(self):
        return MappingProxyType(dict(self._stats))

    def availability_at(self, acid, time_s):
        """Read registered half-open blackout intervals; does not infer presence.

        A future query includes only already registered events. It neither
        samples future decisions nor provides a forecast of new blackouts.
        """
        if not isinstance(acid, str) or not acid:
            raise ValueError("acid must be a nonempty stable ID")
        time = _number(time_s, "time_s")
        if time < 0:
            raise ValueError("Availability query time must be nonnegative")
        return _available(self._intervals, acid, time)

    def _shift(self, state, east, north):
        return displace_enu(state, east, north, earth_radius_m=self._cfg["earth_radius_m"],
                            pole_guard_degrees=self._cfg["pole_guard_degrees"])

    def snapshot(self, states, time_s):
        time, truth = _decision_time(time_s), _ordered_states(states)
        records = [_record(state) for state in truth.values()]
        fingerprint = _fingerprint(records)
        if self._time is not None and time == self._time:
            if fingerprint != self._input_fingerprint:
                raise ValueError("Physical input changed at the cached decision time")
            return self._cache
        expected = 0.0 if self._time is None else self._time + 5.0
        if time != expected:
            raise ValueError(f"Expected decision time {expected:g}; no skipped or reversed decisions")
        p, q, sigma = (self._cfg[name] for name in ("position_probability", "communication_probability", "position_sigma_m"))
        # Stage all mutable changes and RNG draws; failed geometry never consumes
        # a decision or advances this object's RNG/state.
        rng = self._rng if p + q == 0 else _copy_rng(self._rng)
        until, intervals = dict(self._until), copy.deepcopy(self._intervals)
        stats, sensed, events = dict(self._stats), dict(truth), []
        stats["decision_snapshots"] += 1
        stats["eligible_aircraft_decisions"] += len(truth)
        for acid, state in truth.items():
            category = 1.0 if p + q == 0 else float(rng.random())
            stats["categorical_draws"] += int(p + q > 0)
            if category < p:
                east, north = (0.0, 0.0) if sigma == 0 else (float(v) for v in rng.normal(0.0, sigma, size=2))
                sensed[acid] = self._shift(state, east, north)
                stats["position_triggers"] += 1
                stats["position_noise_vectors"] += int(sigma > 0)
                events.append({"acid": acid, "kind": "position", "time_s": time,
                               "east_m": east, "north_m": north})
            elif category < p + q:
                draw = float(rng.random())
                durations = self._cfg["communication_duration_seconds"]
                probabilities = self._cfg["communication_duration_probabilities"]
                duration = float(durations[0 if draw < probabilities[0] else 1 if draw < sum(probabilities[:2]) else 2])
                old = until.get(acid, 0.0)
                ending = max(old, time + duration)
                prior = intervals.setdefault(acid, [])
                retrigger = time < old
                new_segment = not prior or time > prior[-1][1]
                continued_at_expiry = bool(prior) and time == prior[-1][1]
                if new_segment:
                    prior.append([time, ending])
                else:
                    prior[-1][1] = max(prior[-1][1], ending)
                until[acid] = ending
                stats["communication_triggers"] += 1
                stats["communication_duration_draws"] += 1
                stats["communication_retriggers"] += int(retrigger)
                stats["blackout_segments_started"] += int(new_segment)
                stats["blackout_segments_continued_at_expiry"] += int(continued_at_expiry)
                events.append({"acid": acid, "kind": "communication", "time_s": time,
                               "duration_s": duration, "previous_until_s": old, "until_s": ending,
                               "retrigger": retrigger, "new_segment": new_segment})
            else:
                stats["none_selections"] += 1
        available = tuple(acid for acid in truth if _available(intervals, acid, time))
        stats["currently_unavailable"] = len(truth) - len(available)
        snapshot = _snapshot(time, sensed, available, events, until, stats)
        self._rng, self._until, self._intervals, self._stats = rng, until, intervals, stats
        self._time, self._cache = time, snapshot
        self._truth_records, self._input_fingerprint = records, fingerprint
        return snapshot

    def state_dict(self):
        cache = None if self._cache is None else {
            "time_s": self._time, "input_states": self._truth_records,
            "input_fingerprint": self._input_fingerprint,
            "sensed_states": [_record(state) for state in self._cache.sensed_states.values()],
            "available_ids": list(self._cache.available_ids), "events": [dict(event) for event in self._cache.events],
            "until": dict(self._cache.until), "statistics": dict(self._cache.statistics)}
        state = {"schema": "observation-perturbation-state.v1", "config": self._cfg,
                 "config_fingerprint": self._config_fingerprint, "numpy_version": np.__version__,
                 "seed": self._seed, "rng_state": self._rng.bit_generator.state,
                 "last_time_s": self._time, "until": self._until, "blackout_intervals": self._intervals,
                 "statistics": self._stats, "cache": cache}
        state = _primitive_copy(state)
        state["payload_sha256"] = _fingerprint(state)
        return state

    def load_state_dict(self, state):
        """Validate a detached primitive checkpoint, then replace state atomically."""
        try:
            decoded = self._decode_checkpoint(_primitive_copy(state))
        except (KeyError, TypeError, IndexError, OverflowError) as exc:
            raise ValueError("Malformed perturbation checkpoint") from exc
        (seed, rng, time, until, intervals, stats, snapshot, records, fingerprint) = decoded
        self._seed, self._rng, self._time = seed, rng, time
        self._until, self._intervals, self._stats = until, intervals, stats
        self._cache, self._truth_records, self._input_fingerprint = snapshot, records, fingerprint

    def _decode_checkpoint(self, state):
        required = {"schema", "config", "config_fingerprint", "numpy_version", "seed", "rng_state",
                    "last_time_s", "until", "blackout_intervals", "statistics", "cache", "payload_sha256"}
        if set(state) != required:
            raise ValueError("Checkpoint fields differ from the current schema")
        digest = state.pop("payload_sha256")
        if digest != _fingerprint(state):
            raise ValueError("Checkpoint payload identity mismatch")
        if (state["schema"] != "observation-perturbation-state.v1" or state["numpy_version"] != np.__version__
                or state["config_fingerprint"] != self._config_fingerprint
                or _canonical(state["config"]) != _canonical(self._cfg)):
            raise ValueError("Checkpoint schema, NumPy or configuration identity mismatch")
        seed = _seed(state["seed"])
        rng_data = state["rng_state"]
        if (set(rng_data) != {"bit_generator", "state", "has_uint32", "uinteger"}
                or rng_data["bit_generator"] != "PCG64" or set(rng_data["state"]) != {"state", "inc"}):
            raise ValueError("Checkpoint RNG is not the declared PCG64 state")
        for value, limit in ((rng_data["state"]["state"], 2**128), (rng_data["state"]["inc"], 2**128),
                             (rng_data["has_uint32"], 2), (rng_data["uinteger"], 2**32)):
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < limit:
                raise ValueError("Invalid primitive PCG64 field")
        if rng_data["state"]["inc"] % 2 != 1:
            raise ValueError("PCG64 stream increment must be odd")
        rng = np.random.Generator(np.random.PCG64(seed))
        rng.bit_generator.state = rng_data
        time = None if state["last_time_s"] is None else _decision_time(state["last_time_s"])
        stats, until, intervals = state["statistics"], state["until"], state["blackout_intervals"]
        if set(stats) != set(STAT_KEYS) or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in stats.values()):
            raise ValueError("Invalid perturbation statistics")
        if stats["decision_snapshots"] != (0 if time is None else int(time / 5) + 1):
            raise ValueError("Checkpoint clock does not match decision count")
        eligible, pos, com = stats["eligible_aircraft_decisions"], stats["position_triggers"], stats["communication_triggers"]
        p, q, sigma = (self._cfg[name] for name in ("position_probability", "communication_probability", "position_sigma_m"))
        if (pos + com + stats["none_selections"] != eligible
                or stats["categorical_draws"] != (eligible if p + q > 0 else 0)
                or stats["communication_duration_draws"] != com
                or stats["position_noise_vectors"] != (pos if sigma > 0 else 0)
                or (p == 0 and pos) or (q == 0 and com)
                or stats["communication_retriggers"] + stats["blackout_segments_started"]
                   + stats["blackout_segments_continued_at_expiry"] != com):
            raise ValueError("Inconsistent event/draw counters")
        if not isinstance(until, dict) or not isinstance(intervals, dict) or set(until) != set(intervals):
            raise ValueError("Blackout interval and expiry IDs differ")
        for acid, segments in intervals.items():
            if not isinstance(acid, str) or not acid or not isinstance(segments, list) or not segments or time is None:
                raise ValueError("Invalid blackout history")
            previous_end = -1.0
            for segment in segments:
                if not isinstance(segment, list) or len(segment) != 2:
                    raise ValueError("Blackout segment must contain start and end")
                start, end = (_decision_time(value) for value in segment)
                if not previous_end < start < end or start > time or end > time + 15.0:
                    raise ValueError("Blackout segments are not valid merged half-open intervals")
                previous_end = end
            if _number(until[acid], "until") != previous_end:
                raise ValueError("Expiry differs from the last registered interval")
        if stats["blackout_segments_started"] != sum(len(segments) for segments in intervals.values()):
            raise ValueError("Blackout segment count differs from interval history")
        cache = state["cache"]
        if time is None:
            if cache is not None or until or any(stats.values()) or rng_data != np.random.PCG64(seed).state:
                raise ValueError("A reset checkpoint must be empty with its initial RNG state")
            return seed, rng, time, until, intervals, stats, None, None, None
        if set(cache) != {"time_s", "input_states", "input_fingerprint", "sensed_states", "available_ids", "events", "until", "statistics"}:
            raise ValueError("Invalid cached snapshot fields")
        if cache["time_s"] != time or cache["until"] != until or cache["statistics"] != stats:
            raise ValueError("Cached time, expiries or statistics differ from checkpoint state")
        truth, sensed = _decode_records(cache["input_states"]), _decode_records(cache["sensed_states"])
        if set(truth) != set(sensed) or cache["input_fingerprint"] != _fingerprint(cache["input_states"]):
            raise ValueError("Cached aircraft/input identity mismatch")
        available = [acid for acid in truth if _available(intervals, acid, time)]
        if cache["available_ids"] != available or stats["currently_unavailable"] != len(truth) - len(available):
            raise ValueError("Cached availability differs from registered intervals")
        if eligible < len(truth):
            raise ValueError("Cached fleet exceeds the eligible decision count")
        expected_sensed, event_ids = dict(truth), set()
        for event in cache["events"]:
            acid, kind = event["acid"], event["kind"]
            if acid not in truth or acid in event_ids or event["time_s"] != time:
                raise ValueError("Cached event ID/time is inconsistent")
            event_ids.add(acid)
            if kind == "position":
                if p == 0 or set(event) != {"acid", "kind", "time_s", "east_m", "north_m"}:
                    raise ValueError("Invalid cached position event")
                east, north = _number(event["east_m"], "east_m"), _number(event["north_m"], "north_m")
                if sigma == 0 and (east != 0 or north != 0):
                    raise ValueError("Zero sigma checkpoint contains a displacement")
                expected_sensed[acid] = self._shift(truth[acid], east, north)
            elif kind == "communication":
                if q == 0 or set(event) != {"acid", "kind", "time_s", "duration_s", "previous_until_s", "until_s", "retrigger", "new_segment"}:
                    raise ValueError("Invalid cached communication event")
                duration = _number(event["duration_s"], "duration_s")
                old = _number(event["previous_until_s"], "previous_until_s")
                if (duration not in (5, 10, 15) or old < 0 or event["until_s"] != max(old, time + duration)
                        or event["until_s"] != until.get(acid) or type(event["retrigger"]) is not bool
                        or type(event["new_segment"]) is not bool or event["retrigger"] != (time < old)
                        or event["new_segment"] != (intervals[acid][-1][0] == time)):
                    raise ValueError("Cached communication expiry/retrigger is inconsistent")
            else:
                raise ValueError("Unsupported cached perturbation kind")
        if [_record(s) for s in expected_sensed.values()] != cache["sensed_states"]:
            raise ValueError("Cached sensed states do not match the declared position-only events")
        if p + q == 0 and rng_data != np.random.PCG64(seed).state:
            raise ValueError("Disabled perturbation checkpoint consumed its private RNG")
        snapshot = _snapshot(time, sensed, available, cache["events"], until, stats)
        return seed, rng, time, until, intervals, stats, snapshot, cache["input_states"], cache["input_fingerprint"]
