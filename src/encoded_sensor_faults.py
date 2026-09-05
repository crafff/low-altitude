"""Explicit scalar faults AFTER normalized own[7]/intruders[N,10] encoding.

Source facts (paper pp.10-11; REPRODUCTION 08:02): missing information is
represented at the policy input by a large finite value; abnormal information
lies outside the normalization interval; falsification shuffles at least two
elements. The paper does not specify executable value/selection distributions.

Choices here: fixed missing sentinel 2*Nmax=2; caller-supplied abnormal values;
explicit scalar permutations; no overlapping plans and all reads from the
original normalized input. No sampler, RNG, clipping, aircraft disappearance,
terminal handling or training integration is provided. An empty plan is closed.

Full intruder-row permutation preserves the current attention input set. The
shared_ppo attention is mathematically permutation invariant (subject to floating
reduction roundoff); row IDs are metadata, not model features. Such a permutation
is labelled in traces and is NOT evidence of harmful falsification. Even a scalar
permutation can exchange equal values and cause no numeric change.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict

import numpy as np

from paper_observation import INTRUDER_FEATURES, OWN_FEATURES


MISSING_SENTINEL = 2.0
FLOAT_DTYPES = (np.dtype("float16"), np.dtype("float32"), np.dtype("float64"))


class EncodedObservation(TypedDict):
    """Local typing for the dictionary returned by the existing observe API."""
    own: np.ndarray
    intruders: np.ndarray
    intruder_ids: tuple[str, ...]
    action_mask: NotRequired[np.ndarray]
    clipping_counts: NotRequired[dict[str, tuple[int, ...]]]


@dataclass(frozen=True)
class Slot:
    """One scalar: Slot('own', column) or Slot('intruders', column, row)."""
    block: Literal["own", "intruders"]
    column: int
    row: int | None = None


@dataclass(frozen=True)
class FaultPlan:
    """permutation[k] supplies the source slot index for destination slots[k]."""
    kind: Literal["missing", "abnormal", "falsified"]
    slots: tuple[Slot, ...]
    value: float | None = None
    permutation: tuple[int, ...] | None = None


@dataclass(frozen=True)
class EntryTrace:
    target: Slot
    target_intruder_id: str | None
    feature_name: str
    source: Slot | None
    source_intruder_id: str | None
    before: float
    after: float
    before_hex: str
    after_hex: str
    numeric_changed: bool
    bits_changed: bool


@dataclass(frozen=True)
class FaultEvent:
    plan_index: int
    kind: str
    applied: bool
    requested_value: float | None
    encoded_replacement: float | None
    permutation: tuple[int, ...] | None
    planned_entries: int
    numerically_changed_entries: int
    bitwise_changed_entries: int
    full_intruder_row_permutation: bool
    entries: tuple[EntryTrace, ...]


@dataclass(frozen=True)
class FaultCounts:
    plans: int
    planned_entries: int
    applied_entries: int
    numerically_changed_entries: int
    bitwise_changed_entries: int


@dataclass(frozen=True)
class FaultResult:
    observation: EncodedObservation
    enabled: bool
    feature_dtype: str
    events: tuple[FaultEvent, ...]
    counts: FaultCounts

    def trace_dict(self) -> dict[str, Any]:
        """Detached primitive JSON trace; includes exact IEEE bytes per scalar."""
        return json.loads(json.dumps({"schema": "encoded-sensor-fault-trace.v1", "enabled": self.enabled,
            "feature_dtype": self.feature_dtype, "counts": asdict(self.counts),
            "events": [asdict(event) for event in self.events]}, allow_nan=False))


def load_config(path: str | Path) -> dict[str, Any]:
    cfg = json.loads(Path(path).read_text())
    if (cfg["schema"] != "paper.encoded-sensor-faults.v1" or cfg["normalization_interval"] != [0.0, 1.0]
            or cfg["missing_sentinel"] != MISSING_SENTINEL or cfg["default_plans"] != []
            or cfg["fault_kinds"] != ["missing", "abnormal", "falsified"]
            or cfg["accepted_float_dtypes"] != ["float16", "float32", "float64"]):
        raise ValueError("Configuration differs from the explicit fixed sensor-fault contract")
    return cfg


def _integer(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{label} must be an integer, not a boolean or rounded index")
    return int(value)


def _validate_observation(observation: Mapping[str, Any]) -> None:
    if not isinstance(observation, Mapping) or not {"own", "intruders", "intruder_ids"} <= set(observation):
        raise ValueError("Expected an encoded own/intruders/intruder_ids mapping")
    own, intruders = observation["own"], observation["intruders"]
    if type(own) is not np.ndarray or type(intruders) is not np.ndarray:
        raise ValueError("Features must be NumPy arrays without implicit conversion")
    if own.shape != (7,) or intruders.ndim != 2 or intruders.shape[1] != 10:
        raise ValueError("Expected own[7] and intruders[N,10], including N=0")
    if own.dtype not in FLOAT_DTYPES or intruders.dtype != own.dtype:
        raise ValueError("Feature arrays must share a native float16/32/64 dtype")
    for array in (own, intruders):
        if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
            raise ValueError("Source features must be finite and already normalized to [0,1]")
    ids = observation["intruder_ids"]
    if (not isinstance(ids, tuple) or len(ids) != len(intruders)
            or any(not isinstance(acid, str) or not acid for acid in ids) or len(set(ids)) != len(ids)):
        raise ValueError("Each intruder row needs one distinct nonempty ID in an ordered tuple")
    if "action_mask" in observation:
        mask = observation["action_mask"]
        if type(mask) is not np.ndarray or mask.dtype != np.bool_ or mask.shape != (60,) or not mask.any():
            raise ValueError("A supplied action_mask must be bool[60] with a legal action")


def _slot(slot: Slot, count: int) -> Slot:
    if not isinstance(slot, Slot) or slot.block not in ("own", "intruders"):
        raise ValueError("Selected entries must be explicit own/intruders Slot objects")
    column = _integer(slot.column, "column")
    if slot.block == "own":
        if slot.row is not None or not 0 <= column < 7:
            raise ValueError("Ownship selection requires column 0..6 and row=None")
        return Slot("own", column)
    row = _integer(slot.row, "intruder row")
    if not 0 <= row < count or not 0 <= column < 10:
        raise ValueError("Intruder slot lies outside the existing N by 10 matrix")
    return Slot("intruders", column, row)


def _get(observation: Mapping[str, Any], slot: Slot) -> np.floating:
    return observation[slot.block][slot.column if slot.block == "own" else (slot.row, slot.column)]


def _put(observation: EncodedObservation, slot: Slot, value: np.floating) -> None:
    observation[slot.block][slot.column if slot.block == "own" else (slot.row, slot.column)] = value


def _abnormal(value: Any, dtype: np.dtype) -> tuple[float, np.floating]:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError("Abnormal plans require an explicit finite real numeric value")
    try:
        requested = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError("Abnormal value must be finite") from exc
    if not math.isfinite(requested) or 0 <= requested <= 1:
        raise ValueError("Abnormal value must lie genuinely outside [0,1]")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        encoded = dtype.type(requested)
    if not np.isfinite(encoded) or 0 <= encoded <= 1:
        raise ValueError("Abnormal value must remain finite and outside [0,1] in the feature dtype")
    return requested, encoded


def _whole_row_permutation(slots: tuple[Slot, ...], permutation: tuple[int, ...]) -> bool:
    if any(slot.block != "intruders" for slot in slots):
        return False
    rows = {slot.row for slot in slots}
    if len(slots) != 10 * len(rows):
        return False
    mapping = {}
    for target, source_index in zip(slots, permutation):
        source = slots[source_index]
        if target.column != source.column or (target.row in mapping and mapping[target.row] != source.row):
            return False
        mapping[target.row] = source.row
    return set(mapping.values()) == rows


def transform(observation: EncodedObservation, plans: Sequence[FaultPlan] = (), *, enabled: bool = True) -> FaultResult:
    """Validate all input/plans, then return independent copies and exact traces.

    Plans must select disjoint scalar slots. A falsification uses an explicit
    bijection with at least two moved slot indices, reading original values.
    Validation also runs when disabled. Existing corrupted outputs are not clean
    normalized inputs; put all intended disjoint faults in one call instead.
    Metadata/masks are deep-copied, never recomputed or added to the features.
    """
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    _validate_observation(observation)
    if not isinstance(plans, Sequence) or isinstance(plans, (str, bytes)):
        raise ValueError("plans must be an explicit sequence of FaultPlan objects")
    prepared, occupied = [], set()
    dtype = observation["own"].dtype
    for plan in plans:
        if (not isinstance(plan, FaultPlan) or plan.kind not in ("missing", "abnormal", "falsified")
                or not isinstance(plan.slots, tuple) or not plan.slots):
            raise ValueError("Each plan needs a supported kind and a nonempty immutable slots tuple")
        slots = tuple(_slot(slot, len(observation["intruders"])) for slot in plan.slots)
        if len(set(slots)) != len(slots) or occupied.intersection(slots):
            raise ValueError("Duplicate/overlapping scalar selections are ambiguous and rejected")
        requested = replacement = permutation = None
        if plan.kind == "missing":
            if plan.value is not None or plan.permutation is not None:
                raise ValueError("Missing uses fixed 2.0; do not supply values or a permutation")
            replacement = dtype.type(MISSING_SENTINEL)
        elif plan.kind == "abnormal":
            if plan.permutation is not None:
                raise ValueError("Abnormal replacement does not accept a permutation")
            requested, replacement = _abnormal(plan.value, dtype)
        else:
            if plan.value is not None or not isinstance(plan.permutation, tuple) or len(slots) < 2:
                raise ValueError("Falsified requires at least two slots and an explicit permutation tuple")
            permutation = tuple(_integer(index, "permutation index") for index in plan.permutation)
            if sorted(permutation) != list(range(len(slots))) or permutation == tuple(range(len(slots))):
                raise ValueError("Permutation must be a nonidentity bijection of the selected scalar slots")
        occupied.update(slots)
        prepared.append((plan.kind, slots, requested, replacement, permutation))

    # No source or output writes occur until every plan passes validation.
    result: EncodedObservation = copy.deepcopy(dict(observation))
    events = []
    for index, (kind, slots, requested, replacement, permutation) in enumerate(prepared):
        entries = []
        for position, target in enumerate(slots):
            source = slots[permutation[position]] if permutation is not None else None
            before = _get(observation, target)
            after = (_get(observation, source) if source is not None else replacement) if enabled else before
            _put(result, target, after)
            before_hex, after_hex = before.tobytes().hex(), after.tobytes().hex()
            entries.append(EntryTrace(target, observation["intruder_ids"][target.row] if target.row is not None else None,
                (OWN_FEATURES if target.block == "own" else INTRUDER_FEATURES)[target.column], source,
                observation["intruder_ids"][source.row] if source is not None and source.row is not None else None,
                float(before), float(after), before_hex, after_hex, bool(before != after), before_hex != after_hex))
        events.append(FaultEvent(index, kind, enabled, requested, float(replacement) if replacement is not None else None,
            permutation, len(slots), sum(entry.numeric_changed for entry in entries), sum(entry.bits_changed for entry in entries),
            permutation is not None and _whole_row_permutation(slots, permutation), tuple(entries)))
    count = len(occupied)
    counts = FaultCounts(len(events), count, count if enabled else 0,
                        sum(event.numerically_changed_entries for event in events),
                        sum(event.bitwise_changed_entries for event in events))
    return FaultResult(result, enabled, dtype.str, tuple(events), counts)
