"""Explicit ordinary-flyby current-state reconstruction for native guidance.

The formula follows the audited ordinary-flyby branch in author fork
849d76fd44880f8d17a69aefa0bd37208f2b2fbb. It does not identify the author's
2026 setup or guarantee corridor containment. No diagnostic module is imported.
"""
from __future__ import annotations

from contextlib import contextmanager
import math
import random

import numpy as np


GUIDANCE_MODES = ("native_cached", "current_state_refresh")


def validate_guidance(mode):
    if mode not in GUIDANCE_MODES:
        raise ValueError("ordinary_flyby_guidance must be native_cached or current_state_refresh")
    return mode


def navigation_audit(mode):
    """New episode-local counters; the default mode never installs a wrapper."""
    validate_guidance(mode)
    return {"mode": mode, "enabled": mode == "current_state_refresh",
            "context_entries": 0, "context_exits": 0, "context_active": False,
            "wrapper_calls": 0, "native_calls": 0, "empty_population_calls": 0,
            "flyby_entries_refreshed": 0, "unsupported_calls": 0,
            "max_abs_cache_change_m": 0., "rng_checks": 0, "adapter_rng_unchanged": True,
            "rng_audit_scope": "Python/NumPy global streams across refresh computation, excluding native reached",
            "facade_identity_preserved": True, "wrapper_identity_preserved": True,
            "method_restored": True}


def refreshed_flyby_distance(tas_mps, bank_rad, qdr_deg, next_qdr_deg, gravity_mps2):
    """Current TAS radius times tan(half wrapped turn angle), without a cap.

    A next-bearing sentinel below -900 means continue on the current bearing.
    Cached turnspd and selected target speed deliberately do not enter this
    ordinary-flyby formula. Native reached retains all waypoint-pass decisions.
    """
    tas, bank, qdr, next_qdr = np.broadcast_arrays(*(
        np.asarray(value, dtype=float) for value in (tas_mps, bank_rad, qdr_deg, next_qdr_deg)))
    if not all(np.isfinite(value).all() for value in (tas, bank, qdr, next_qdr)) or np.any(tas < 0):
        raise ValueError("Flyby inputs must be finite with nonnegative TAS")
    if not math.isfinite(gravity_mps2) or gravity_mps2 <= 0:
        raise ValueError("Gravity must be finite and positive")
    following = np.where(next_qdr < -900., qdr, next_qdr)
    difference = (qdr % 360. - following % 360. + 180.) % 360. - 180.
    radius = tas * tas / (gravity_mps2 * np.maximum(.01, np.tan(bank)))
    distance = np.abs(radius * np.tan(np.radians(.5 * np.abs(difference))))
    if not np.isfinite(distance).all():
        raise ValueError("Flyby refresh produced a nonfinite distance")
    return distance


def _rng_state():
    return random.getstate(), np.random.get_state()


def _same_rng(first, second):
    a, b = first[1], second[1]
    return first[0] == second[0] and a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]


@contextmanager
def ordinary_flyby_guidance(bs, mode, audit, *, gravity_mps2):
    """Install at the actual Proxy facade for one environment step, then restore.

    Current arrays are fetched on every native call, including empty traffic and
    admissions/deletions. The sole pre-native traffic mutation is turndist at
    ordinary flyby entries. Unsupported flyturn modes fail before that mutation.
    """
    validate_guidance(mode)
    if mode == "native_cached":
        yield audit
        return
    if audit["mode"] != mode or audit["context_active"]:
        raise RuntimeError("Guidance audit mode differs or refresh context is already active")
    facade = bs.traf.actwp
    namespace = vars(facade)
    had_member, old_member = "reached" in namespace, namespace.get("reached")
    original = facade.reached
    audit["context_entries"] += 1
    audit["context_active"] = True
    audit["native_method"] = f"{original.__module__}.{original.__qualname__}"
    audit["facade_class"] = f"{type(facade).__module__}.{type(facade).__qualname__}"

    def wrapped(qdr, dist):
        audit["wrapper_calls"] += 1
        if bs.traf.actwp is not facade:
            audit["facade_identity_preserved"] = False
            raise RuntimeError("ActiveWaypoint facade changed during a guidance step")
        rng_before = _rng_state()
        count = len(bs.traf.id)
        unsupported = [np.asarray(getattr(facade, name), dtype=bool)
                       for name in ("flyturn", "turnfromlastwp", "turntonextwp")]
        qdr_array, dist_array = np.asarray(qdr, dtype=float), np.asarray(dist, dtype=float)
        tas, bank = np.asarray(bs.traf.tas, dtype=float), np.asarray(bs.traf.ap.bankdef, dtype=float)
        next_qdr = np.asarray(facade.next_qdr, dtype=float)
        before = np.array(facade.turndist, dtype=float, copy=True)
        flyby = np.asarray(facade.flyby, dtype=bool)
        arrays = (qdr_array, dist_array, tas, bank, next_qdr, before, flyby, *unsupported)
        if any(array.shape != (count,) for array in arrays):
            raise RuntimeError("Native flyby arrays differ from current traffic IDs")
        if any(array.any() for array in unsupported):
            audit["unsupported_calls"] += 1
            raise RuntimeError("Flyturn and flyturn-transition modes are unsupported by current_state_refresh")
        refreshed = refreshed_flyby_distance(tas, bank, qdr_array, next_qdr, gravity_mps2)
        facade.turndist[flyby] = refreshed[flyby]
        audit["flyby_entries_refreshed"] += int(flyby.sum())
        audit["empty_population_calls"] += int(count == 0)
        if flyby.any():
            audit["max_abs_cache_change_m"] = max(audit["max_abs_cache_change_m"],
                                                  float(np.max(np.abs(refreshed[flyby]-before[flyby]))))
        audit["rng_checks"] += 1
        audit["adapter_rng_unchanged"] &= _same_rng(rng_before, _rng_state())
        if not audit["adapter_rng_unchanged"]:
            raise RuntimeError("Guidance adapter changed a global random stream")
        audit["native_calls"] += 1
        return original(qdr, dist)

    namespace["reached"] = wrapped
    try:
        yield audit
    finally:
        audit["facade_identity_preserved"] &= bs.traf.actwp is facade
        audit["wrapper_identity_preserved"] &= namespace.get("reached") is wrapped
        if had_member:
            namespace["reached"] = old_member
        else:
            namespace.pop("reached", None)
        audit["method_restored"] &= facade.reached == original
        audit["context_active"] = False
        audit["context_exits"] += 1
