"""Geometric arrival diagnostics; no movement, native-passed or safety shortcut.

Coordinates are local east/north metres. An endpoint disk and a finite corridor
exit are distinct task definitions; callers must explicitly select one.
"""
from __future__ import annotations

import math


def _finite(values):
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Arrival geometry requires finite coordinates")


def segment_nearest_endpoint(before_xy, after_xy, endpoint_xy):
    """Minimum distance over the closed sampled motion chord, and its fraction."""
    _finite((*before_xy, *after_xy, *endpoint_xy))
    dx, dy = (after_xy[i] - before_xy[i] for i in range(2))
    length2 = dx*dx + dy*dy
    fraction = (0. if length2 == 0 else max(0., min(1.,
        ((endpoint_xy[0]-before_xy[0])*dx + (endpoint_xy[1]-before_xy[1])*dy)/length2)))
    distance = math.hypot(before_xy[0] + fraction*dx - endpoint_xy[0],
                          before_xy[1] + fraction*dy - endpoint_xy[1])
    return distance, fraction


def finite_exit_crossing(before_xyz, after_xyz, endpoint_xy, final_unit_xy,
                         half_width_m, floor_m, ceiling_m, *, width_tolerance_m=1e-7):
    """Report a forward crossing of the *finite* final corridor cross-section.

    The caller separately requires genuine final-leg mission progress, including
    for nearly closed/self-intersecting routes. Altitude and lateral position are
    interpolated at the crossing, not measured at the end of the physical tick.
    Width tolerance is an explicit terminal-label convention, not extra action
    space or a bound on tracking error. The default preserves historical callers.
    """
    _finite((*before_xyz, *after_xyz, *endpoint_xy, *final_unit_xy,
             half_width_m, floor_m, ceiling_m, width_tolerance_m))
    if isinstance(width_tolerance_m, bool) or width_tolerance_m < 0:
        raise ValueError("Exit width tolerance must be finite and nonnegative")
    if half_width_m <= 0 or floor_m >= ceiling_m:
        raise ValueError("Invalid finite exit dimensions")
    if not math.isclose(math.hypot(*final_unit_xy), 1., abs_tol=1e-10):
        raise ValueError("Final course must be a unit vector")
    ux, uy = final_unit_xy
    along_before = sum((before_xyz[i]-endpoint_xy[i])*final_unit_xy[i] for i in range(2))
    along_after = sum((after_xyz[i]-endpoint_xy[i])*final_unit_xy[i] for i in range(2))
    if not along_before < 0 <= along_after:
        return None
    fraction = -along_before/(along_after-along_before)
    point = tuple(a + fraction*(b-a) for a, b in zip(before_xyz, after_xyz))
    lateral = (point[0]-endpoint_xy[0])*uy - (point[1]-endpoint_xy[1])*ux
    # Height semantics are unchanged; only width has a configurable tolerance.
    height_tolerance = 1e-7
    return {"fraction": float(fraction), "cross_track_m": float(lateral),
            "altitude_m": float(point[2]),
            "within_width": bool(abs(lateral) <= half_width_m+width_tolerance_m),
            "within_height": bool(floor_m-height_tolerance <= point[2] <= ceiling_m+height_tolerance)}
