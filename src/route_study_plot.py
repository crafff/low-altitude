"""Plot the seven saved route-execution diagnostics, without running BlueSky.

Input JSON schema (all seven cases and every 0.25 s sample are required)::

    {
      "schema": "route-study-plot.v1",
      "source_run": "runs/20260905T062257Z-route-action-seven-725d5871",
      "route_xy": [[0, 0], [4630, 0], [4630, 4630]],
      "half_width_m": 76.2,
      "bank_limit_deg": 25.0,
      "sample_interval_s": 0.25,
      "cases": [{
        "id": "mavic_center", "type": "Mavic", "status": "arrived",
        "lane": "center", "command_t": 0.0,
        "points": [{"t": 0.0, "x_m": 0.0, "y_m": 0.0, "deviation_m": 0.0}]
      }]
    }

The example shows only one case/sample to document the fields. Valid case IDs,
types and lane tags are listed in CASES below. For the equatorial source route,
x_m = lon_deg*pi*6371000/180 and y_m = lat_deg*pi*6371000/180. deviation_m is
the saved physics.csv centerline_distance_m; do not substitute lane error.
command_t is the triggering decision time in seconds, or null if never issued.

Controller invocation inside lab: python -B -m route_study_plot --input FILE.
Only route_execution.svg/png/pdf and plot_metadata.json are explicitly written,
all below the launcher's LAB_RUN_DIR. Matplotlib uses its noninteractive backend.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path


CASES = {
    "mavic_center": ("Mavic", "center"),
    "mavic_inner": ("Mavic", "inner"),
    "mavic_outer": ("Mavic", "outer"),
    "amzn_center": ("Amzn", "center"),
    "amzn_inner": ("Amzn", "inner"),
    "amzn_outer": ("Amzn", "outer"),
    "amzn_late_after_native_advance": ("Amzn", "late_inner"),
}
STYLES = {
    "center": {"color": "#2463A8", "label": "Center (37)", "linestyle": "-"},
    "inner": {"color": "#D65D20", "label": "Inner (36)", "linestyle": "-"},
    "outer": {"color": "#7950A4", "label": "Outer (38)", "linestyle": "-"},
    "late_inner": {"color": "#138678", "label": "Late inner (36)", "linestyle": (0, (5, 3))},
}
SPATIAL_WINDOWS = {
    "Mavic": {"start": (-30.0, 330.0, -110.0, 110.0),
              "corner": (-220.0, 160.0, -130.0, 250.0)},
    "Amzn": {"start": (-30.0, 330.0, -110.0, 110.0),
             "corner": (-1800.0, 300.0, -300.0, 1800.0)},
}


def _number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


def load_input(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema") != "route-study-plot.v1":
        raise ValueError("Expected route-study-plot.v1 input")
    if not isinstance(data.get("source_run"), str) or not data["source_run"].strip():
        raise ValueError("source_run must identify the saved source run")
    route = data["route_xy"]
    expected = ((0.0, 0.0), (4630.0, 0.0), (4630.0, 4630.0))
    if len(route) != 3 or any(len(point) != 2 for point in route):
        raise ValueError("route_xy must contain the three nominal vertices")
    if any(not math.isclose(_number(v, "route_xy"), e, abs_tol=1e-6, rel_tol=0)
           for point, reference in zip(route, expected) for v, e in zip(point, reference)):
        raise ValueError("This figure is scoped to the 4630 m east / 4630 m north diagnostic")
    width = _number(data["half_width_m"], "half_width_m")
    bank = _number(data["bank_limit_deg"], "bank_limit_deg")
    dt = _number(data["sample_interval_s"], "sample_interval_s")
    if not 0 < width < 4630 or not 0 < bank < 90 or not math.isclose(dt, .25, abs_tol=1e-12):
        raise ValueError("Expected positive tube/bank and all 0.25 s physics samples")
    cases = data["cases"]
    if len(cases) != len(CASES) or {case["id"] for case in cases} != set(CASES):
        raise ValueError("Input must contain each of the seven predeclared cases once")
    by_id = {}
    for case in cases:
        name = case["id"]
        if (case["type"], case["lane"]) != CASES[name]:
            raise ValueError(f"Case type/lane does not match its ID: {name}")
        if case["status"] not in ("arrived", "flight_timeout", "error"):
            raise ValueError(f"Unsupported terminal status: {name}")
        points = case["points"]
        if len(points) < 2:
            raise ValueError(f"Need at least two physical samples: {name}")
        previous_t = None
        for point in points:
            for key in ("t", "x_m", "y_m", "deviation_m"):
                _number(point[key], f"{name}.{key}")
            if point["t"] < 0 or point["deviation_m"] < 0:
                raise ValueError(f"Time and distance must be nonnegative: {name}")
            if previous_t is None:
                if not math.isclose(point["t"], 0.0, abs_tol=1e-8):
                    raise ValueError(f"Initial t=0 sample is missing: {name}")
            elif not math.isclose(point["t"] - previous_t, dt, abs_tol=1e-8, rel_tol=0):
                raise ValueError(f"Missing, reordered or decimated physics samples: {name}")
            previous_t = point["t"]
        command = case["command_t"]
        if command is not None:
            command = _number(command, f"{name}.command_t")
            if not 0 <= command <= points[-1]["t"]:
                raise ValueError(f"Command time is outside the recorded trajectory: {name}")
            if not math.isclose(command / 5.0, round(command / 5.0), abs_tol=1e-8):
                raise ValueError(f"Command did not occur at a 5 s decision boundary: {name}")
        by_id[name] = case
    data["cases"] = [by_id[name] for name in CASES]
    return data, hashlib.sha256(raw).hexdigest()


def finite_tube_polygon(route, half_width, arc_segments=96):
    """Boundary of the finite L-polyline's distance <= radius set.

    The concave inner corner is the intersection of the two offset segments;
    the outer corner and both finite endpoints are round. This constructs one
    polygon, avoiding dark overlaps from separately painted segment capsules.
    Circular arcs use <= pi/96 radians per chord (sagitta <0.011 m at 76.2 m).
    """
    (x0, y0), (xc, yc), (xe, ye) = route
    radius = half_width
    if y0 != yc or xc != xe or not x0 < xc or not yc < ye:
        raise ValueError("Tube polygon requires an east-then-north finite route")

    def arc(x, y, start, stop):
        return [(x + radius * math.cos(start + (stop - start) * j / arc_segments),
                 y + radius * math.sin(start + (stop - start) * j / arc_segments))
                for j in range(arc_segments + 1)]

    boundary = [(x0, y0 - radius), (xc, yc - radius)]
    boundary.extend(arc(xc, yc, -math.pi / 2, 0)[1:])
    boundary.append((xe + radius, ye))
    boundary.extend(arc(xe, ye, 0, math.pi)[1:])
    boundary.extend(((xc - radius, yc + radius), (x0, y0 + radius)))
    boundary.extend(arc(x0, y0, math.pi / 2, 3 * math.pi / 2)[1:])
    return boundary


def _peak(case, predicate=None):
    points = case["points"] if predicate is None else [p for p in case["points"] if predicate(p)]
    return max(points, key=lambda p: p["deviation_m"]) if points else None


def _near_command(case):
    if case["command_t"] is None:
        return None
    return min(case["points"], key=lambda p: abs(p["t"] - case["command_t"]))


def _label_peak(ax, case, point, origin, position, half_width, *, window=False, align="left"):
    color = STYLES[case["lane"]]["color"]
    xy = (point["x_m"] - origin[0], point["y_m"] - origin[1])
    amount = point["deviation_m"]
    suffix = f"; {amount - half_width:.1f} m outside" if amount > half_width else ""
    value = f"{amount:.1f} m{suffix}" if not window else f"Window max {amount:.1f} m{suffix}"
    ax.scatter(*xy, s=30, facecolor="white", edgecolor=color, linewidth=1.4, zorder=7)
    ax.annotate(f"{case['id']}\n{value}", xy=xy, xycoords="data",
                xytext=position, textcoords="axes fraction", ha=align, va="top",
                fontsize=9, color=color,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "edgecolor": "none", "alpha": .94},
                arrowprops={"arrowstyle": "-", "color": color, "lw": .9}, zorder=8)


def _spatial_panel(ax, cases, data, kind, letter):
    from matplotlib.patches import Polygon
    from matplotlib.ticker import MaxNLocator

    aircraft_type = cases[0]["type"]
    origin = data["route_xy"][0 if kind == "start" else 1]
    bounds = SPATIAL_WINDOWS[aircraft_type][kind]
    shifted = lambda p: (p[0] - origin[0], p[1] - origin[1])
    tube = [shifted(p) for p in finite_tube_polygon(data["route_xy"], data["half_width_m"])]
    ax.add_patch(Polygon(tube, closed=True, facecolor="#E9EDF1", edgecolor="#929BA4", lw=.85, zorder=0))
    route = [shifted(p) for p in data["route_xy"]]
    ax.plot([p[0] for p in route], [p[1] for p in route], color="#56616B", lw=1,
            linestyle=(0, (3, 3)), zorder=1)
    visible_peaks = []
    for case in cases:
        style = STYLES[case["lane"]]
        ax.plot([p["x_m"] - origin[0] for p in case["points"]],
                [p["y_m"] - origin[1] for p in case["points"]],
                color=style["color"], linestyle=style["linestyle"],
                lw=2.05 if case["lane"] == "late_inner" else 1.8, zorder=4)
        point = _peak(case, lambda p: bounds[0] <= p["x_m"] - origin[0] <= bounds[1]
                      and bounds[2] <= p["y_m"] - origin[1] <= bounds[3])
        if point is not None:
            visible_peaks.append((case, point))
    ax.set(xlim=bounds[:2], ylim=bounds[2:],
           xlabel=f"East of {'start' if kind == 'start' else 'corner'} (m)",
           ylabel=f"North of {'start' if kind == 'start' else 'corner'} (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    title = "Initial lane change" if kind == "start" else "Left-turn corner"
    ax.set_title(f"({letter}) {aircraft_type} — {title}", loc="left", fontsize=11.5, pad=9)
    if visible_peaks:
        worst, point = max(visible_peaks, key=lambda item: item[1]["deviation_m"])
        _label_peak(ax, worst, point, origin, (.03, .97), data["half_width_m"], window=kind == "start")
        if kind == "corner":
            center = next(c for c in cases if c["lane"] == "center")
            center_point = _peak(center)
            x, y = center_point["x_m"] - origin[0], center_point["y_m"] - origin[1]
            if worst["id"] != center["id"] and bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]:
                _label_peak(ax, center, center_point, origin, (.97, .70), data["half_width_m"], align="right")
    if kind == "corner":
        for case in cases:
            if case["lane"] != "late_inner":
                continue
            command = _near_command(case)
            if command is None:
                ax.text(.03, .30, "Late command was not issued", transform=ax.transAxes,
                        color=STYLES["late_inner"]["color"], fontsize=9)
                continue
            x, y = command["x_m"] - origin[0], command["y_m"] - origin[1]
            if bounds[0] <= x <= bounds[1] and bounds[2] <= y <= bounds[3]:
                color = STYLES["late_inner"]["color"]
                ax.scatter(x, y, marker="D", s=36, facecolor="white", edgecolor=color, lw=1.4, zorder=8)
                ax.annotate(f"Late inner command\nt = {case['command_t']:g} s", xy=(x, y),
                            xytext=(.03, .34), textcoords="axes fraction", ha="left", va="top",
                            fontsize=9, color=color,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": .93},
                            arrowprops={"arrowstyle": "-", "color": color, "lw": .9}, zorder=8)
    return {"aircraft_type": aircraft_type, "kind": kind, "origin_xy_m": origin,
            "limits_relative_to_origin_m": bounds,
            "visible_peak_by_case": {case["id"]: point for case, point in visible_peaks}}


def _time_panel(ax, cases, data, letter):
    from matplotlib.ticker import MaxNLocator

    width = data["half_width_m"]
    max_t = max(case["points"][-1]["t"] for case in cases)
    maximum = max(_peak(case)["deviation_m"] for case in cases)
    ax.axhspan(0, width, color="#E9EDF1", zorder=0)
    ax.axhline(width, color="#66727C", lw=1, linestyle=(0, (3, 3)), zorder=1)
    for case in cases:
        style = STYLES[case["lane"]]
        ax.plot([p["t"] for p in case["points"]], [p["deviation_m"] for p in case["points"]],
                color=style["color"], linestyle=style["linestyle"], lw=1.9, zorder=4)
    worst = max(cases, key=lambda case: _peak(case)["deviation_m"])
    peak = _peak(worst)
    color = STYLES[worst["lane"]]["color"]
    ax.scatter(peak["t"], peak["deviation_m"], s=30, facecolor="white", edgecolor=color, lw=1.4, zorder=7)
    ax.annotate(f"{worst['id']}\nPeak {peak['deviation_m']:.1f} m at {peak['t']:g} s",
                xy=(peak["t"], peak["deviation_m"]), xytext=(.96, .97), textcoords="axes fraction",
                ha="right", va="top", fontsize=9, color=color,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": .94},
                arrowprops={"arrowstyle": "-", "color": color, "lw": .9}, zorder=8)
    for case in cases:
        if case["lane"] == "late_inner" and case["command_t"] is not None:
            color = STYLES["late_inner"]["color"]
            ax.axvline(case["command_t"], color=color, lw=.9, linestyle=(0, (2, 3)), alpha=.8, zorder=2)
            point = _near_command(case)
            ax.scatter(point["t"], point["deviation_m"], marker="D", s=32,
                       facecolor="white", edgecolor=color, lw=1.3, zorder=7)
            ax.text(case["command_t"] + max_t * .015, maximum * .18,
                    f"Late command\n{case['command_t']:g} s", color=color, fontsize=9,
                    bbox={"facecolor": "white", "edgecolor": "none", "alpha": .9}, zorder=8)
    ax.text(.02, width / (max(width, maximum) * 1.31) + .025, f"Tube half-width: {width:g} m",
            transform=ax.transAxes, fontsize=9, color="#56616B")
    ax.set(xlim=(0, max_t * 1.02), ylim=(0, max(width, maximum) * 1.31),
           xlabel="Simulation time (s)", ylabel="Distance to nominal finite route (m)")
    ax.xaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.set_title(f"({letter}) {cases[0]['type']} — Deviation over the full flight", loc="left", fontsize=11.5, pad=9)


def plot(data, output, *, input_sha256, input_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    files = [output / f"route_execution.{suffix}" for suffix in ("svg", "png", "pdf")]
    metadata_path = output / "plot_metadata.json"
    if any(path.exists() for path in (*files, metadata_path)):
        raise ValueError("Plot outputs already exist in LAB_RUN_DIR")
    statistics = {}
    for case in data["cases"]:
        points, width = case["points"], data["half_width_m"]
        peak = _peak(case)
        statistics[case["id"]] = {"type": case["type"], "lane": case["lane"], "status": case["status"],
            "command_t_s": case["command_t"], "input_sample_count": len(points),
            "last_sample_time_s": points[-1]["t"], "global_peak": peak,
            "exceeded_tube": peak["deviation_m"] > width,
            "post_sample_outside_seconds": math.fsum((b["t"] - a["t"]) * (b["deviation_m"] > width)
                for a, b in zip(points, points[1:]))}
    arrivals = sum(case["status"] == "arrived" for case in data["cases"])
    excursions = sum(record["exceeded_tube"] for record in statistics.values())
    settings = {"font.family": "DejaVu Sans", "font.size": 10.0,
                "axes.labelsize": 10.0, "xtick.labelsize": 9.0, "ytick.labelsize": 9.0,
                "axes.spines.top": False, "axes.spines.right": False,
                "axes.edgecolor": "#6C7782", "axes.linewidth": .7,
                "svg.fonttype": "none", "pdf.fonttype": 42,
                "text.usetex": False, "path.simplify": False, "savefig.facecolor": "white"}
    with plt.rc_context(settings):
        fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.0))
        fig.subplots_adjust(left=.065, right=.982, top=.817, bottom=.145, wspace=.31, hspace=.42)
        fig.suptitle("Native route execution: lane capture and turn tracking", y=.976, fontsize=19, fontweight="semibold")
        fig.text(.5, .936, f"Raw native guidance | Bank limit {data['bank_limit_deg']:g}° | "
                 f"{arrivals}/7 arrived | {excursions}/7 exceeded the {data['half_width_m']:g} m tube", ha="center", fontsize=12)
        legend = [Line2D([0], [0], color=style["color"], linestyle=style["linestyle"], lw=2,
                         label=style["label"]) for style in STYLES.values()]
        legend.extend((Line2D([0], [0], color="#56616B", lw=1, linestyle=(0, (3, 3)), label="Nominal route"),
                       Patch(facecolor="#E9EDF1", edgecolor="#929BA4", label="Finite corridor tube")))
        fig.legend(handles=legend, loc="upper center", bbox_to_anchor=(.5, .902), ncol=6,
                   frameon=False, fontsize=10, handlelength=2.7, columnspacing=1.7)
        panels = []
        for row, kind in enumerate(("Mavic", "Amzn")):
            cases = [case for case in data["cases"] if case["type"] == kind]
            panels.append(_spatial_panel(axes[row, 0], cases, data, "start", "ad"[row]))
            panels.append(_spatial_panel(axes[row, 1], cases, data, "corner", "be"[row]))
            _time_panel(axes[row, 2], cases, data, "cf"[row])
            for ax in axes[row]:
                ax.grid(color="#BBC3CB", alpha=.25, linewidth=.5, zorder=-1)
                ax.set_axisbelow(True)
        fig.text(.065, .095, "Development diagnostics; no containment guarantee. Spatial axes use equal metres; corner zoom scales differ. "
                 "Late inner: first 5 s decision after native waypoint advance.", fontsize=9, color="#37434D")
        fig.text(.065, .067, "Source: " + data["source_run"], fontsize=9, color="#58636D")
        fig.text(.065, .039, f"All {data['sample_interval_s']:g} s physics samples are retained. "
                 "Tube boundaries include the finite endpoints, a round outer corner and the correct concave inner join.",
                 fontsize=9, color="#58636D")
        for path in files:
            fig.savefig(path, dpi=300, metadata={"Creator": "route_study_plot.py"})
        plt.close(fig)
    metadata = {"schema": "route-study-plot-metadata.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_run": data["source_run"], "input_name": input_name, "input_sha256": input_sha256,
                "matplotlib_version": matplotlib.__version__, "outputs": [path.name for path in files],
                "figure_size_inches": [17.5, 10.0], "png_dpi": 300,
                "route_xy": data["route_xy"], "half_width_m": data["half_width_m"],
                "bank_limit_deg": data["bank_limit_deg"], "sample_interval_s": data["sample_interval_s"],
                "all_samples_retained": True, "line_simplification": False,
                "case_statistics": statistics, "spatial_panels": panels,
                "style_by_lane": STYLES,
                "tube_geometry": "Single finite L-polyline distance-tube polygon; round outer corner and endpoints, concave inner offset intersection.",
                "tube_arc_segments": 96,
                "maximum_arc_chord_sagitta_m": data["half_width_m"] * (1 - math.cos(math.pi / 192)),
                "interpretation_limits": [
                    "Only constructed single-aircraft development trajectories are shown; these are not conflict-resolution or policy-effectiveness results.",
                    "Spatial panels show declared zoom windows; the time panels include complete supplied trajectories and global peaks.",
                    "deviation_m is the saved diagnostic distance to the full nominal polyline, not accepted-lane error.",
                    "Bank limit is supplied source metadata, not measured roll; trajectory samples do not alone establish the mechanism of a guidance failure.",
                    "The plotted tube is defined in the supplied local metre plane; source deviation used its documented local equirectangular calculation."]}
    with metadata_path.open("x") as handle:
        json.dump(metadata, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Controller-prepared route-study-plot.v1 JSON")
    args = parser.parse_args()
    data, digest = load_input(args.input)
    output = Path(os.environ["LAB_RUN_DIR"])
    if not output.is_absolute() or not output.is_dir():
        raise ValueError("LAB_RUN_DIR must be the launcher's existing absolute artifact directory")
    result = plot(data, output, input_sha256=digest, input_name=str(args.input))
    print(json.dumps({"outputs": result["outputs"] + ["plot_metadata.json"],
                      "source_run": result["source_run"], "input_sha256": digest}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
