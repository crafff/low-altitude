"""Export an unsmoothed six-panel PPO development diagnostic via tools/lab.py.

The controller assembles execution-learning-curve.v1 inputs from completed run
records and resolves duplicate checkpoints before invoking this renderer.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile


INPUT_SCHEMA = "execution-learning-curve.v1"
SCOPE = "execution-semantics learning diagnostic; corridor containment unresolved"
TITLE = "Execution-semantics PPO development diagnostic"
OUTPUT_NAMES = ["learning_curve.svg", "learning_curve.png", "learning_curve.pdf"]
FIGURE_SIZE_INCHES = (14., 8.7)
PNG_DPI = 300


def _reject_constant(value):
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(value, path="input"):
    # JSON's numeric overflow (e.g. 1e999) is not handled by parse_constant.
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{path} contains a non-finite number")
    if isinstance(value, dict):
        for key, child in value.items():
            _reject_nonfinite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_nonfinite(child, f"{path}[{index}]")


def _count(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, int) or value < int(positive):
        raise ValueError(f"{name} must be a {'positive' if positive else 'nonnegative'} integer")
    return value


def _number(value, name, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'nonnegative'}")
    return float(value)


def _equal_rate(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{name} disagrees with its numerator and denominator")


def _aggregate_metrics(aggregate, name):
    if not isinstance(aggregate, dict):
        raise ValueError(f"{name} must be a paper_train aggregate object")
    planned = _count(aggregate["planned"], f"{name}.planned", positive=True)
    completed = _count(aggregate["completed"], f"{name}.completed")
    timeout = _count(aggregate["failed_timeout"], f"{name}.failed_timeout")
    exhausted = _count(aggregate["failed_route_exhausted"], f"{name}.failed_route_exhausted")
    outside = _count(aggregate["outside_corridor_flights"], f"{name}.outside_corridor_flights")
    if completed + timeout + exhausted != planned:
        raise ValueError(f"{name}: completed + timeout + route-exhausted must equal planned")
    if outside > planned:
        raise ValueError(f"{name}: outside-corridor flight count exceeds planned population")
    hours = _number(aggregate["flight_hours"], f"{name}.flight_hours", positive=True)
    if "completed_fraction" in aggregate:
        fraction = _number(aggregate["completed_fraction"], f"{name}.completed_fraction")
        _equal_rate(fraction, completed/planned, f"{name}.completed_fraction")
    metrics = {"completed_percent": 100*completed/planned,
               "outside_corridor_percent": 100*outside/planned,
               "timeout_percent": 100*timeout/planned,
               "route_exhausted_percent": 100*exhausted/planned,
               "mean_flight_seconds": hours*3600/planned}
    pair_seconds = {}
    for level in ("nmac", "lowc"):
        risk = aggregate["risk"][level]
        prefix = f"{name}.risk.{level}"
        exposure = _number(risk["unordered_pair_seconds"], f"{prefix}.unordered_pair_seconds")
        rate = _number(risk["unordered_seconds_per_flight_hour"],
                       f"{prefix}.unordered_seconds_per_flight_hour")
        _equal_rate(rate, exposure/hours, f"{prefix}.unordered_seconds_per_flight_hour")
        if "directed_pair_seconds" in risk:
            directed = _number(risk["directed_pair_seconds"], f"{prefix}.directed_pair_seconds")
            _equal_rate(directed, 2*exposure, f"{prefix}.directed_pair_seconds")
        pair_seconds[level] = exposure
        metrics[f"{level}_seconds_per_flight_hour"] = exposure/hours
    if pair_seconds["nmac"] > pair_seconds["lowc"] and not math.isclose(
            pair_seconds["nmac"], pair_seconds["lowc"], rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"{name}: NMAC exposure cannot exceed LoWC exposure")
    if not all(math.isfinite(value) for value in metrics.values()):
        raise ValueError(f"{name}: derived metrics must be finite")
    return planned, metrics


def load_input(path):
    """Validate provenance and paired aggregate denominators without sorting rows."""
    raw = Path(path).read_bytes()
    document = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    _reject_nonfinite(document)
    if not isinstance(document, dict) or document.get("schema") != INPUT_SCHEMA:
        raise ValueError(f"input schema must be {INPUT_SCHEMA!r}")
    if document.get("scope") != SCOPE:
        raise ValueError("input scope must match the execution-semantics development diagnostic")
    figure_title = document.get("title", TITLE)
    if not isinstance(figure_title, str) or not figure_title.strip() or len(figure_title) > 120 or "\n" in figure_title:
        raise ValueError("title must be a nonempty single line of at most 120 characters")
    rows = document.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("rows must contain at least one completed-checkpoint evaluation")
    sources = document.get("source_records")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source_records must preserve at least one source record")
    for index, source in enumerate(sources):
        if not isinstance(source, dict):
            raise ValueError(f"source_records[{index}] must be an object")
        for key in ("run", "path"):
            if not isinstance(source.get(key), str) or not source[key].strip():
                raise ValueError(f"source_records[{index}].{key} must be a nonempty string")
        if not isinstance(source.get("sha256"), str) or re.fullmatch(r"[0-9a-fA-F]{64}", source["sha256"]) is None:
            raise ValueError(f"source_records[{index}].sha256 must be a full hexadecimal SHA-256")
        _count(source.get("line"), f"source_records[{index}].line", positive=True)
    expected_planned, previous_episode, parsed = None, -1, []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"rows[{index}] must be an object")
        episode = _count(row["completed_episodes"], f"rows[{index}].completed_episodes")
        if episode <= previous_episode:
            raise ValueError("completed_episodes must be strictly increasing; merge duplicates upstream")
        previous_episode = episode
        metrics = {"completed_episodes": episode}
        for policy in ("sample", "nr"):
            planned, metrics[policy] = _aggregate_metrics(row[policy], f"rows[{index}].{policy}")
            if expected_planned is None:
                expected_planned = planned
            if planned != expected_planned:
                raise ValueError("planned population must match sample/NR and every checkpoint")
        parsed.append(metrics)
    return document, parsed, expected_planned, hashlib.sha256(raw).hexdigest()


def render(input_path, output_directory):
    """Render SVG/PNG/PDF with CPU Agg and an owned writable font-cache directory."""
    document, rows, planned, input_sha = load_input(input_path)
    figure_title = document.get("title", TITLE)
    output = Path(output_directory)
    if not output.is_dir():
        raise ValueError("LAB_RUN_DIR must be the launcher's existing output directory")
    x = [row["completed_episodes"] for row in rows]
    parameters = {"figure_size_inches": list(FIGURE_SIZE_INCHES), "png_dpi": PNG_DPI,
                  "x_scale": "linear", "episode_coordinates": x,
                  "smoothing": False, "error_bars": False, "significance_annotations": False,
                  "policy": "sample", "comparison": "NR", "fraction_denominator": "planned",
                  "planned_flights_per_policy_per_checkpoint": planned,
                  "risk_units": "unordered pair-seconds / flight-hour",
                  "risk_rate_formula": "sum(unordered_pair_seconds) / sum(flight_hours)",
                  "mean_flight_seconds_formula": "flight_hours * 3600 / planned",
                  "percentage_axis_limits": [-3., 103.],
                  "percentage_axis_padding": "Visual marker padding around the complete physical 0–100% range",
                  "numeric_consistency_relative_tolerance": 1e-10,
                  "numeric_consistency_absolute_tolerance": 1e-10}
    previous_mpl_config = os.environ.get("MPLCONFIGDIR")
    with tempfile.TemporaryDirectory(prefix="learning-curve-mpl-") as mpl_config:
        os.environ["MPLCONFIGDIR"] = mpl_config
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
            import matplotlib.pyplot as plt
            from matplotlib.ticker import MaxNLocator

            settings = {"font.family": "DejaVu Sans", "font.size": 10,
                        "axes.titlesize": 11, "axes.labelsize": 10,
                        "xtick.labelsize": 9, "ytick.labelsize": 9,
                        "svg.fonttype": "none", "pdf.fonttype": 42,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "savefig.facecolor": "white"}
            with plt.rc_context(settings):
                figure, axes = plt.subplots(2, 3, figsize=FIGURE_SIZE_INCHES, sharex=True)
                try:
                    blue, gray, orange = "#0072B2", "#606060", "#D55E00"
                    styles = {"sample": {"label": "Sampled PPO", "color": blue,
                                          "linestyle": "-", "marker": "o", "zorder": 3},
                              "nr": {"label": "No Resolution (NR)", "color": gray,
                                     "linestyle": "--", "marker": "s", "zorder": 2}}
                    panels = [(axes[0, 0], "completed_percent", "(a) Completed flights", "% of planned flights"),
                              (axes[0, 1], "nmac_seconds_per_flight_hour", "(b) NMAC exposure", "Unordered pair-s / flight-hour"),
                              (axes[0, 2], "lowc_seconds_per_flight_hour", "(c) LoWC exposure", "Unordered pair-s / flight-hour"),
                              (axes[1, 0], "outside_corridor_percent", "(d) Flights outside lateral corridor", "% of planned flights"),
                              (axes[1, 2], "mean_flight_seconds", "(f) Mean flight time, all planned flights", "Flight time (s)")]
                    y_limits = {}
                    for axis, metric, title, ylabel in panels:
                        for policy in ("sample", "nr"):
                            values = [row[policy][metric] for row in rows]
                            axis.plot(x, values, linewidth=1.8, markersize=4.8,
                                      clip_on=False, **styles[policy])
                        axis.set_title(title, loc="left", pad=10)
                        axis.set_ylabel(ylabel)
                        if metric.endswith("_percent"):
                            axis.set_ylim(-3., 103.)
                            axis.set_yticks(range(0, 101, 20))
                        else:
                            highest = max(row[policy][metric] for row in rows for policy in ("sample", "nr"))
                            axis.set_ylim(0., max(1., 1.12*highest))
                        y_limits[metric] = list(axis.get_ylim())
                    failures = axes[1, 1]
                    for policy, policy_name, line in (("sample", "Sampled PPO", "-"), ("nr", "NR", "--")):
                        for metric, reason, color, marker in (
                                ("timeout_percent", "timeout", blue, "o"),
                                ("route_exhausted_percent", "route exhausted", orange, "^")):
                            failures.plot(x, [row[policy][metric] for row in rows], color=color,
                                          marker=marker, linestyle=line, linewidth=1.8, markersize=4.8,
                                          label=f"{policy_name}: {reason}")
                    failures.set_title("(e) Task failures by cause", loc="left", pad=10)
                    failures.set_ylabel("% of planned flights")
                    failures.set_ylim(-3., 103.)
                    failures.set_yticks(range(0, 101, 20))
                    failures.legend(loc="upper right", fontsize=8, framealpha=.92)
                    y_limits["failure_percent"] = list(failures.get_ylim())
                    span = max(1., float(x[-1]-x[0]))
                    x_limits = ((max(0., x[0]-1.), x[0]+1.) if len(x) == 1 else
                                (max(0., x[0]-.04*span), x[-1]+.04*span))
                    for axis in axes.flat:
                        axis.set_xscale("linear")
                        axis.set_xlim(*x_limits)
                        axis.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True, min_n_ticks=2))
                        axis.grid(axis="y", color="#DADADA", linewidth=.6, alpha=.8)
                        axis.set_axisbelow(True)
                    for axis in axes[1]:
                        axis.set_xlabel("Completed training episodes")
                    handles, labels = axes[0, 0].get_legend_handles_labels()
                    figure.legend(handles, labels, loc="upper center", bbox_to_anchor=(.5, .929),
                                  ncol=2, frameon=False, fontsize=10)
                    figure.suptitle(figure_title, fontsize=16, fontweight="semibold", y=.992)
                    figure.text(.5, .949, "Development only / containment unresolved",
                                ha="center", va="center", fontsize=11, color="#8C2D04")
                    figure.text(.075, .045,
                        "Markers are checkpoint evaluations from one training trajectory; no independent replicates, smoothing or uncertainty estimates.\n"
                        "All flight fractions and mean flight time use planned flights. Shorter failed flights can lower exposure; read alongside completion and failures.",
                        ha="left", va="center", fontsize=8.7, linespacing=1.5)
                    figure.subplots_adjust(left=.075, right=.99, bottom=.15, top=.85, wspace=.32, hspace=.36)
                    for name in OUTPUT_NAMES:
                        figure.savefig(output/name, dpi=PNG_DPI, bbox_inches="tight", pad_inches=.12)
                    parameters.update(x_axis_limits=list(x_limits), y_axis_limits=y_limits)
                finally:
                    plt.close(figure)
            matplotlib_version = str(matplotlib.__version__)
            backend = str(matplotlib.get_backend())
        finally:
            if previous_mpl_config is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = previous_mpl_config
    metadata = {"schema": "execution-learning-curve-plot-metadata.v1", "scope": document["scope"],
                "input_schema": INPUT_SCHEMA, "input_path": str(input_path), "input_sha256": input_sha,
                "title": figure_title, "parameters": parameters, "checkpoint_count": len(rows),
                "renderer_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "source_records": document["source_records"],
                "source_record_verification": "Provenance copied from controller input; referenced source files are not read by this renderer.",
                "interpretation": "Paired development observations; not held-out evidence, independent replicates, an effective baseline or a containment guarantee.",
                "plotted_values": rows,
                "versions": {"python": platform.python_version(), "matplotlib": matplotlib_version, "backend": backend},
                "outputs": [{"filename": name, "sha256": hashlib.sha256((output/name).read_bytes()).hexdigest(),
                             "bytes": (output/name).stat().st_size} for name in OUTPUT_NAMES]}
    (output/"plot_metadata.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n")
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="Controller-prepared execution-learning-curve.v1 JSON")
    args = parser.parse_args(argv)
    metadata = render(args.input, os.environ["LAB_RUN_DIR"])
    print(json.dumps({"scope": metadata["scope"], "checkpoint_count": metadata["checkpoint_count"],
                      "input_sha256": metadata["input_sha256"],
                      "outputs": [item["filename"] for item in metadata["outputs"]]+["plot_metadata.json"]}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
