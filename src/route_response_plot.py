"""Render controller-verified fixed route responses; never rerun native physics.

Input: route-response-plot.v1 with all 50 response rows and four full traces.
Outputs: route_response.{svg,png,pdf} and plot_metadata.json in LAB_RUN_DIR.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import tempfile


INPUT_SCHEMA = 'route-response-plot.v1'
TITLE = 'Finite fixed-action route responses'
FIGURE_SIZE_INCHES = (14., 9.)
PNG_DPI = 300
OUTPUT_NAMES = ('route_response.svg', 'route_response.png', 'route_response.pdf')
TRACE_IDS = ('native-amzn-north-s0-l1', 'refresh-amzn-north-s0-l1',
             'refresh-amzn-north-s0-l2', 'native-amzn-north-s2-l1')
ARRAY_FIELDS = ('sim_time_s', 'deviation_m', 'cached_turn_distance_m',
                'instantaneous_bank_radius_m', 'active_wp_nominal_mapping')
SPEED_RATIOS = (.5, .8, 1., 1.05)


def _reject_constant(value):
    raise ValueError(f'Non-finite JSON constant: {value}')


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def _finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError('Input contains a non-finite number')
    if isinstance(value, dict):
        for child in value.values():
            _finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            _finite_tree(child)


def _number(value, label):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label} must be a finite nonnegative number')
    return float(value)


def expected_cases():
    cases = {f'native-mavic-{turn}': ('native', turn, 'Mavic', 2, 1)
             for turn in ('north', 'south')}
    for arm in ('native', 'refresh'):
        for turn in ('north', 'south'):
            for speed in range(4):
                for lane in range(3):
                    cases[f'{arm}-amzn-{turn}-s{speed}-l{lane}'] = (arm, turn, 'Amzn', speed, lane)
    return cases


def load_input(path):
    """Reject partial panels, non-finite values and hidden trace downsampling."""
    raw = Path(path).read_bytes()
    document = json.loads(raw, parse_constant=_reject_constant, object_pairs_hook=_unique_object)
    _finite_tree(document)
    if not isinstance(document, dict) or document.get('schema') != INPUT_SCHEMA:
        raise ValueError(f'Expected schema {INPUT_SCHEMA}')
    if _number(document['half_width_m'], 'half_width_m') != 76.2:
        raise ValueError('This fixed diagnostic retains the 76.2 m corridor half-width')
    sources = document['source_records']
    if not isinstance(sources, list) or not sources:
        raise ValueError('source_records must contain source path/SHA-256 objects')
    for source in sources:
        if (not isinstance(source, dict) or not isinstance(source.get('path'), str)
                or not source['path'].strip() or not isinstance(source.get('sha256'), str)
                or re.fullmatch(r'[0-9a-fA-F]{64}', source['sha256']) is None):
            raise ValueError('Each source record requires a nonempty path and a full SHA-256')
    expected = expected_cases()
    rows = document['rows']
    if not isinstance(rows, list) or len(rows) != 50:
        raise ValueError('Preserve all 50 predeclared response rows, including failures')
    by_case = {}
    for row in rows:
        case_id = row['case_id']
        if case_id not in expected or case_id in by_case:
            raise ValueError(f'Unexpected or repeated case: {case_id}')
        identity = tuple(row[k] for k in ('arm', 'turn', 'aircraft_type', 'speed_index', 'lane_index'))
        if identity != expected[case_id] or type(row['speed_index']) is not int or type(row['lane_index']) is not int:
            raise ValueError(f'Case identity/indices disagree with {case_id}')
        for key in ('max_deviation_m', 'outside_seconds'):
            _number(row[key], f'{case_id}.{key}')
        if not isinstance(row['status'], str) or not row['status'].strip():
            raise ValueError(f'Preserve an explicit terminal status for {case_id}')
        by_case[case_id] = row
    traces = document['traces']
    if not isinstance(traces, list) or len(traces) != len(TRACE_IDS):
        raise ValueError('Exactly the four predeclared complete traces are required')
    by_trace = {}
    for trace in traces:
        case_id = trace['case_id']
        if case_id not in TRACE_IDS or case_id in by_trace:
            raise ValueError(f'Unexpected or repeated trace: {case_id}')
        arrays = [trace[field] for field in ARRAY_FIELDS]
        if (any(not isinstance(values, list) for values in arrays)
                or len(arrays[0]) < 2 or len({len(values) for values in arrays}) != 1):
            raise ValueError(f'{case_id}: trace arrays must have matched lengths >= 2')
        for field, values in zip(ARRAY_FIELDS[:-1], arrays[:-1]):
            for value in values:
                _number(value, f'{case_id}.{field}')
        times = trace['sim_time_s']
        if times[0] != 0 or any(b-a != .25 for a, b in zip(times, times[1:])):
            raise ValueError(f'{case_id}: retain every 0.25 s sample from t=0')
        mapping = trace['active_wp_nominal_mapping']
        if any(type(value) is not int or value not in (1, 2) for value in mapping):
            raise ValueError(f'{case_id}: these four recorded traces require explicit nominal mapping 1 or 2')
        advance = _number(trace['first_advance_time_s'], f'{case_id}.first_advance_time_s')
        trigger = _number(trace['trigger_time_s'], f'{case_id}.trigger_time_s')
        first_second = next((i for i, value in enumerate(mapping) if value==2), None)
        if (first_second is None or times[first_second] != advance or not 0 < trigger < advance
                or trigger not in times or any(value != 2 for value in mapping[first_second:])):
            raise ValueError(f'{case_id}: trigger or first advance disagrees with actual nominal mapping')
        if not math.isclose(max(trace['deviation_m']), by_case[case_id]['max_deviation_m'], rel_tol=0., abs_tol=1e-9):
            raise ValueError(f'{case_id}: full trace maximum disagrees with its retained summary')
        by_trace[case_id] = trace
    return document, by_case, by_trace, hashlib.sha256(raw).hexdigest()


def first_leg_indices(trace):
    """Exclude the actual advance sample and every final-leg cached zero."""
    return [i for i, (t, mapping) in enumerate(zip(trace['sim_time_s'], trace['active_wp_nominal_mapping']))
            if t < trace['first_advance_time_s'] and mapping==1]


def render(input_path, output_directory):
    document, rows, traces, input_sha = load_input(input_path)
    output = Path(output_directory)
    if not output.is_dir():
        raise ValueError('LAB_RUN_DIR must be an existing launcher output directory')
    if any((output/name).exists() for name in (*OUTPUT_NAMES, 'plot_metadata.json')):
        raise ValueError('Refusing to overwrite existing figure artifacts')
    half_width = document['half_width_m']
    heatmaps = {arm: [[rows[f'{arm}-amzn-north-s{s}-l{l}']['max_deviation_m']
                      for l in range(3)] for s in range(4)] for arm in ('native', 'refresh')}
    highest = max(value for matrix in heatmaps.values() for row in matrix for value in row)
    cache_ids = TRACE_IDS[:2]
    cache_indices = {case_id: first_leg_indices(traces[case_id]) for case_id in cache_ids}
    outer = rows[TRACE_IDS[2]]
    outer_trace = traces[TRACE_IDS[2]]
    above_indices = [i for i, value in enumerate(outer_trace['deviation_m']) if value > half_width]
    terminal_only = above_indices == [len(outer_trace['sim_time_s'])-1]
    amzn = [row for row in rows.values() if row['aircraft_type']=='Amzn']
    outside_count = sum(row['outside_seconds'] > 0 for row in amzn)
    parameters = dict(figure_size_inches=list(FIGURE_SIZE_INCHES), png_dpi=PNG_DPI,
        smoothing=False, downsampling=False, confidence_intervals=False, path_simplification=False,
        heatmap_scale='shared linear, nonnegative maximum deviation in metres',
        heatmap_color_limits=[0., highest], heatmap_annotation_decimal_places=3,
        speed_ratios=list(SPEED_RATIOS), lane_offsets_m=[-half_width, 0., half_width],
        heatmap_selection='All 24 north-turn Amzn responses; all other rows retained in metadata',
        trace_selection=list(TRACE_IDS), trace_sample_interval_s=.25,
        cache_selection=list(cache_ids), cache_filter='sim_time_s < first_advance_time_s AND active_wp_nominal_mapping == 1',
        cache_plotted_indices=cache_indices,
        radius_quantity='Recorded instantaneous v^2 / (g tan(25 degrees)); not measured flown-path curvature',
        source_numeric_consistency_absolute_tolerance_m=1e-9)
    previous_mpl_config = os.environ.get('MPLCONFIGDIR')
    with tempfile.TemporaryDirectory(prefix='route-response-mpl-') as mpl_config:
        os.environ['MPLCONFIGDIR'] = mpl_config
        try:
            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
            from matplotlib.colors import Normalize
            settings = {'font.family': 'DejaVu Sans', 'font.size': 10,
                'axes.titlesize': 11, 'axes.labelsize': 10, 'xtick.labelsize': 9,
                'ytick.labelsize': 9, 'svg.fonttype': 'none', 'pdf.fonttype': 42,
                'axes.spines.top': False, 'axes.spines.right': False,
                'savefig.facecolor': 'white', 'path.simplify': False}
            with plt.rc_context(settings):
                figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE_INCHES)
                try:
                    cmap, norm = plt.get_cmap('viridis'), Normalize(vmin=0., vmax=highest)
                    for axis, arm, panel in zip(axes[0], ('native', 'refresh'), ('a', 'b')):
                        matrix = heatmaps[arm]
                        mesh = axis.imshow(matrix, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')
                        for speed in range(4):
                            for lane in range(3):
                                value = matrix[speed][lane]
                                red, green, blue, _ = cmap(norm(value))
                                color = 'black' if .2126*red+.7152*green+.0722*blue > .55 else 'white'
                                axis.text(lane, speed, f'{value:.3f}', ha='center', va='center', color=color, fontsize=11)
                        selected = [row for row in amzn if row['arm']==arm and row['turn']=='north']
                        arrived = sum(row['status']=='arrived' for row in selected)
                        label = 'Native guidance' if arm=='native' else 'Ordinary flyby refresh'
                        axis.set_title(f'({panel}) {label}: maximum deviation (m)', loc='left', pad=10)
                        axis.set_xticks(range(3), [f'−{half_width:g}\ninner', '0\ncentre', f'+{half_width:g}\nouter'])
                        axis.set_yticks(range(4), [f'{ratio:.2f}×' for ratio in SPEED_RATIOS])
                        axis.set_xlabel(f'Absolute lane target (m); {arrived}/12 arrived')
                        axis.set_ylabel('Requested / nominal speed')
                    cbar_axis = figure.add_axes([.935, .585, .014, .29])
                    figure.colorbar(mesh, cax=cbar_axis, label='Maximum distance to nominal polyline (m)')

                    colors = ('#0072B2', '#D55E00', '#009E73', '#656565')
                    names = ('Native: 0.50×, centre', 'Refresh: 0.50×, centre',
                             'Refresh: 0.50×, outer', 'Native: 1.00×, centre')
                    deviation_axis = axes[1, 0]
                    for case_id, color, name in zip(TRACE_IDS, colors, names):
                        trace = traces[case_id]
                        deviation_axis.plot(trace['sim_time_s'], trace['deviation_m'], color=color,
                                            linewidth=1.6, label=name, linestyle='--' if case_id==TRACE_IDS[3] else '-')
                        i = trace['sim_time_s'].index(trace['first_advance_time_s'])
                        deviation_axis.plot(trace['sim_time_s'][i], trace['deviation_m'][i], 'o',
                                            markersize=4, markerfacecolor='white', markeredgecolor=color)
                    deviation_axis.axhline(half_width, color='#333333', linestyle=':', linewidth=1.2,
                                           label=f'Half-width: {half_width:g} m')
                    for trigger in sorted({traces[key]['trigger_time_s'] for key in TRACE_IDS}):
                        deviation_axis.axvline(trigger, color='#888888', linestyle=':', linewidth=.9)
                    deviation_axis.plot(outer_trace['sim_time_s'][-1], outer_trace['deviation_m'][-1],
                                        marker='x', color=colors[2], markersize=6, markeredgewidth=1.5)
                    deviation_axis.set_title('(c) Complete recorded deviation traces', loc='left', pad=10)
                    deviation_axis.set_xlabel('Simulation time (s); circles = first native leg advance')
                    deviation_axis.set_ylabel('Distance to nominal polyline (m)')
                    deviation_axis.set_ylim(0., max(traces[key]['deviation_m'][i] for key in TRACE_IDS
                        for i in range(len(traces[key]['deviation_m'])))*1.12)
                    deviation_axis.set_xlim(0., max(traces[key]['sim_time_s'][-1] for key in TRACE_IDS)*1.025)
                    deviation_axis.legend(loc='upper right', fontsize=8.4, framealpha=.92)

                    cache_axis = axes[1, 1]
                    for case_id, color, label in zip(cache_ids, colors[:2], ('Native', 'Refresh')):
                        trace, indices = traces[case_id], cache_indices[case_id]
                        times = [trace['sim_time_s'][i] for i in indices]
                        cache_axis.plot(times, [trace['cached_turn_distance_m'][i] for i in indices],
                            color=color, linewidth=2., label=f'{label} cached distance')
                        cache_axis.plot(times, [trace['instantaneous_bank_radius_m'][i] for i in indices],
                            color=color, linewidth=1.2, linestyle='--' if label=='Native' else ':',
                            label=f'{label} analytic radius')
                        cache_axis.plot(times[-1], trace['cached_turn_distance_m'][indices[-1]],
                                        marker='s', color=color, markersize=4)
                        cache_axis.axvline(trace['first_advance_time_s'], color=color, linestyle=':', alpha=.45, linewidth=.8)
                    for trigger in sorted({traces[key]['trigger_time_s'] for key in cache_ids}):
                        cache_axis.axvline(trigger, color='#777777', linestyle='-.', linewidth=.9)
                        cache_axis.text(trigger+.8, .03, 'action', rotation=90, fontsize=8, va='bottom',
                                        color='#555555', transform=cache_axis.get_xaxis_transform())
                    cache_axis.set_title('(d) Half speed, centre: before first leg advance', loc='left', pad=10)
                    cache_axis.set_xlabel('Simulation time (s); each trace stops before its own advance')
                    cache_axis.set_ylabel('Cached distance / analytic radius (m)')
                    cache_axis.set_xlim(0., max(traces[key]['first_advance_time_s'] for key in cache_ids)*1.04)
                    cache_axis.set_ylim(bottom=0.)
                    cache_axis.legend(loc='upper right', fontsize=8.4, framealpha=.92)
                    cache_axis.text(.02, .43, r'$R(v,25^\circ)=v^2/(g\tan25^\circ)$',
                                    transform=cache_axis.transAxes, va='top', fontsize=10)
                    for axis in axes[1]:
                        axis.grid(axis='both', color='#DDDDDD', linewidth=.6, alpha=.7)
                        axis.set_axisbelow(True)
                    figure.suptitle(TITLE, fontsize=16, fontweight='semibold', y=.98)
                    figure.text(.49, .938,
                        'Mirrored 5 NM routes; north-turn Amzn shown | nominal altitude, 25° bank, no intruders',
                        ha='center', fontsize=10.5)
                    figure.text(.065, .105,
                        f'50 fixed responses (26 native, 24 refresh), not independent random scenarios. Stored outside time is positive in {outside_count}/48 Amzn cases.\n'
                        'No smoothing, downsampling, uncertainty estimates or search over all feasible policies; all 50 outcomes remain in metadata.',
                        fontsize=8.5, linespacing=1.5, va='top')
                    location = 'sole exceedance at the final recorded sample' if terminal_only else 'recorded exceedance retained'
                    figure.text(.065, .048,
                        f'Refresh 0.50× outer: max {outer["max_deviation_m"]:.6f} m '
                        f'(+{outer["max_deviation_m"]-half_width:.6f} m), stored outside {outer["outside_seconds"]:g} s; {location}.\n'
                        'Controller identifies this endpoint sample as post-exit; endpoint bookkeeping is under review. No containment guarantee is inferred.',
                        fontsize=8.5, linespacing=1.5, va='top', color='#7B3B00')
                    figure.subplots_adjust(left=.065, right=.91, bottom=.22, top=.875, wspace=.29, hspace=.57)
                    for name in OUTPUT_NAMES:
                        figure.savefig(output/name, dpi=PNG_DPI, bbox_inches='tight', pad_inches=.12)
                    parameters['axis_limits'] = {name: dict(x=list(axis.get_xlim()), y=list(axis.get_ylim()))
                        for name, axis in zip(('native_heatmap', 'refresh_heatmap', 'deviation', 'cache'), axes.flat)}
                finally:
                    plt.close(figure)
            versions = dict(python=platform.python_version(), matplotlib=str(matplotlib.__version__),
                            backend=str(matplotlib.get_backend()))
        finally:
            if previous_mpl_config is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = previous_mpl_config
    metadata = dict(schema='route-response-plot-metadata.v1', input_schema=INPUT_SCHEMA,
        title=TITLE, input_path=str(input_path), input_sha256=input_sha,
        renderer_source_path=str(Path(__file__).resolve()),
        renderer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_records=document['source_records'],
        source_record_verification='Controller-provided provenance retained; renderer does not reopen referenced run files.',
        scope='Finite deterministic fixed responses on mirrored routes; not independent replicates or a feasible-policy search.',
        parameters=parameters, versions=versions, half_width_m=half_width,
        row_count=len(rows), status_counts=dict(Counter(row['status'] for row in rows.values())),
        amzn_positive_stored_outside_count=outside_count,
        endpoint_caution=dict(case_id=TRACE_IDS[2], max_deviation_m=outer['max_deviation_m'],
            excess_m=outer['max_deviation_m']-half_width, stored_outside_seconds=outer['outside_seconds'],
            sole_trace_exceedance_at_final_sample=terminal_only,
            interpretation='Controller identifies final sample as post-exit; endpoint bookkeeping remains under investigation. No values are corrected or rounded to zero.'),
        plotted_heatmaps=heatmaps, all_response_rows=document['rows'], full_traces=document['traces'],
        outputs=[dict(filename=name, sha256=hashlib.sha256((output/name).read_bytes()).hexdigest(),
                      bytes=(output/name).stat().st_size) for name in OUTPUT_NAMES])
    with (output/'plot_metadata.json').open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Controller-prepared route-response-plot.v1 JSON')
    args = parser.parse_args(argv)
    metadata = render(args.input, os.environ['LAB_RUN_DIR'])
    print(json.dumps(dict(row_count=metadata['row_count'], input_sha256=metadata['input_sha256'],
        outputs=[*OUTPUT_NAMES, 'plot_metadata.json']), allow_nan=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
