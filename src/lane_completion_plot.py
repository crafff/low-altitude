"""Export the saved F020 post-update trace; no simulation or model loading."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path
import platform
import tempfile
import time


OUTPUTS = ('lane_completion.svg', 'lane_completion.png', 'lane_completion.pdf')
FIGURE_SIZE = (12., 10.)
PNG_DPI = 300
DETAIL_LIMIT = 8.
SCOPE = 'Passive lane-completion predicate diagnostic; one selected flight in full traffic; no causal claim'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def finite(value, name):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def close(actual, expected, name):
    if not math.isclose(actual, expected, rel_tol=0., abs_tol=1e-8):
        raise ValueError(f'{name}: {actual} differs from {expected}')


def load_trace(csv_path, result_path):
    """Read every original row; reject incompatible identities, cadence or values."""
    csv_path, result_path = Path(csv_path), Path(result_path)
    if csv_path.stat().st_size > 16*1024**2 or result_path.stat().st_size > 4*1024**2:
        raise ValueError('Saved trace/result exceeds this bounded renderer input size')
    csv_raw, result_raw = csv_path.read_bytes(), result_path.read_bytes()
    result = json.loads(result_raw)
    if (result['schema'] != 'bluesky.lane-completion-probe-result.v1'
            or result['scope'] != SCOPE or result['all_checks_passed'] is not True
            or not all(value is True for value in result['checks'].values())):
        raise ValueError('Require the completed, validated lane-completion probe result')
    cfg, terminal = result['config'], result['target_terminal']
    if (cfg['completed_episodes'] != 100 or cfg['scenario_seed'] != 53004
            or terminal['id'] != 'F020' or terminal['type'] != 'M600'):
        raise ValueError('This figure is explicitly the fresh100 / seed53004 / M600 F020 diagnostic')
    scenario_cfg = result['checkpoint_configs']['environment_parts']['scenario']
    action_cfg = result['checkpoint_configs']['environment_parts']['action']
    dt = finite(scenario_cfg['dt_seconds'], 'dt')
    duration, entry = finite(terminal['flight_seconds'], 'duration'), finite(terminal['actual_entry_s'], 'entry')
    if dt <= 0 or duration <= 0:
        raise ValueError('Physics duration and lifetime must be positive')
    expected = round(duration/dt)
    close(duration/dt, expected, 'lifetime/dt integer row count')
    if not 0 < expected <= 6000 or any(result[key] != expected for key in (
            'expected_target_rows', 'csv_rows', 'target_samples_recorded')):
        raise ValueError('Result row counts differ from retained lifetime/dt')
    if result['csv_bytes'] != len(csv_raw):
        raise ValueError('CSV byte count differs from the saved validated result')
    close(dt, .25, 'original physics dt')
    half_width = finite(scenario_cfg['corridor_width_ft'], 'corridor width')*.3048/2.
    lane_tol = finite(action_cfg['lane_capture_tolerance_m'], 'lane tolerance')
    track_tol = finite(action_cfg['lane_capture_track_tolerance_deg'], 'track tolerance')
    close(half_width, 76.2, 'original half-width')
    close(lane_tol, 2., 'original lane tolerance')
    close(track_tol, 5., 'original track tolerance')
    reader = csv.DictReader(io.StringIO(csv_raw.decode('utf-8'), newline=''))
    header = reader.fieldnames
    required = {'scenario_seed', 'aircraft', 'type', 'sim_time_s', 'age_s', 'lane_error_m', 'track_error_deg',
                'centerline_distance_m', 'generation', 'lane_command_count', 'lane_capture_count',
                'lane_tolerance_m', 'track_tolerance_deg', 'lane_active', 'active_mapping_is_nominal',
                'capture_waypoint_active', 'post_update_category', 'outside'}
    if not header or len(header) != len(set(header)) or not required <= set(header):
        raise ValueError('CSV has missing or repeated required columns')
    strings = {'aircraft', 'type', 'corridor_id', 'native_waypoint_name', 'post_update_category'}
    rows, releases, boundaries = [], [], []
    for index, raw in enumerate(reader):
        if index >= expected or set(raw) != set(header) or any(value is None for value in raw.values()):
            raise ValueError('CSV row count or row shape differs from the complete saved trace')
        row = {}
        for name, value in raw.items():
            if name in strings:
                row[name] = value
            elif value in ('True', 'False'):
                row[name] = value == 'True'
            elif name == 'active_waypoint_nominal_mapping' and value == 'null':
                row[name] = None
            else:
                row[name] = finite(value, f'row {index+1} {name}')
        if row['aircraft'] != 'F020' or row['type'] != 'M600' or row['scenario_seed'] != 53004:
            raise ValueError('Unexpected aircraft or scenario in the saved trace')
        for name in ('lane_active', 'active_mapping_is_nominal', 'capture_waypoint_active', 'outside'):
            if type(row[name]) is not bool:
                raise ValueError(f'row {index+1}: {name} must be an explicit CSV boolean')
        for name in ('scenario_seed', 'sim_time_s', 'age_s', 'lane_error_m', 'track_error_deg',
                     'centerline_distance_m', 'lane_tolerance_m', 'track_tolerance_deg'):
            if type(row[name]) is not float:
                raise ValueError(f'row {index+1}: {name} must be numeric, not boolean')
        for name in ('generation', 'lane_command_count', 'lane_capture_count'):
            if type(row[name]) is not float or not row[name].is_integer() or row[name] < 0:
                raise ValueError(f'row {index+1}: {name} must be a nonnegative integer')
            row[name] = int(row[name])
        close(row['age_s'], (index+1)*dt, 'strictly ordered full physics cadence')
        close(row['sim_time_s'], entry+row['age_s'], 'time origin')
        close(row['lane_tolerance_m'], lane_tol, 'row lane tolerance')
        close(row['track_tolerance_deg'], track_tol, 'row track tolerance')
        if row['centerline_distance_m'] < 0 or row['outside'] != (row['centerline_distance_m'] > half_width):
            raise ValueError('Raw corridor distance/exposure flag is inconsistent')
        previous = rows[-1] if rows else None
        generation_delta = row['generation']-(previous['generation'] if previous else 0)
        capture_delta = row['lane_capture_count']-(previous['lane_capture_count'] if previous else 0)
        if generation_delta not in (0, 1) or capture_delta not in (0, 1) or row['generation'] != row['lane_command_count']:
            raise ValueError('Command generation/capture progression is inconsistent')
        if generation_delta:
            boundaries.append(dict(generation=row['generation'], elapsed_s=row['age_s'], sim_time_s=row['sim_time_s']))
        if capture_delta:
            if row['lane_active']:
                raise ValueError('Release counter increased while the post-update lane lock remains active')
            releases.append(dict(generation=row['generation'], elapsed_s=row['age_s'], sim_time_s=row['sim_time_s']))
        rows.append(row)
    if len(rows) != expected:
        raise ValueError('CSV ends before the complete retained lifetime')
    close(rows[-1]['sim_time_s'], terminal['terminal_time_s'], 'terminal sample time')
    close(sum(row['outside'] for row in rows)*dt, terminal['outside_corridor_seconds'], 'raw exposure reconciliation')
    close(max(row['centerline_distance_m'] for row in rows), terminal['max_centerline_distance_m'], 'maximum distance')
    if rows[-1]['lane_capture_count'] != terminal['action_execution']['command_counts']['lane_captures']:
        raise ValueError('Final capture count differs from the full terminal record')
    generations = []
    for group in result['generation_summary']['generations']:
        members = [row for row in rows if row['generation'] == group['generation']]
        if len(members) != group['samples'] or not members:
            raise ValueError('CSV generation count differs from the saved summary')
        close(members[0]['sim_time_s'], group['first_sample']['sim_time_s'], 'generation first sample')
        close(members[-1]['sim_time_s'], group['last_sample']['sim_time_s'], 'generation last sample')
        generations.append(dict(generation=group['generation'], samples=len(members),
            first_elapsed_s=members[0]['age_s'], last_elapsed_s=members[-1]['age_s']))
    provenance = {name: dict(path=str(path), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
                  for name, path, raw in (('csv', csv_path, csv_raw), ('result', result_path, result_raw))}
    return result, rows, dict(dt_seconds=dt, entry_sim_time_s=entry, duration_s=duration,
        half_width_m=half_width, lane_tolerance_m=lane_tol, track_tolerance_deg=track_tol,
        generations=generations, generation_boundaries=boundaries, lane_lock_releases=releases), provenance


def render(csv_path, result_path, output):
    started = time.perf_counter()
    result, rows, facts, provenance = load_trace(csv_path, result_path)
    output = Path(output)
    if not output.is_dir() or any((output/name).exists() for name in (*OUTPUTS, 'plot_metadata.json')):
        raise ValueError('Require an existing fresh LAB_RUN_DIR for these figure artifacts')
    elapsed = [row['age_s'] for row in rows]
    clipped = sum(abs(row['lane_error_m']) > DETAIL_LIMIT for row in rows)
    side_only = [row['lane_error_m'] if row['post_update_category'] == 'locked_side_error_only'
                 else math.nan for row in rows]
    previous_cache = os.environ.get('MPLCONFIGDIR')
    with tempfile.TemporaryDirectory(prefix='lane-completion-mpl-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        try:
            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            settings = {'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 11,
                'axes.labelsize': 10, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                'svg.fonttype': 'none', 'pdf.fonttype': 42, 'path.simplify': False,
                'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': 'white'}
            with plt.rc_context(settings):
                figure, axes = plt.subplots(4, 1, figsize=FIGURE_SIZE, sharex=True)
                try:
                    blue, orange, purple, green, gray = '#0072B2', '#D55E00', '#7B3294', '#009E73', '#777777'
                    specs = [('lane_error_m', '(a) Lane error: full recorded range', 'Lane error (m)', facts['lane_tolerance_m']),
                        ('lane_error_m', '(b) Lane error: detail window ±8 m (view clipped)', 'Lane error (m)', facts['lane_tolerance_m']),
                        ('track_error_deg', '(c) Track error: full recorded range', 'Track error (°)', facts['track_tolerance_deg']),
                        ('centerline_distance_m', '(d) Raw finite-centerline distance', 'Distance (m)', facts['half_width_m'])]
                    limits = {}
                    for index, (axis, (key, title, ylabel, tolerance)) in enumerate(zip(axes, specs)):
                        values = [row[key] for row in rows]
                        axis.plot(elapsed, values, color=blue, linewidth=1., zorder=3)
                        if index == 1:
                            axis.plot(elapsed, side_only, color=purple, linewidth=1.8, zorder=4)
                            axis.set_ylim(-DETAIL_LIMIT, DETAIL_LIMIT)
                            axis.set_yticks([-8, -4, -2, 0, 2, 4, 8])
                        else:
                            lower = min(0., min(values), -tolerance if index < 3 else 0.)
                            upper = max(max(values), tolerance)
                            pad = .09*max(1., upper-lower)
                            axis.set_ylim(lower-pad if index < 3 else 0., upper+pad)
                        axis.axhline(tolerance, color=orange, linestyle='--', linewidth=1., zorder=2)
                        if index < 3:
                            axis.axhline(-tolerance, color=orange, linestyle='--', linewidth=1., zorder=2)
                        for event in facts['generation_boundaries']:
                            axis.axvline(event['elapsed_s'], color=gray, linestyle=':', linewidth=.7, alpha=.65, zorder=1)
                        for event in facts['lane_lock_releases']:
                            axis.axvline(event['elapsed_s'], color=green, linewidth=.8, alpha=.7, zorder=2)
                        gen3 = next(group for group in facts['generations'] if group['generation'] == 3)
                        axis.axvspan(gen3['first_elapsed_s'], gen3['last_elapsed_s'], color=blue, alpha=.045, zorder=0)
                        axis.set_title(title, loc='left', pad=7)
                        axis.set_ylabel(ylabel)
                        axis.set_xlim(0., facts['duration_s'])
                        axis.grid(axis='y', color='#DADADA', linewidth=.6, alpha=.8)
                        axis.set_axisbelow(True)
                        limits[str(index+1)] = list(axis.get_ylim())
                    axes[0].text((gen3['first_elapsed_s']+gen3['last_elapsed_s'])/2., .92, 'Generation 3 (shaded)',
                        transform=axes[0].get_xaxis_transform(), ha='center', va='top', color='#444444', fontsize=9)
                    axes[1].text(.99, .06, f'{clipped:,}/{len(rows):,} samples outside this detail window; all retained in panel (a)',
                        transform=axes[1].transAxes, ha='right', va='bottom', fontsize=8.5,
                        bbox=dict(facecolor='white', edgecolor='none', alpha=.85, pad=2.))
                    axes[3].text(.99, .91, f'Half-width = {facts["half_width_m"]:.1f} m', transform=axes[3].transAxes,
                                 ha='right', va='top', fontsize=9, color=orange)
                    axes[-1].set_xlabel('Elapsed time from actual entry (s)')
                    legend = [Line2D([], [], color=blue, label='Recorded trace'),
                        Line2D([], [], color=purple, linewidth=1.8, label='Locked + nominal; only side tolerance fails'),
                        Line2D([], [], color=orange, linestyle='--', label='±2 m / ±5° / 76.2 m thresholds'),
                        Line2D([], [], color=gray, linestyle=':', label='Generation change'),
                        Line2D([], [], color=green, label='Lane-lock release (not CAP passage)')]
                    figure.suptitle('F020 (M600): sampled-policy lane tracking', fontsize=16, fontweight='semibold', y=.988)
                    figure.text(.5, .956, 'Fresh-refresh100 · seed 53004 · full 30-aircraft replay · development only',
                                ha='center', fontsize=10, color='#444444')
                    figure.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .941), ncol=2,
                                  frameon=False, fontsize=9, handlelength=2.8)
                    figure.text(.105, .035,
                        f'Lines connect all {len(rows):,} post-update {facts["dt_seconds"]:g} s samples; no smoothing or fitting. Only panel (b) clips its view.\n'
                        'Raw entry/terminal intervals retained. Lock release differs from CAP passage; observations imply no causal or continuous-time safety claim.',
                        fontsize=8.5, va='center', linespacing=1.5)
                    figure.subplots_adjust(left=.105, right=.985, bottom=.105, top=.835, hspace=.36)
                    for name in OUTPUTS:
                        figure.savefig(output/name, dpi=PNG_DPI, bbox_inches='tight', pad_inches=.12)
                finally:
                    plt.close(figure)
            mpl_version = matplotlib.__version__
        finally:
            if previous_cache is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = previous_cache
    if any(digest(item['path']) != item['sha256'] for item in provenance.values()):
        raise RuntimeError('A declared input changed while rendering')
    metadata = dict(schema='lane-completion-plot.v1', scope=SCOPE, source_files=provenance,
        renderer_sha256=digest(__file__), rows=len(rows), all_source_rows_retained=True,
        source_result_all_checks_passed=result['all_checks_passed'], source_model_sha256=result['model_before_sha256'],
        terminal_status=result['target_terminal']['status'], facts=facts,
        plotted_columns=['age_s', 'lane_error_m', 'track_error_deg', 'centerline_distance_m', 'post_update_category'],
        raw_column_ranges={key: dict(minimum=min(row[key] for row in rows), maximum=max(row[key] for row in rows))
            for key in ('lane_error_m', 'track_error_deg', 'centerline_distance_m')},
        parameters=dict(figure_size_inches=list(FIGURE_SIZE), png_dpi=PNG_DPI, x_axis='elapsed time from actual entry',
            x_axis_limits_s=[0., facts['duration_s']], panel_y_limits=limits, detail_window_m=[-DETAIL_LIMIT, DETAIL_LIMIT],
            detail_window_clipped_samples=clipped, clipped_values_modified=False, smoothing=False, fitting=False,
            downsampling=False, matplotlib_path_simplification=False, error_bars=False,
            event_definition='Markers use the first post-update sample with increased generation or lane_capture_count. Release is not CAP passage.',
            highlight_definition='Detail overlay uses original locked_side_error_only rows. Generation3 shading covers its first-to-last recorded sample, not a claim of continuous lock.',
            interpretation='One selected development trace. Connecting sampled points establishes neither continuous-time safety nor a causal mechanism.'),
        software=dict(python=platform.python_version(), matplotlib=mpl_version), wall_seconds=time.perf_counter()-started,
        outputs={name: dict(sha256=digest(output/name), bytes=(output/name).stat().st_size) for name in OUTPUTS})
    (output/'plot_metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--csv', required=True)
    parser.add_argument('--result', required=True)
    args = parser.parse_args(argv)
    if os.environ.get('LAB_RUN_DIR') != '/output' or Path.cwd() != Path('/workspace'):
        raise ValueError('Render the declared read-only inputs through the isolated lab launcher')
    metadata = render(args.csv, args.result, '/output')
    print(json.dumps(dict(rows=metadata['rows'], outputs=list(metadata['outputs']), wall_seconds=metadata['wall_seconds'])))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
