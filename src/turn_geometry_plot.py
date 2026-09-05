"""Ideal circular 90-degree turn geometry, separate from saved native responses.

No trajectory fitting, native simulation, controller search or containment claim.
The controller supplies turn-geometry-plot.v1 JSON; exports use LAB_RUN_DIR.
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


SCHEMA = 'turn-geometry-plot.v1'
A = 1.-1./math.sqrt(2.)
LANES = ('centre', 'inner', 'outer')
CASE_IDS = ('mavic', 'amzn_half', 'amzn_nominal')
OUTPUTS = ('turn_geometry.svg', 'turn_geometry.png', 'turn_geometry.pdf')
COMPARISONS = ('same_midpoint_protocol', 'different_start_capture_protocol',
               'excluded_final_post_exit_sample')
FIGURE_SIZE = (14., 10.)
COLORS = dict(centre='#0072B2', inner='#D55E00', outer='#009E73')


def nonnegative(value, name, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0 or positive and value==0:
        raise ValueError(f'{name} must be finite and {"positive" if positive else "nonnegative"}')
    return float(value)


def lane_offset(lane, width):
    if lane not in LANES:
        raise ValueError(f'Unknown lane {lane}')
    return {'centre': 0., 'inner': -width, 'outer': width}[lane]


def ideal_max_distance(radius, width, lane):
    """Exact maximum over the prescribed quarter-circle family, including endpoints.

    R=0 is a degenerate limit used only to close the normalized plot. The physical
    radius cases must be positive. This is not an optimum over admissible paths.
    """
    radius, width = nonnegative(radius, 'radius'), nonnegative(width, 'width', positive=True)
    lane_offset(lane, width)
    if lane=='centre':
        return A*radius
    if lane=='inner':
        return width+A*radius
    if radius < width:
        return math.sqrt(2.)*(width-A*radius)
    if radius <= 2.*width/A:
        return width
    return A*radius-width


def arc_point(theta, radius, offset):
    """Gamma: incoming x<=0,y=0; outgoing x=0,y>=0; positive offset is right."""
    return offset-radius+radius*math.sin(theta), radius-offset-radius*math.cos(theta)


def closest_on_nominal_rays(point):
    """Euclidean projection onto a union of rays, not their infinite full lines."""
    x, y = point
    candidates = ((min(x, 0.), 0.), (0., max(y, 0.)))
    distances = [math.hypot(x-p[0], y-p[1]) for p in candidates]
    best = min(distances)
    points = []
    for candidate, distance in zip(candidates, distances):
        if math.isclose(distance, best, rel_tol=1e-12, abs_tol=1e-12) and candidate not in points:
            points.append(candidate)
    return best, points


def maximum_witness(radius, width, lane):
    """Choose an exact maximum witness: symmetric midpoint or a plateau endpoint."""
    theta = 0. if lane=='outer' and width <= radius <= 2.*width/A else math.pi/4.
    point = arc_point(theta, radius, lane_offset(lane, width))
    distance, projections = closest_on_nominal_rays(point)
    return dict(theta_rad=theta, point_m=list(point), projections_m=[list(p) for p in projections],
                projected_distance_m=distance, analytic_max_distance_m=ideal_max_distance(radius, width, lane))


def fits_finite_legs(radius, offset, legs):
    return radius-offset <= min(legs)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key {key}')
        result[key] = value
    return result


def _reject_nonfinite(value):
    raise ValueError(f'Non-finite JSON constant {value}')


def load_input(path):
    raw = Path(path).read_bytes()
    document = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_nonfinite)
    if document['schema'] != SCHEMA or nonnegative(document['half_width_m'], 'half_width_m') != 76.2:
        raise ValueError('Require turn-geometry-plot.v1 and half-width 76.2 m')
    legs = document['leg_lengths_m']
    if not isinstance(legs, list) or len(legs) != 2 or any(nonnegative(v, 'leg', positive=True) != 4630. for v in legs):
        raise ValueError('Require the declared two 4630 m nominal legs')
    sources = document['source_records']
    if not isinstance(sources, list) or not sources:
        raise ValueError('Preserve source records')
    for source in sources:
        if (not isinstance(source['path'], str) or not source['path'].strip()
                or not isinstance(source['sha256'], str) or not re.fullmatch(r'[0-9a-fA-F]{64}', source['sha256'])):
            raise ValueError('Each source requires a path and full SHA-256')
    cases = document['radius_cases']
    if not isinstance(cases, list) or len(cases) != 3 or {c['id'] for c in cases} != set(CASE_IDS):
        raise ValueError('Preserve the three declared aircraft/radius cases')
    by_case = {c['id']: c for c in cases}
    for case in cases:
        radius = nonnegative(case['radius_m'], 'radius_m', positive=True)
        if not isinstance(case['label'], str) or not case['label'].strip() or '\n' in case['label']:
            raise ValueError('Each radius case needs a single-line label')
        if not all(fits_finite_legs(radius, lane_offset(lane, document['half_width_m']), legs) for lane in LANES):
            raise ValueError('The prescribed circular turn does not fit the finite nominal legs')
    rows = document['native_comparisons']
    if not isinstance(rows, list) or len(rows) != 9:
        raise ValueError('Preserve all nine native comparison rows')
    by_row = {}
    for row in rows:
        key = row['radius_case_id'], row['lane']
        if key[0] not in CASE_IDS or key[1] not in LANES or key in by_row:
            raise ValueError('Repeated or unknown native comparison case/lane')
        nonnegative(row['max_deviation_m'], 'native maximum')
        expected = ('different_start_capture_protocol' if key[0]=='mavic' and key[1]!='centre'
                    else 'excluded_final_post_exit_sample' if key==('amzn_half', 'outer')
                    else 'same_midpoint_protocol')
        if row['comparison'] != expected:
            raise ValueError(f'Preserve declared protocol/exclusion for {key}: {expected}')
        for name in ('source_case_id', 'source_path'):
            if not isinstance(row[name], str) or not row[name].strip():
                raise ValueError('Native comparison must retain its source case and path')
        by_row[key] = row
    return document, by_case, by_row, hashlib.sha256(raw).hexdigest()


def _geometry_panel(axis, case, width, panel):
    from matplotlib.patches import Polygon
    radius = case['radius_m']
    pad = max(width*.5, radius*.08)
    xlow, xhigh, ylow, yhigh = -radius-width-pad, width+pad, -width-pad, radius+width+pad
    outer_corner = [(width*math.cos(-math.pi/2.+i*math.pi/512.),
                     width*math.sin(-math.pi/2.+i*math.pi/512.)) for i in range(257)]
    tube = [(xlow, -width), *outer_corner, (width, yhigh), (-width, yhigh), (-width, width), (xlow, width)]
    axis.add_patch(Polygon(tube, closed=True, facecolor='#E9EDF0', edgecolor='none', zorder=0))
    axis.plot([xlow, 0.], [-width, -width], color='#A2AAB0', linewidth=.8)
    axis.plot(*zip(*outer_corner), color='#A2AAB0', linewidth=.8)
    axis.plot([width, width], [0., yhigh], color='#A2AAB0', linewidth=.8)
    axis.plot([xlow, -width, -width], [width, width, yhigh], color='#A2AAB0', linewidth=.8)
    axis.plot([xlow, 0., 0.], [0., 0., yhigh], color='#222222', linewidth=1.2, zorder=2)
    witnesses = {}
    for lane in LANES:
        points = [arc_point(i*math.pi/1024., radius, lane_offset(lane, width)) for i in range(513)]
        axis.plot(*zip(*points), color=COLORS[lane], linewidth=2., zorder=3)
        witness = maximum_witness(radius, width, lane)
        witnesses[lane] = witness
        point = witness['point_m']
        axis.plot(*point, marker='o', color=COLORS[lane], markersize=4, zorder=5)
        for projection in witness['projections_m']:
            axis.plot([point[0], projection[0]], [point[1], projection[1]],
                      color=COLORS[lane], linestyle=':', linewidth=1.1, zorder=4)
            axis.plot(*projection, marker='+', color=COLORS[lane], markersize=5, zorder=4)
        # Label each maximum via offset text, keeping all projected distances visible.
        offsets = {'centre': (8, -15), 'inner': (-8, 10), 'outer': (8, -16)}
        axis.annotate(f'{witness["analytic_max_distance_m"]:.3f} m', point,
                      textcoords='offset points', xytext=offsets[lane],
                      ha='right' if lane=='inner' else 'left', fontsize=8.2, color=COLORS[lane],
                      bbox=dict(facecolor='white', edgecolor='none', alpha=.8, pad=1.5))
    axis.set_aspect('equal', adjustable='box')
    axis.set_xlim(xlow, xhigh)
    axis.set_ylim(ylow, yhigh)
    axis.set_xlabel('East coordinate x (m)')
    axis.set_ylabel('North coordinate y (m)')
    axis.set_title(f'({panel}) {case["label"]}: ideal R = {radius:.3f} m', loc='left', pad=10)
    axis.grid(color='#CCCCCC', linewidth=.4, alpha=.4)
    axis.set_axisbelow(True)
    return dict(radius_m=radius, witnesses=witnesses, x_limits_m=[xlow, xhigh], y_limits_m=[ylow, yhigh])


def render(input_path, output_directory):
    document, cases, native, input_sha = load_input(input_path)
    width = document['half_width_m']
    output = Path(output_directory)
    if not output.is_dir() or any((output/name).exists() for name in (*OUTPUTS, 'plot_metadata.json')):
        raise ValueError('Require an existing LAB_RUN_DIR without prior figure artifacts')
    comparison_rows = []
    for case_id in CASE_IDS:
        for lane in LANES:
            source = native[case_id, lane]
            prediction = ideal_max_distance(cases[case_id]['radius_m'], width, lane)
            eligible = source['comparison']=='same_midpoint_protocol'
            comparison_rows.append(dict(source, ideal_max_distance_m=prediction,
                native_minus_ideal_m=source['max_deviation_m']-prediction if eligible else None,
                discrepancy_scope='Descriptive only: native response is not the ideal constant-radius model' if eligible
                                  else 'Excluded from turn-error comparison; original maximum retained'))
    ratio_max = max(8., 1.08*max(case['radius_m']/width for case in cases.values()))
    ratios = sorted(set([ratio_max*i/1200 for i in range(1201)]+[1., 1./A, 2./A]
                        +[case['radius_m']/width for case in cases.values()]))
    normalized = {lane: [ideal_max_distance(ratio, 1., lane) for ratio in ratios] for lane in LANES}
    geometry_metadata = {}
    previous = os.environ.get('MPLCONFIGDIR')
    with tempfile.TemporaryDirectory(prefix='turn-geometry-mpl-') as cache:
        os.environ['MPLCONFIGDIR'] = cache
        try:
            import matplotlib
            matplotlib.use('Agg', force=True)
            import matplotlib.pyplot as plt
            from matplotlib.lines import Line2D
            settings = {'font.family': 'DejaVu Sans', 'font.size': 9.5, 'axes.titlesize': 11,
                'axes.labelsize': 9.5, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
                'svg.fonttype': 'none', 'pdf.fonttype': 42, 'path.simplify': False,
                'axes.spines.top': False, 'axes.spines.right': False, 'savefig.facecolor': 'white'}
            with plt.rc_context(settings):
                figure, axes = plt.subplots(2, 2, figsize=FIGURE_SIZE)
                try:
                    normalized_axis = axes[0, 0]
                    normalized_axis.axhspan(0., 1., facecolor='#EEF3EF', zorder=0)
                    for lane in LANES:
                        normalized_axis.plot(ratios, normalized[lane], color=COLORS[lane], linewidth=2.)
                    normalized_axis.axhline(1., color='#333333', linestyle='--', linewidth=1.)
                    for case_id in CASE_IDS:
                        case = cases[case_id]
                        ratio = case['radius_m']/width
                        normalized_axis.axvline(ratio, color='#AAAAAA', linestyle=':', linewidth=.8)
                        for lane in LANES:
                            normalized_axis.plot(ratio, ideal_max_distance(ratio, 1., lane),
                                                 marker='o', color=COLORS[lane], markersize=4.)
                        normalized_axis.annotate(case['label'], (ratio, .97), xycoords=('data', 'axes fraction'),
                            textcoords='offset points', xytext=(3, 0), rotation=90, va='top', fontsize=8.)
                    normalized_axis.set(xlim=(0., ratio_max), ylim=(0., max(normalized['inner'])*1.07),
                        xlabel='Turn radius / half-width, R/w', ylabel='Maximum distance / half-width, D/w')
                    normalized_axis.set_title('(a) Exact maxima in the ideal arc family', loc='left', pad=10)
                    normalized_axis.grid(color='#DDDDDD', linewidth=.5)
                    normalized_axis.set_axisbelow(True)
                    normalized_axis.text(.97, .05,
                        f'Centre: R/w ≤ {1./A:.3f}\nOuter: 1 ≤ R/w ≤ {2./A:.3f}\nInner: exceeds w for every R > 0',
                        transform=normalized_axis.transAxes, ha='right', va='bottom', fontsize=8.6,
                        bbox=dict(facecolor='white', edgecolor='#CCCCCC', alpha=.93, pad=4.))
                    geometry_metadata['mavic'] = _geometry_panel(axes[0, 1], cases['mavic'], width, 'b')
                    geometry_metadata['amzn_half'] = _geometry_panel(axes[1, 0], cases['amzn_half'], width, 'c')
                    table_axis = axes[1, 1]
                    table_axis.set_axis_off()
                    table_axis.set_title('(d) Ideal maxima and saved native maxima', loc='left', pad=10)
                    short_labels = {'mavic':'Mavic', 'amzn_half':'Amzn 0.5×', 'amzn_nominal':'Amzn 1.0×'}
                    body = []
                    for row in comparison_rows:
                        flag = row['comparison']
                        note = {'same_midpoint_protocol':'midpoint', 'different_start_capture_protocol':'start capture †',
                                'excluded_final_post_exit_sample':'post-exit ‡'}[flag]
                        body.append([short_labels[row['radius_case_id']], row['lane'],
                            f'{row["ideal_max_distance_m"]:.3f}', f'{row["max_deviation_m"]:.3f}', note])
                    table = table_axis.table(cellText=body,
                        colLabels=['Case', 'Lane', 'Ideal (m)', 'Native (m)', 'Protocol'],
                        colWidths=[.21, .15, .18, .18, .28], cellLoc='center', colLoc='center', bbox=[0., .23, 1., .77])
                    table.auto_set_font_size(False)
                    table.set_fontsize(8.7)
                    for (row, column), cell in table.get_celld().items():
                        cell.set_edgecolor('#DDDDDD')
                        cell.set_linewidth(.5)
                        if row==0:
                            cell.set_facecolor('#E9EDF0')
                            cell.set_text_props(fontweight='semibold')
                        elif comparison_rows[row-1]['comparison']!='same_midpoint_protocol':
                            cell.set_facecolor('#FFF3DF')
                    excluded = native['amzn_half', 'outer']['max_deviation_m']
                    table_axis.text(0., .16,
                        '† Different start-capture protocol: no turn-error subtraction.\n'
                        f'‡ Whole final-step maximum {excluded:.6f} m retained;\n'
                        '   post-exit sample excluded from turn-error comparison.',
                        transform=table_axis.transAxes, va='top', fontsize=8.5, linespacing=1.5)
                    handles = [Line2D([0], [0], color=COLORS[lane], linewidth=2., label=label)
                        for lane, label in zip(LANES, ('Centre d = 0', 'Inner d = −w', 'Outer d = +w'))]
                    figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .92), ncol=3, frameon=False)
                    figure.suptitle('Ideal 90° turn geometry and saved native responses',
                                       fontsize=16, fontweight='semibold', y=.982)
                    figure.text(.5, .945,
                        f'Derived constant-radius arcs, not fitted trajectories | half-width w = {width:g} m',
                        ha='center', fontsize=10.5)
                    figure.text(.06, .095,
                        'Black L = nominal rays; grey = Euclidean w-tube. Dotted segments join exact maximum witnesses to closest ray projections.\n'
                        'All three radii satisfy R − d ≤ min(L₁, L₂), with L₁ = L₂ = 4630 m. Geometry panels show local coordinates in metres.',
                        fontsize=8.7, va='top', linespacing=1.5)
                    figure.text(.06, .043,
                        'Bounds exclude initial 45° capture, acceleration, guidance error and exit overrun. They concern this circular-arc family only;\n'
                        'they do not establish controller optimality, arbitrary-policy infeasibility, continuous native safety or novelty.',
                        fontsize=8.7, va='top', linespacing=1.5, color='#7B3B00')
                    figure.subplots_adjust(left=.07, right=.96, top=.858, bottom=.19, wspace=.28, hspace=.38)
                    for name in OUTPUTS:
                        figure.savefig(output/name, dpi=300, bbox_inches='tight', pad_inches=.12)
                finally:
                    plt.close(figure)
            versions = dict(python=platform.python_version(), matplotlib=str(matplotlib.__version__),
                            backend=str(matplotlib.get_backend()))
        finally:
            if previous is None:
                os.environ.pop('MPLCONFIGDIR', None)
            else:
                os.environ['MPLCONFIGDIR'] = previous
    metadata = dict(schema='turn-geometry-plot-metadata.v1', input_schema=SCHEMA,
        input_path=str(input_path), input_sha256=input_sha,
        renderer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_records=document['source_records'], source_verification='Controller-provided hashes retained; referenced files are not reopened.',
        versions=versions, figure_size_inches=list(FIGURE_SIZE), png_dpi=300,
        smoothing=False, trajectory_fitting=False, native_simulation=False, uncertainty_estimates=False,
        half_width_m=width, leg_lengths_m=document['leg_lengths_m'], radius_cases=document['radius_cases'],
        definition=dict(a=A, nominal_rays='Gamma = {(x,0): x<=0} union {(0,y): y>=0}',
            arc='x=d-R+R*sin(theta), y=R-d-R*cos(theta), theta in [0,pi/2]',
            offsets='right positive; north turn centre=0, inner=-w, outer=+w',
            centre='a*R', inner='w+a*R',
            outer='sqrt(2)*(w-a*R) if R<w; w if w<=R<=2*w/a; a*R-w otherwise',
            finite_leg_condition='R-d<=min(L1,L2)',
            containment='For this ideal arc family: centre R<=w/a; outer w<=R<=2*w/a; inner never for R>0'),
        exact_containment_limits_m=dict(centre_upper=width/A, outer_lower=width, outer_upper=2.*width/A),
        normalized_curve=dict(radius_over_width=ratios, distance_over_width=normalized, zero_radius_is_limit_only=True),
        analytic_arc_samples_per_lane=513, geometry_panels=geometry_metadata,
        native_comparisons=comparison_rows,
        all_native_maxima_retained=True,
        comparison_flags={
            'same_midpoint_protocol': 'Native and ideal values are shown side by side; their difference is descriptive, not a fitted model error.',
            'different_start_capture_protocol': 'Original native maximum remains in the figure and metadata; no ideal turn-error subtraction.',
            'excluded_final_post_exit_sample': 'Excluded only from ideal turn-error interpretation; the original whole-step native maximum remains in the figure and metadata.'},
        comparison_scope='Native maxima retained without fitting; protocol-mismatched and post-exit rows excluded from turn-error subtraction.',
        excluded_physics=['initial 45-degree capture', 'acceleration', 'guidance errors', 'exit overrun'],
        interpretation='An independently derived fixed circular-arc model, not optimality or arbitrary-policy infeasibility, native safety or novelty evidence.',
        outputs=[dict(filename=name, bytes=(output/name).stat().st_size,
                      sha256=hashlib.sha256((output/name).read_bytes()).hexdigest()) for name in OUTPUTS])
    with (output/'plot_metadata.json').open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(metadata, indent=2, allow_nan=False)+'\n')
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Controller-prepared turn-geometry-plot.v1 JSON')
    args = parser.parse_args(argv)
    metadata = render(args.input, os.environ['LAB_RUN_DIR'])
    print(json.dumps(dict(input_sha256=metadata['input_sha256'],
                         outputs=[*OUTPUTS, 'plot_metadata.json']), allow_nan=False), flush=True)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
