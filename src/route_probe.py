"""Single-flight native route trace; run only inside the controller's lab job.

The default reproduces the saved route/type with admission shifted to t=0.
An explicit --bank-deg is a diagnostic variant, not a baseline correction.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import tempfile

from bluesky_diagnostic import FT, distance_m, initialise
from nr_pilot import arrived_on_route, centerline_distance_m
from paper_performance import install_performance, load_types


def angle_difference(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg['schema'] != 'bluesky.route-probe.v1' or cfg['stage'] != 'dev':
        raise ValueError('Expected a development route-probe configuration')
    if cfg['dt_seconds'] != 0.25 or not 0 < cfg['timeout_seconds'] <= 1200:
        raise ValueError('Trace requires 0.25-second steps and a bounded timeout')
    if not math.isclose(cfg['timeout_seconds'] / cfg['dt_seconds'],
                        round(cfg['timeout_seconds'] / cfg['dt_seconds'])):
        raise ValueError('Timeout must lie on a physics tick')
    for field in ('altitude_ft', 'arrival_radius_m', 'corridor_width_ft'):
        if not math.isfinite(cfg[field]) or cfg[field] <= 0:
            raise ValueError(f'Invalid {field}')
    points = cfg['corridor']['waypoints_lat_lon_deg']
    if not 2 <= len(points) <= 5:
        raise ValueError('Expected two through five saved route points')
    for point in points:
        if (len(point) != 2 or not all(math.isfinite(v) for v in point)
                or not -90 < point[0] < 90 or not -180 <= point[1] <= 180):
            raise ValueError('Invalid saved latitude/longitude')
    if any(a == b for a, b in zip(points, points[1:])):
        raise ValueError('Consecutive route points must differ')
    if cfg['flight']['corridor_id'] != cfg['corridor']['id']:
        raise ValueError('Flight and corridor IDs differ')
    return cfg


def sample(bs, cfg, index, previous=None):
    """Observe after native motion; command fields are those used in that step."""
    from bluesky.tools import geo
    from bluesky.tools.aero import g0

    acid = cfg['flight']['id']
    i = bs.traf.id2idx(acid)
    if i < 0 or bs.traf.ntraf != 1:
        raise RuntimeError('Unexpected native disappearance or extra aircraft')
    traf, route = bs.traf, bs.traf.ap.route[i]
    points = cfg['corridor']['waypoints_lat_lon_deg']
    active = int(route.iactwp)
    if not 0 <= active < len(route.wpname):
        raise RuntimeError('Invalid native active waypoint index')
    state = (float(traf.lat[i]), float(traf.lon[i]))
    deviation = centerline_distance_m(state, points)
    endpoint = distance_m(state, points[-1])
    bankdef, turnphi = float(traf.ap.bankdef[i]), float(traf.ap.turnphi[i])
    bank = turnphi if turnphi > float(traf.eps[i]) ** 2 else bankdef
    row = {
        'row_index': index,
        'sim_time_s': float(bs.sim.simt),
        'original_scenario_time_s': float(bs.sim.simt) + cfg['source']['actual_entry_s'],
        'aircraft': acid,
        'lat_deg': state[0], 'lon_deg': state[1],
        'alt_m': float(traf.alt[i]), 'tas_mps': float(traf.tas[i]),
        'actual_hdg_deg': float(traf.hdg[i]), 'actual_track_deg': float(traf.trk[i]),
        'target_hdg_deg': float(traf.aporasas.hdg[i]) % 360.0,
        'target_track_deg': float(traf.ap.trk[i]) % 360.0,
        'bank_default_deg': math.degrees(bankdef),
        'bank_turnphi_deg': math.degrees(turnphi),
        'bank_limit_deg': math.degrees(bank),
        'observed_heading_rate_deg_s': None,
        'equivalent_kinematic_bank_deg': None,
        'heading_rate_limited': bool(traf.swhdgsel[i]),
        'active_waypoint_index': active,
        'nominal_point_index': active + 1,
        'active_waypoint_name': route.wpname[active],
        'active_waypoint_lat_deg': float(traf.actwp.lat[i]),
        'active_waypoint_lon_deg': float(traf.actwp.lon[i]),
        'active_waypoint_distance_m': float(geo.qdrdist(
            *state, float(traf.actwp.lat[i]), float(traf.actwp.lon[i]))[1]) * 1852.0,
        'guidance_distance_before_motion_m': float(traf.ap.dist2wp[i]),
        'turn_initiation_distance_m': float(traf.actwp.turndist[i]),
        'native_turn_radius_m': float(traf.actwp.turnrad[i]),
        'active_leg_direction_deg': float(traf.actwp.curlegdir[i]),
        'flyby': bool(traf.actwp.flyby[i]), 'flyturn': bool(traf.actwp.flyturn[i]),
        'lnav': bool(traf.swlnav[i]), 'vnav': bool(traf.swvnav[i]),
        'vnav_speed': bool(traf.swvnavspd[i]),
        'centerline_distance_m': deviation,
        'outside_corridor': deviation > cfg['corridor_width_ft'] * FT / 2,
        'endpoint_distance_m': endpoint,
        'final_waypoint_active': active == len(route.wpname) - 1,
        'arrival_guard': bool(arrived_on_route(route, endpoint, cfg['arrival_radius_m'])),
    }
    if previous is not None:
        dt = row['sim_time_s'] - previous['sim_time_s']
        if not math.isclose(dt, cfg['dt_seconds'], abs_tol=1e-8):
            raise RuntimeError(f'Unexpected native physics interval: {dt}')
        rate = angle_difference(row['actual_hdg_deg'], previous['actual_hdg_deg']) / dt
        row['observed_heading_rate_deg_s'] = rate
        # Native Traffic has a bank-limited heading update, not a roll state.
        row['equivalent_kinematic_bank_deg'] = math.degrees(
            math.atan(math.radians(rate) * row['tas_mps'] / g0))
    if any(isinstance(v, float) and not math.isfinite(v) for v in row.values()):
        raise RuntimeError('Non-finite native trace state')
    return row


def summarize(rows, cfg):
    """Preserve sampled crossings; match NR's right-endpoint width integral."""
    crossings, switches, episodes = [], [], []
    active = None
    outside_seconds = 0.0
    for previous, row in zip(rows, rows[1:]):
        dt = row['sim_time_s'] - previous['sim_time_s']
        if row['outside_corridor'] != previous['outside_corridor']:
            crossings.append({
                'direction': 'exit_corridor' if row['outside_corridor'] else 'enter_corridor',
                'crossing_time_bracket_s': [previous['sim_time_s'], row['sim_time_s']],
                'before_row': previous, 'after_row': row,
            })
        if row['outside_corridor']:
            outside_seconds += dt
            if active is None:
                active = {'first_outside_sample_s': row['sim_time_s'],
                          'integration_start_s': previous['sim_time_s'],
                          'sampled_width_seconds': 0.0, 'peak_row': row}
            active['sampled_width_seconds'] += dt
            active['last_outside_sample_s'] = row['sim_time_s']
            if row['centerline_distance_m'] > active['peak_row']['centerline_distance_m']:
                active['peak_row'] = row
        elif active is not None:
            episodes.append(active | {'first_inside_sample_s': row['sim_time_s'],
                                      'end_reason': 'sampled_return'})
            active = None
        if row['active_waypoint_index'] != previous['active_waypoint_index']:
            switches.append({'switch_time_bracket_s': [previous['sim_time_s'], row['sim_time_s']],
                             'before_row': previous, 'after_row': row})
    if active is not None:
        episodes.append(active | {'first_inside_sample_s': None, 'end_reason': 'flight_terminal'})
    peak = max(rows, key=lambda row: row['centerline_distance_m'])
    return {
        'peak_centerline_distance_m': peak['centerline_distance_m'], 'peak_row': peak,
        'outside_corridor_seconds': outside_seconds,
        'corridor_half_width_m': cfg['corridor_width_ft'] * FT / 2,
        'width_exposure_rule': 'Sum dt for each post-step sample strictly beyond half-width, matching nr_pilot.',
        'crossing_rule': 'Adjacent sampled-state brackets; no sub-tick interpolation or containment guarantee.',
        'crossings': crossings, 'outside_episodes': episodes, 'waypoint_switches': switches,
        'row_count': len(rows), 'steps': len(rows) - 1,
        'terminal_row': rows[-1],
        'status': 'arrived' if rows[-1]['arrival_guard'] else 'flight_timeout',
    }


def run_probe(bs, cfg, types, output, bank_deg=None):
    from bluesky.core import simtime
    from bluesky.core.entity import getproxied
    from bluesky.traffic.asas import ConflictDetection, ConflictResolution
    from bluesky.tools import geo
    from bluesky.tools.aero import tas2cas

    bs.sim.reset()
    performance = install_performance(bs, types)
    performance.select()
    if type(getproxied(bs.traf.perf)) is not performance:
        raise RuntimeError('PaperPerformance is not the active native performance class')
    ConflictResolution.setmethod('OFF')
    ConflictDetection.setmethod('OFF')
    bs.traf.wind.clear()
    bs.traf.setnoise(False)
    simtime.setdt(cfg['dt_seconds'])
    acid, kind = cfg['flight']['id'], cfg['flight']['type']
    points = cfg['corridor']['waypoints_lat_lon_deg']
    altitude = cfg['altitude_ft'] * FT
    tas = types[kind]['nominal_tas_mps']
    heading = float(geo.qdrdist(*points[0], *points[1])[0])
    if bs.traf.cre(acid, kind, *points[0], heading, altitude,
                   float(tas2cas(tas, altitude))) is not True:
        raise RuntimeError('Native aircraft creation failed')
    i = bs.traf.id2idx(acid)
    native_bank = math.degrees(float(bs.traf.ap.bankdef[i]))
    if bank_deg is not None:
        bs.traf.ap.bankdef[i] = math.radians(bank_deg)
    route = bs.traf.ap.route[i]
    for j, (lat, lon) in enumerate(points[1:]):
        if route.addwpt(i, f'{acid}P{j}', route.wplatlon, lat, lon, altitude, -999.) < 0:
            raise RuntimeError('Native waypoint creation failed')
    if route.direct(i, route.wpname[0]) is not True:
        raise RuntimeError('Native direct-to failed')
    bs.traf.swlnav[i], bs.traf.swvnav[i], bs.traf.swvnavspd[i] = True, False, False
    native_settings = {
        'performance_class': type(getproxied(bs.traf.perf)).__name__,
        'performance_envelope': types[kind], 'physics_dt_s': cfg['dt_seconds'],
        'native_bank_default_deg': native_bank,
        'selected_bank_default_deg': math.degrees(float(bs.traf.ap.bankdef[i])),
        'waypoint_flyby': [bool(v) for v in route.wpflyby],
        'waypoint_flyturn': [bool(v) for v in route.wpflyturn],
        'wind_dimensions': int(bs.traf.wind.winddim),
        'conflict_resolution_class': ConflictResolution.selected().__name__,
        'conflict_detection_class': ConflictDetection.selected().__name__,
        'noise_requested': False, 'vnav_requested': False, 'vnav_speed_requested': False,
        'timers': {timer.name: {'requested_s': float(timer.dt_requested),
                               'actual_s': float(timer.dt_act)} for timer in simtime.Timer.timers()},
    }
    rows = [sample(bs, cfg, 0)]
    bs.sim.op()
    with (output / 'trajectory.csv').open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerow(rows[0])
        for step in range(1, round(cfg['timeout_seconds'] / cfg['dt_seconds']) + 1):
            bs.sim.step()
            row = sample(bs, cfg, step, rows[-1])
            rows.append(row)
            writer.writerow(row)
            if row['arrival_guard']:
                break
    result = summarize(rows, cfg)
    result.update({
        'scope': 'Development-only isolated replay of one saved flight, starting at t=0; no policy.',
        'variant': 'native_baseline' if bank_deg is None else 'diagnostic_bank_override',
        'bank_override_deg': bank_deg, 'native_settings': native_settings, 'input': cfg,
        'versions': {name: version(name) for name in ('bluesky-simulator', 'numpy')},
        'trace_semantics': {
            'row_zero': 'Creation/direct-to state, before first native guidance/motion update.',
            'later_rows': 'After each physics step; guidance and targets were computed before that motion.',
            'bank': 'bank_limit_deg selects turnphi or bankdef exactly as native Traffic; it is not a measured roll state.',
            'equivalent_bank': 'Signed diagnostic atan(TAS * observed heading rate / g0), not an additional dynamics state.',
            'waypoint_indices': 'Native route excludes the origin: native index 0 is saved nominal point 1.',
            'turn_distance': 'Native actwp.turndist belongs to the current active waypoint; switch records retain both sides.',
            'original_time': 'Replay age plus saved actual admission time; timer phases may differ from population execution.',
            'centerline_distance': 'Same local equirectangular distance to the full nominal polyline as nr_pilot.',
        },
    })
    # Retain final state and guard before native deletion, exactly as NR does.
    bs.traf.delete(bs.traf.id2idx(acid))
    with (output / 'result.json').open('x') as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--bank-deg', type=float,
                        help='Explicit diagnostic override of native bankdef; not the baseline.')
    args = parser.parse_args()
    if args.bank_deg is not None and not (math.isfinite(args.bank_deg) and 0 < args.bank_deg <= 60):
        parser.error('--bank-deg must be finite and in (0, 60]')
    cfg = read_config(args.config)
    if version('bluesky-simulator') != cfg['bluesky_version'] or cfg['bluesky_version'] != '1.1.1':
        raise ValueError('This probe requires BlueSky 1.1.1')
    types_path = Path(cfg['types_path'])
    if hashlib.sha256(types_path.read_bytes()).hexdigest() != cfg['types_sha256']:
        raise ValueError('Performance table differs from the saved diagnostic input')
    types = load_types(types_path)
    output = Path(os.environ['LAB_RUN_DIR']).resolve(strict=True)
    if not output.is_dir():
        raise ValueError('LAB_RUN_DIR must be an existing output directory')
    bs = initialise(Path(tempfile.mkdtemp(prefix='bluesky-route-probe-', dir='/tmp')))
    result = run_probe(bs, cfg, types, output, args.bank_deg)
    print(json.dumps({key: result[key] for key in (
        'variant', 'status', 'steps', 'peak_centerline_distance_m', 'outside_corridor_seconds')},
        allow_nan=False), flush=True)
    return 0 if result['status'] == 'arrived' else 1


if __name__ == '__main__':
    raise SystemExit(main())
