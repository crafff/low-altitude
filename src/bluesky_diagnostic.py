"""Bounded native BlueSky route diagnostics; no policy or surrogate dynamics."""
from __future__ import annotations

import argparse
import csv
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import random
import resource
import time

FT = 0.3048
EARTH_M = 6371000.0


def distance_m(a, b):
    lat1, lat2 = math.radians(a[0]), math.radians(b[0])
    dlat, dlon = lat2 - lat1, math.radians(b[1] - a[1])
    h = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    return 2 * EARTH_M * math.asin(math.sqrt(min(1.0, max(0.0, h))))


def risk_flags(horizontal_m, vertical_m, cfg):
    ratio = abs(vertical_m) / (cfg['vertical_tolerance_ft'] * FT)
    if ratio >= 1:
        return False, False
    factor = math.sqrt(1 - ratio)
    return tuple(horizontal_m < cfg[key] * FT * factor
                 for key in ('lowc_horizontal_ft', 'nmac_horizontal_ft'))


def initialise(workdir):
    os.environ['MPLCONFIGDIR'] = str(workdir / 'matplotlib')
    os.environ['MPLBACKEND'] = 'Agg'
    workdir.mkdir(parents=True, exist_ok=True)
    # Preserve mandatory upstream paths; disable external-data plugins locally.
    config = workdir / 'diagnostic.cfg'
    import bluesky as bs
    template = Path(bs.__file__).parent / 'resources' / 'default.cfg'
    config.write_text(template.read_text() + '\nenabled_plugins = []\n')
    bs.init(mode='sim', detached=True, workdir=workdir, configfile=str(config))
    from bluesky.traffic.performance.perfbase import PerfBase

    class DiagnosticPerformance(PerfBase):
        """Unmodified native kinematics with a concrete singleton constructor."""

    bs._diagnostic_performance = DiagnosticPerformance
    return bs


def run_case(bs, cfg, kind, output):
    from bluesky.core import simtime
    from bluesky.core.entity import getproxied
    from bluesky.traffic.asas import ConflictDetection, ConflictResolution
    from bluesky.tools import geo
    from bluesky.tools.aero import tas2cas

    bs.sim.reset()
    # The base constructor dispatches to the selected generator and returns a
    # proxy; a named derived constructor returns the concrete implementation.
    bs._diagnostic_performance.select()
    assert getproxied(bs.traf.perf) is bs._diagnostic_performance.implinstance()
    assert getproxied(bs.traf.perf) is not bs.traf.perf
    ConflictResolution.setmethod('OFF')
    ConflictDetection.setmethod('OFF')
    bs.traf.wind.clear()
    bs.traf.setnoise(False)
    simtime.setdt(cfg['dt_seconds'])
    rotation = random.Random(cfg['seed']).uniform(0, 360)

    def point(x, y):
        bearing = math.degrees(math.atan2(x, y)) + rotation
        p = geo.qdrpos(*cfg['origin_lat_lon_deg'], bearing, math.hypot(x, y) / 1852)
        return tuple(float(v) for v in p)

    radius = cfg['half_route_m']
    routes = [('A1', (-radius, 0), [(radius, 0)], 0)]
    if kind == 'single_turn':
        routes = [('A1', (-radius, 0), [(0, 0), (0, radius)], 0)]
    else:
        offset = cfg['vertical_control_offset_ft'] if kind == 'vertical_control' else 0
        routes.append(('A2', (0, -radius), [(0, radius)], offset))
    destinations, nominal_alt = {}, {}
    for acid, start_xy, waypoints_xy, offset in routes:
        start = point(*start_xy)
        points = [point(*p) for p in waypoints_xy]
        altitude = (cfg['altitude_ft'] + offset) * FT
        heading = float(geo.qdrdist(*start, *points[0])[0])
        result = bs.traf.cre(acid, 'B744', *start, heading, altitude,
                             float(tas2cas(cfg['initial_tas_mps'], altitude)))
        if result is not True:
            raise RuntimeError(f'Aircraft creation failed: {result}')
        i = bs.traf.id2idx(acid)
        route = bs.traf.ap.route[i]
        for j, (lat, lon) in enumerate(points):
            assert route.addwpt(i, f'{acid}P{j}', route.wplatlon, lat, lon, altitude, -999.) >= 0
        route.direct(i, route.wpname[0])
        bs.traf.swlnav[i] = True
        bs.traf.swvnav[i] = False
        bs.traf.swvnavspd[i] = False
        destinations[acid], nominal_alt[acid] = points[-1], altitude

    def states():
        return {acid: (float(bs.traf.lat[i]), float(bs.traf.lon[i]),
                       float(bs.traf.alt[i]), float(bs.traf.tas[i]), int(bs.traf.ap.route[i].iactwp))
                for i, acid in enumerate(bs.traf.id)}

    output.mkdir()
    trace = output / 'trajectory.csv'
    lowc = nmac = flight_seconds = 0.0
    min_horizontal = None
    max_alt_error = max_speed_error = 0.0
    path_m = dict.fromkeys(destinations, 0.0)
    arrivals = {}
    reason = 'horizon'
    wall_start = time.perf_counter()
    steps = 0
    bs.sim.op()
    with trace.open('x', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sim_time_s', 'aircraft', 'lat_deg', 'lon_deg', 'alt_m', 'tas_mps', 'active_waypoint'])
        before = states()
        for acid, values in before.items():
            writer.writerow([0.0, acid, *values])
        while bs.traf.ntraf and bs.sim.simt < cfg['horizon_seconds']:
            t0 = bs.sim.simt
            pairs = []
            ids = list(before)
            for i, aid in enumerate(ids):
                for bid in ids[i+1:]:
                    a, b = before[aid], before[bid]
                    horizontal = distance_m(a, b)
                    min_horizontal = horizontal if min_horizontal is None else min(min_horizontal, horizontal)
                    pairs.append(risk_flags(horizontal, a[2] - b[2], cfg))
            bs.sim.step()
            dt = bs.sim.simt - t0
            assert math.isclose(dt, cfg['dt_seconds'], abs_tol=1e-9), dt
            flight_seconds += len(before) * dt
            lowc += sum(p[0] for p in pairs) * dt
            nmac += sum(p[1] for p in pairs) * dt
            after = states()
            if set(after) != set(before):
                raise RuntimeError('Unexpected simulator deletion; cannot treat missing traffic as safe')
            for acid, values in after.items():
                assert all(math.isfinite(v) for v in values), (acid, values)
                writer.writerow([bs.sim.simt, acid, *values])
                path_m[acid] += distance_m(before[acid], values)
                max_alt_error = max(max_alt_error, abs(values[2] - nominal_alt[acid]))
                max_speed_error = max(max_speed_error, abs(values[3] - cfg['initial_tas_mps']))
                remaining = distance_m(values, destinations[acid])
                if remaining <= cfg['arrival_radius_m']:
                    arrivals[acid] = {'time_s': bs.sim.simt, 'endpoint_distance_m': remaining}
            # Record the terminal sample/exposure before removing arrivals; indices shift.
            for acid in list(after):
                if acid in arrivals:
                    bs.traf.delete(bs.traf.id2idx(acid))
            before = states()
            steps += 1
        if not bs.traf.ntraf:
            reason = 'all_arrived'
    elapsed = time.perf_counter() - wall_start
    flight_hours = flight_seconds / 3600
    expected_conflict = kind == 'crossing'
    checks = {'all_arrived': len(arrivals) == len(routes),
              'altitude_held': max_alt_error < 0.01,
              'speed_held': max_speed_error < 0.05,
              'expected_exposure': nmac > 0 if expected_conflict else lowc == 0,
              'native_performance_selected': type(getproxied(bs.traf.perf)) is bs._diagnostic_performance,
              'resolution_off': ConflictResolution.selected() is ConflictResolution,
              'wind_off': bs.traf.wind.winddim == 0}
    result = {'kind': kind, 'seed': cfg['seed'], 'rotation_deg': rotation,
              'planned': len(routes), 'completed': len(arrivals),
              'failed_or_unfinished': len(routes) - len(arrivals), 'termination': reason,
              'arrivals': arrivals, 'path_length_m': path_m,
              'flight_seconds': flight_seconds, 'flight_hours': flight_hours,
              'lowc_unordered_pair_seconds': lowc, 'nmac_unordered_pair_seconds': nmac,
              'lowc_directed_pair_seconds': 2*lowc, 'nmac_directed_pair_seconds': 2*nmac,
              'lowc_unordered_seconds_per_flight_hour': lowc / flight_hours,
              'nmac_unordered_seconds_per_flight_hour': nmac / flight_hours,
              'min_sampled_horizontal_m': min_horizontal,
              'max_altitude_error_m': max_alt_error, 'max_tas_error_mps': max_speed_error,
              'sim_seconds': bs.sim.simt, 'steps': steps, 'wall_seconds': elapsed,
              'steps_per_wall_second': steps/elapsed,
              'process_peak_rss_mib': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
              'control_commands': 0, 'checks': checks,
              'trajectory_sha256': hashlib.sha256(trace.read_bytes()).hexdigest()}
    (output/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    assert 0 < cfg['dt_seconds'] <= 1 and 0 < cfg['horizon_seconds'] <= 600
    assert version('bluesky-simulator') == cfg['bluesky_version']
    output = Path(os.environ['LAB_RUN_DIR'])
    start = time.perf_counter()
    bs = initialise(Path('/tmp/bluesky-diagnostic'))
    init_seconds = time.perf_counter() - start
    cases = {}
    for label, kind in [('single_turn', 'single_turn'), ('crossing', 'crossing'),
                        ('vertical_control', 'vertical_control'), ('crossing_repeat', 'crossing')]:
        cases[label] = run_case(bs, cfg, kind, output/label)
    repeat = cases['crossing']['trajectory_sha256'] == cases['crossing_repeat']['trajectory_sha256']
    summary = {'scope': 'native BlueSky diagnostic kinematics, not effective MARL baseline',
               'versions': {p:version(p) for p in ('bluesky-simulator','numpy','openap')},
               'config':cfg, 'initialization_wall_seconds':init_seconds,
               'cases':cases, 'crossing_repeats_exactly':repeat,
               'all_checks_passed':repeat and all(all(c['checks'].values()) for c in cases.values())}
    (output/'result.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary))
    return 0 if summary['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
