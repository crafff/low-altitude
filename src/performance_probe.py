"""Small native envelope/lifecycle probe, executed only by the lab controller."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from paper_performance import install_performance, load_types


def run_probe(bs, types, output):
    """Step all 12 types through excessive climb/speed and descent/brake intent."""
    from bluesky.core import simtime
    from bluesky.core.entity import getproxied
    from bluesky.traffic.asas import ConflictDetection, ConflictResolution
    from bluesky.tools.aero import tas2cas, vtas2cas

    bs.sim.reset()
    performance = install_performance(bs, types)
    performance.select()
    ConflictResolution.setmethod('OFF')
    ConflictDetection.setmethod('OFF')
    bs.traf.wind.clear()
    bs.traf.setnoise(False)
    simtime.setdt(.25)
    names = list(types)
    for j, name in enumerate(names):
        speed = types[name]['nominal_tas_mps']
        created = bs.traf.cre(f'PERF{j:02}', name.upper(), 52. + .02*j,
                              4., 90., 1000., float(tas2cas(speed, 1000.)))
        if created is not True:
            raise RuntimeError(f'Native creation failed: {created}')
    implementation = getproxied(bs.traf.perf)
    vmax = np.array([types[name]['maximum_tas_mps'] for name in names])
    climb = np.array([types[name]['climb_mps'] for name in names])
    descent = np.array([types[name]['descent_mps'] for name in names])
    checks = {
        'all_twelve_created': len(names) == bs.traf.ntraf == 12,
        'concrete_performance_selected': type(implementation) is performance,
        'full_type_names_preserved': list(implementation.actype) == [n.upper() for n in names],
        'maximum_tas_array_matches': bool(np.allclose(implementation.vmax, vmax)),
        'climb_array_matches': bool(np.array_equal(implementation.vsmax, climb)),
        'descent_array_matches': bool(np.array_equal(implementation.vsmin, -descent)),
        'acceleration_array_matches': bool(np.all(implementation.axmax == 3.5)),
    }
    phases = {}
    bs.sim.op()
    for label, target_alt, speed_scale in [('climb_accelerate', 3000., 2.),
                                          ('descend_brake', 100., 0.)]:
        before = bs.traf.tas.copy()
        peak_accel = np.zeros(len(names))
        minimum_accel = np.zeros(len(names))
        envelope_ok = True
        # Positive VS for descent deliberately exercises native magnitude semantics.
        bs.traf.selvs[:] = 100.
        bs.traf.selalt[:] = target_alt
        bs.traf.swlnav[:] = False
        bs.traf.swvnav[:] = False
        bs.traf.swvnavspd[:] = False
        for _ in range(128):
            bs.traf.selspd[:] = vtas2cas(speed_scale * vmax, bs.traf.alt)
            bs.sim.step()
            accel = (bs.traf.tas - before) / .25
            peak_accel = np.maximum(peak_accel, accel)
            minimum_accel = np.minimum(minimum_accel, accel)
            envelope_ok = envelope_ok and bool(
                np.all(bs.traf.tas >= -1e-8) and np.all(bs.traf.tas <= vmax + 1e-6)
                and np.all(bs.traf.vs <= climb + 1e-8)
                and np.all(bs.traf.vs >= -descent - 1e-8)
                and np.all(np.abs(accel) <= 3.5 + 1e-6))
            before = bs.traf.tas.copy()
        expected_vs = climb if label == 'climb_accelerate' else -descent
        expected_tas = vmax if label == 'climb_accelerate' else np.zeros(len(names))
        checks[f'{label}_sampled_envelopes'] = envelope_ok
        checks[f'{label}_tas_reached'] = bool(np.allclose(bs.traf.tas, expected_tas, atol=1e-6))
        checks[f'{label}_vs_reached'] = bool(np.allclose(bs.traf.vs, expected_vs, atol=1e-6))
        checks[f'{label}_altitude_intent_preserved'] = bool(np.all(bs.traf.aporasas.alt == target_alt))
        observed = peak_accel if label == 'climb_accelerate' else minimum_accel
        checks[f'{label}_acceleration_reached'] = bool(np.allclose(observed, 3.5 if speed_scale else -3.5))
        phases[label] = {
            name: {'tas_mps': float(bs.traf.tas[i]), 'vs_mps': float(bs.traf.vs[i]),
                   'altitude_m': float(bs.traf.alt[i]),
                   'maximum_acceleration_mps2': float(peak_accel[i]),
                   'minimum_acceleration_mps2': float(minimum_accel[i])}
            for i, name in enumerate(names)
        }
    bs.traf.delete(3)
    checks['delete_arrays_aligned'] = all(len(getattr(implementation, key)) == 11
                                          for key in implementation._ArrVars)
    checks['delete_preserves_type_order'] = list(implementation.actype) == [
        name.upper() for j, name in enumerate(names) if j != 3]
    bs.sim.reset()
    checks['reset_arrays_empty'] = bs.traf.ntraf == 0 and all(
        len(getattr(implementation, key)) == 0 for key in implementation._ArrVars)
    performance.select()
    name = names[0]
    bs.traf.cre('RECREATE', name, 52., 4., 90., 1000.,
                float(tas2cas(types[name]['nominal_tas_mps'], 1000.)))
    checks['recreate_after_reset'] = (
        bs.traf.ntraf == 1 and list(implementation.actype) == [name.upper()]
        and bool(np.allclose(implementation.vmax, [types[name]['maximum_tas_mps']])))
    bs.sim.reset()
    performance.select()
    try:
        bs.traf.cre('UNKNOWN', 'NOT_IN_TABLE3', 52., 4., 90., 1000., 10.)
    except ValueError as exc:
        checks['unknown_type_rejected'] = 'Unknown Table 3 aircraft types' in str(exc)
    else:
        checks['unknown_type_rejected'] = False
    finally:
        bs.sim.reset()
    result = {
        'scope': 'Native Table 3 envelope and lifecycle diagnostic; no learned policy',
        'dt_seconds': .25, 'simulated_seconds': 64., 'phases': phases,
        'checks': checks, 'all_checks_passed': all(checks.values()),
        'limitations': 'Bank and vertical acceleration remain native; sampled endpoints only.',
    }
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'performance_probe.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


def main():
    from bluesky_diagnostic import initialise
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--types', default='configs/uav_types.json')
    args = parser.parse_args()
    bs = initialise(Path('/tmp/bluesky-performance-probe'))
    result = run_probe(bs, load_types(args.types), Path(os.environ['LAB_RUN_DIR']))
    print(json.dumps(result, allow_nan=False))
    return 0 if result['all_checks_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
