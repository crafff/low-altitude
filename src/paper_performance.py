"""Table 3 envelopes applied through native BlueSky 1.1.1 kinematics.

Importing this module does not import BlueSky: its settings must be initialized
before importing PerfBase. Call install_performance after bs.init, then select
the returned concrete class after every bs.sim.reset and before creating traffic.

This reconstructs TAS, climb/descent and horizontal acceleration envelopes only.
Minimum TAS is zero. Native bank, vertical acceleration and altitude capture are
unchanged; no additional altitude envelope or speed/vertical coupling is inferred.
Unknown aircraft types fail the job; reset after a failed native creation because
Traffic.cre has already appended parent arrays before invoking performance.create.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

KNOT_MPS = 1852.0 / 3600.0


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{label} must be a finite positive number')
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{label} must be a finite positive number')
    return float(value)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def load_types(path):
    """Load Table 3 records, preserving published IDs and returning SI values."""
    document = json.loads(Path(path).read_text(), object_pairs_hook=_unique_object)
    fraction = _positive(document['nominal_speed_fraction'], 'nominal_speed_fraction')
    acceleration = _positive(document['acceleration_mps2'], 'acceleration_mps2')
    if fraction != 0.8 or acceleration != 3.5:
        raise ValueError('Table 3 requires nominal fraction 0.8 and acceleration 3.5')
    rows = document['types']
    if not isinstance(rows, dict) or not rows:
        raise ValueError('types must be a nonempty object')
    result, seen = {}, set()
    for name, row in rows.items():
        if not name or name.strip() != name or name.upper() in seen:
            raise ValueError(f'Invalid or case-insensitive duplicate aircraft type: {name!r}')
        seen.add(name.upper())
        maximum = _positive(row['maximum_speed_kt'], f'{name}.maximum_speed_kt') * KNOT_MPS
        result[name] = {
            'maximum_tas_mps': maximum,
            'nominal_tas_mps': fraction * maximum,
            'climb_mps': _positive(row['climb_mps'], f'{name}.climb_mps'),
            'descent_mps': _positive(row['descent_mps'], f'{name}.descent_mps'),
            'acceleration_mps2': acceleration,
        }
    return result


def clip_intent(intent_v, intent_vs, intent_h, current_h, vmax, climb, descent):
    """Clip native TAS/VS intent; preserve the target altitude object unchanged.

    Traffic.update_airspeed uses sign(target altitude - current altitude) and
    abs(requested VS), so VS sign alone cannot select an asymmetric limit.
    At the target altitude the commanded VS is zero.
    """
    direction = np.sign(np.asarray(intent_h) - np.asarray(current_h))
    cap = np.where(direction > 0, climb, descent)
    vs = direction * np.minimum(np.abs(intent_vs), cap)
    return np.clip(intent_v, 0.0, vmax), vs, intent_h


def install_performance(bs, types):
    """Return an unselected named PerfBase subclass bound to this table.

    Requires initialized BlueSky and empty traffic. Repeated calls with an
    identical table reuse the class; changing its table in-process is rejected.
    """
    if getattr(bs, 'traf', None) is None or bs.traf.ntraf:
        raise RuntimeError('Initialize BlueSky and reset traffic before installing performance')
    table = {name.upper(): dict(row) for name, row in types.items()}
    if not table or len(table) != len(types):
        raise ValueError('Performance table must have unique case-insensitive type names')
    for name, row in table.items():
        for field in ('maximum_tas_mps', 'nominal_tas_mps', 'climb_mps',
                      'descent_mps', 'acceleration_mps2'):
            _positive(row[field], f'{name}.{field}')
        if row['nominal_tas_mps'] > row['maximum_tas_mps']:
            raise ValueError(f'{name}: nominal TAS exceeds maximum TAS')
    existing = getattr(bs, '_paper_performance', None)
    if existing is not None:
        if bs._paper_performance_table != table:
            raise ValueError('A different paper performance table is already installed')
        return existing

    # Import only after bs.init has loaded the caller's configuration.
    from bluesky.traffic.performance.perfbase import PerfBase

    class PaperPerformance(PerfBase):
        """Static paper envelopes; native Traffic applies all motion updates."""

        def create(self, n=1):
            if n == 0:
                return
            names = [str(name).upper() for name in bs.traf.type[-n:]]
            missing = sorted(set(names) - table.keys())
            if missing:
                raise ValueError(f'Unknown Table 3 aircraft types: {missing}')
            records = [table[name] for name in names]
            super().create(n)
            # PerfBase initially uses a narrow Unicode dtype; replace the array
            # to preserve full type names while retaining array registration.
            self.actype = np.asarray(list(self.actype[:-n]) + names, dtype=str)
            self.vmin[-n:] = 0.0
            self.vmax[-n:] = [row['maximum_tas_mps'] for row in records]
            self.vsmax[-n:] = [row['climb_mps'] for row in records]
            self.vsmin[-n:] = [-row['descent_mps'] for row in records]
            self.axmax[-n:] = [row['acceleration_mps2'] for row in records]

        def limits(self, intent_v, intent_vs, intent_h, ax):
            return clip_intent(intent_v, intent_vs, intent_h, bs.traf.alt,
                               self.vmax, self.vsmax, -self.vsmin)

    bs._paper_performance = PaperPerformance
    bs._paper_performance_table = table
    return PaperPerformance
