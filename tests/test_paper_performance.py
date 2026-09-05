"""Pure table and clipping regressions; native stepping belongs to the probe."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from paper_performance import KNOT_MPS, clip_intent, load_types


class PaperPerformanceTests(unittest.TestCase):
    def test_all_twelve_published_rows_and_units(self):
        expected = {
            'Amzn': (196, 8, 8), 'Cranfield': (85, 3, 3),
            'CranfieldV2': (39, 3, 3), 'M100': (43, 5, 4),
            'M200': (45, 5, 3), 'M600': (35, 5, 5),
            'Phan4': (39, 6, 4), 'Mavic': (35, 5, 3),
            'Eh216': (69, 11, 11), 'Horsefly': (43, 8, 8),
            'Mnet': (21, 5, 5), 'Tecnalia': (21, 3, 3),
        }
        table = load_types(Path(__file__).resolve().parents[1] / 'configs/uav_types.json')
        self.assertEqual(set(table), set(expected))
        for name, (knots, climb, descent) in expected.items():
            with self.subTest(name=name):
                self.assertAlmostEqual(table[name]['maximum_tas_mps'], knots * 1852 / 3600)
                self.assertAlmostEqual(table[name]['nominal_tas_mps'], knots * KNOT_MPS * .8)
                self.assertEqual(table[name]['climb_mps'], climb)
                self.assertEqual(table[name]['descent_mps'], descent)
                self.assertEqual(table[name]['acceleration_mps2'], 3.5)

    def test_vertical_limit_uses_altitude_direction_not_requested_sign(self):
        altitude = np.array([200., 0., 200., 0., 100.])
        v, vs, h = clip_intent(
            np.array([99., 99., -5., 5., 4.]),
            np.array([-99., 99., 2., -2., 99.]), altitude,
            np.full(5, 100.), np.full(5, 10.), np.full(5, 5.), np.full(5, 3.))
        np.testing.assert_array_equal(v, [10., 10., 0., 5., 4.])
        np.testing.assert_array_equal(vs, [5., -3., 2., -2., 0.])
        self.assertIs(h, altitude)

    def test_invalid_limits_and_duplicate_names_rejected(self):
        source = {
            'nominal_speed_fraction': .8, 'acceleration_mps2': 3.5,
            'types': {'M100': {'maximum_speed_kt': 43, 'climb_mps': 5, 'descent_mps': 4}},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'types.json'
            for invalid in (-1, 0, float('nan'), float('inf'), True, '4'):
                with self.subTest(value=invalid):
                    source['types']['M100']['descent_mps'] = invalid
                    path.write_text(json.dumps(source))
                    with self.assertRaises(ValueError):
                        load_types(path)
            source['types']['M100']['descent_mps'] = 4
            source['types']['m100'] = dict(source['types']['M100'])
            path.write_text(json.dumps(source))
            with self.assertRaisesRegex(ValueError, 'duplicate'):
                load_types(path)
            path.write_text('{"types": {}, "types": {}}')
            with self.assertRaisesRegex(ValueError, 'Duplicate JSON key'):
                load_types(path)
