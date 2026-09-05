"""Pure formula and fake-facade regressions; execute only through lab.py."""
import math
from types import SimpleNamespace
import unittest

import numpy as np

from navigation_sensitivity import refresh_reached, refreshed_flyby_distance, select_variants


class FakeActive:
    def __init__(self):
        self.next_qdr = np.array([0.0, -999.0])
        self.turndist = np.array([1.0, 123.0])
        self.flyby = np.array([True, False])
        self.flyturn = np.array([False, False])
        self.turnfromlastwp = np.array([False, False])
        self.turntonextwp = np.array([False, False])
        self.calls = 0
        self.fail = False
        self.return_value = np.array([0])

    def reached(self, qdr, dist):
        self.calls += 1
        self.received_arguments = (qdr, dist)
        self.received_cache = self.turndist.copy()
        if self.fail:
            raise RuntimeError("native error")
        self.turndist = np.logical_or(self.flyby, self.flyturn) * self.turndist
        return self.return_value


class CachedMethodProxy:
    """Matches the relevant cached-method / delegated-setattr BlueSky behavior."""
    def __init__(self, target):
        self.__dict__["_target"] = target
        self.__dict__["reached"] = target.reached

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __setattr__(self, name, value):
        setattr(self._target, name, value)


def fake_bs():
    active = FakeActive()
    proxy = CachedMethodProxy(active)
    traf = SimpleNamespace(id=["A", "B"], actwp=proxy, tas=np.array([80.0, 20.0]),
                           ap=SimpleNamespace(bankdef=np.radians([25.0, 25.0])))
    return SimpleNamespace(traf=traf, sim=SimpleNamespace(simt=1.0)), active


class NavigationSensitivityTests(unittest.TestCase):
    def test_wrapped_angle_terminal_sentinel_and_no_invented_four_radius_cap(self):
        distance, radius, following = refreshed_flyby_distance(
            [44, 44, 44], math.radians(25), [350, 0, 0], [10, -999, 170], 9.80665)
        expected_radius = 44**2 / (9.80665 * math.tan(math.radians(25)))
        np.testing.assert_allclose(radius, [expected_radius] * 3)
        np.testing.assert_allclose(distance, [expected_radius * math.tan(math.radians(10)),
                                             0, expected_radius * math.tan(math.radians(85))])
        self.assertGreater(distance[2], 4 * radius[2])
        np.testing.assert_array_equal(following, [10, 0, 170])

    def test_current_tas_and_bearing_recompute_each_call(self):
        bs, active = fake_bs()
        audit = {}
        with refresh_reached(bs, 9.80665, audit):
            bs.traf.actwp.reached(np.array([90.0, 0.0]), np.array([1000.0, 500.0]))
            first = active.received_cache[0]
            bs.traf.tas[0] = 160
            bs.traf.actwp.reached(np.array([45.0, 0.0]), np.array([900.0, 400.0]))
            self.assertAlmostEqual(active.received_cache[0], 4 * first * math.tan(math.radians(22.5)))
        self.assertEqual(active.calls, 2)
        self.assertEqual(audit["native_calls"], audit["wrapper_calls"])
        self.assertTrue(audit["method_restored"])

    def test_proxy_wrapper_calls_original_once_preserves_result_and_only_refreshes_flyby(self):
        bs, active = fake_bs()
        original = bs.traf.actwp.reached
        qdr, dist = np.array([90.0, 0.0]), np.array([1000.0, 500.0])
        before_tas, before_bank = bs.traf.tas.copy(), bs.traf.ap.bankdef.copy()
        audit = {}
        with refresh_reached(bs, 9.80665, audit):
            result = bs.traf.actwp.reached(qdr, dist)
            self.assertIs(result, active.return_value)
            self.assertIs(active.received_arguments[0], qdr)
            self.assertIs(active.received_arguments[1], dist)
            self.assertGreater(active.received_cache[0], 1)
            self.assertEqual(active.received_cache[1], 123)  # Native reached owns flyover zeroing.
            self.assertEqual(active.turndist[1], 0)
        self.assertIs(bs.traf.actwp.reached, original)
        self.assertEqual(active.calls, 1)
        self.assertEqual(audit["wrapper_calls"], 1)
        self.assertEqual(audit["native_calls"], 1)
        np.testing.assert_array_equal(bs.traf.tas, before_tas)
        np.testing.assert_array_equal(bs.traf.ap.bankdef, before_bank)

    def test_original_method_restored_when_native_call_raises(self):
        bs, active = fake_bs()
        original = bs.traf.actwp.reached
        active.fail = True
        audit = {}
        with self.assertRaisesRegex(RuntimeError, "native error"):
            with refresh_reached(bs, 9.80665, audit):
                bs.traf.actwp.reached(np.array([90.0, 0.0]), np.array([1000.0, 500.0]))
        self.assertIs(bs.traf.actwp.reached, original)
        self.assertEqual(active.calls, 1)
        self.assertTrue(audit["method_restored"])

    def test_flyturn_rejected_before_cache_mutation_or_native_call(self):
        for flag in ("flyturn", "turnfromlastwp", "turntonextwp"):
            with self.subTest(flag=flag):
                bs, active = fake_bs()
                getattr(active, flag)[0] = True
                before = active.turndist.copy()
                audit = {}
                with self.assertRaisesRegex(RuntimeError, "unsupported"):
                    with refresh_reached(bs, 9.80665, audit):
                        bs.traf.actwp.reached(np.array([90.0, 0.0]), np.array([1000.0, 500.0]))
                np.testing.assert_array_equal(active.turndist, before)
                self.assertEqual(active.calls, 0)
                self.assertEqual(audit["unsupported_calls"], 1)
                self.assertTrue(audit["method_restored"])

    def test_only_separate_predeclared_variant_names_are_accepted(self):
        self.assertEqual(select_variants("amzn_public_44mps,refresh_flyby_turn_distance"),
                         ["amzn_public_44mps", "refresh_flyby_turn_distance"])
        for names in ("", "speed_and_refresh", "baseline,baseline"):
            with self.subTest(names=names), self.assertRaises(ValueError):
                select_variants(names)


if __name__ == "__main__":
    unittest.main()
