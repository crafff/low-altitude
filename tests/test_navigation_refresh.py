"""Ordinary-flyby formula and runtime facade regressions; run through lab."""
import copy
import json
import math
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

import numpy as np

from navigation_refresh import (navigation_audit, ordinary_flyby_guidance,
                                refreshed_flyby_distance, validate_guidance)
from paper_environment import PaperEnvironment, load_environment_config


class FakeActive:
    def __init__(self):
        self.resize(2)
        self.calls = 0
        self.fail = False
        self.return_value = np.array([0])

    def resize(self, count):
        self.next_qdr = np.zeros(count)
        self.turndist = np.full(count, 1400.)
        self.flyby = np.ones(count, dtype=bool)
        self.flyturn = np.zeros(count, dtype=bool)
        self.turnfromlastwp = np.zeros(count, dtype=bool)
        self.turntonextwp = np.zeros(count, dtype=bool)
        self.turnspd = np.full(count, 170.)

    def reached(self, qdr, dist):
        self.calls += 1
        self.received_arguments = qdr, dist
        self.received_cache = self.turndist.copy()
        if self.fail:
            raise RuntimeError("native fixture error")
        self.turndist = np.logical_or(self.flyby, self.flyturn)*self.turndist
        return self.return_value


class CachedMethodProxy:
    """BlueSky facade caches methods while ordinary setattr delegates."""
    def __init__(self, target):
        self.__dict__["_target"] = target
        self.__dict__["reached"] = target.reached

    def __getattr__(self, name):
        return getattr(self._target, name)

    def __setattr__(self, name, value):
        setattr(self._target, name, value)


def fixture(proxy=True):
    active = FakeActive()
    traf = SimpleNamespace(id=["A", "B"], actwp=CachedMethodProxy(active) if proxy else active,
                           tas=np.array([80., 20.]),
                           ap=SimpleNamespace(bankdef=np.radians([25., 25.]),
                                              target_tas=np.array([200., 200.])))
    return SimpleNamespace(traf=traf), active


class NavigationRefreshTests(unittest.TestCase):
    def test_native_default_installs_nothing_and_does_not_modify_state_or_rng(self):
        bs, active = fixture()
        before, original = active.turndist.copy(), bs.traf.actwp.reached
        audit = navigation_audit("native_cached")
        audit_before, namespace_before = copy.deepcopy(audit), dict(vars(bs.traf.actwp))
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        with ordinary_flyby_guidance(bs, "native_cached", audit, gravity_mps2=9.80665):
            self.assertIs(bs.traf.actwp.reached, original)
        self.assertEqual(vars(bs.traf.actwp), namespace_before)
        self.assertEqual(audit, audit_before)
        np.testing.assert_array_equal(active.turndist, before)
        self.assertEqual(active.calls, 0)
        self.assertEqual(random.getstate(), python_rng)
        np.testing.assert_array_equal(np.random.get_state()[1], numpy_rng[1])
        self.assertEqual(np.random.get_state()[2:], numpy_rng[2:])

    def test_formula_wrap_sentinel_and_large_turn_do_not_add_a_radius_cap(self):
        radius = 44.**2/(9.80665*math.tan(math.radians(25.)))
        actual = refreshed_flyby_distance([44., 44., 44.], math.radians(25.),
                                          [350., 0., 0.], [10., -999., 170.], 9.80665)
        np.testing.assert_allclose(actual, [radius*math.tan(math.radians(10.)),
                                           0., radius*math.tan(math.radians(85.))])
        self.assertGreater(actual[2], 4.*radius)
        for mode in (None, "refresh", True):
            with self.assertRaises(ValueError):
                validate_guidance(mode)

    def test_current_tas_and_call_bearing_override_stale_cache_not_target_turnspd(self):
        bs, active = fixture()
        active.flyby[1] = False
        active.turndist[1] = 123.
        original, audit = bs.traf.actwp.reached, navigation_audit("current_state_refresh")
        turnspd, target, bank = active.turnspd.copy(), bs.traf.ap.target_tas.copy(), bs.traf.ap.bankdef.copy()
        qdr, dist = np.array([90., 0.]), np.array([1000., 500.])
        with ordinary_flyby_guidance(bs, "current_state_refresh", audit, gravity_mps2=9.80665):
            result = bs.traf.actwp.reached(qdr, dist)
            self.assertIs(result, active.return_value)
            self.assertIs(active.received_arguments[0], qdr)
            self.assertIs(active.received_arguments[1], dist)
            first = 80.**2/(9.80665*math.tan(math.radians(25.)))
            self.assertAlmostEqual(active.received_cache[0], first)
            self.assertEqual(active.received_cache[1], 123.)
            self.assertEqual(active.turndist[1], 0.)  # Native owns flyover handling.
            bs.traf.tas[0] = 40.
            bs.traf.actwp.reached(np.array([45., 0.]), dist)
            self.assertAlmostEqual(active.received_cache[0], .25*first*math.tan(math.radians(22.5)))
        self.assertIs(bs.traf.actwp.reached, original)
        self.assertEqual(active.calls, 2)
        self.assertEqual(audit["wrapper_calls"], audit["native_calls"])
        self.assertEqual(audit["rng_checks"], 2)
        self.assertTrue(audit["adapter_rng_unchanged"])
        np.testing.assert_array_equal(active.turnspd, turnspd)
        np.testing.assert_array_equal(bs.traf.ap.target_tas, target)
        np.testing.assert_array_equal(bs.traf.ap.bankdef, bank)

    def test_empty_and_changed_populations_use_fresh_arrays_and_accumulate_contexts(self):
        bs, active = fixture()
        audit, original = navigation_audit("current_state_refresh"), bs.traf.actwp.reached
        for count in (0, 3, 1):
            with ordinary_flyby_guidance(bs, "current_state_refresh", audit, gravity_mps2=9.80665):
                # Population changes after installation, as with native admission/deletion.
                bs.traf.id = [f"A{i}" for i in range(count)]
                active.resize(count)
                bs.traf.tas = np.full(count, 40.)
                bs.traf.ap.bankdef = np.full(count, math.radians(25.))
                bs.traf.actwp.reached(np.full(count, 90.), np.full(count, 1000.))
                self.assertEqual(active.received_cache.shape, (count,))
                np.testing.assert_allclose(active.received_cache,
                                          40.**2/(9.80665*math.tan(math.radians(25.))))
            self.assertIs(bs.traf.actwp.reached, original)
        self.assertEqual(audit["context_entries"], 3)
        self.assertEqual(audit["context_exits"], 3)
        self.assertEqual(audit["wrapper_calls"], 3)
        self.assertEqual(audit["native_calls"], 3)
        self.assertEqual(audit["empty_population_calls"], 1)
        self.assertEqual(audit["flyby_entries_refreshed"], 4)
        self.assertFalse(audit["context_active"])
        self.assertTrue(audit["method_restored"])
        json.dumps(audit, allow_nan=False)

    def test_unsupported_and_mismatched_arrays_fail_before_native_or_cache_mutation(self):
        for flag in ("flyturn", "turnfromlastwp", "turntonextwp", "wrong_population"):
            with self.subTest(flag=flag):
                bs, active = fixture()
                if flag == "wrong_population":
                    bs.traf.tas = np.array([80.])
                else:
                    getattr(active, flag)[0] = True
                before, original = active.turndist.copy(), bs.traf.actwp.reached
                audit = navigation_audit("current_state_refresh")
                with self.assertRaises(RuntimeError):
                    with ordinary_flyby_guidance(bs, "current_state_refresh", audit, gravity_mps2=9.80665):
                        bs.traf.actwp.reached(np.array([90., 0.]), np.array([1000., 500.]))
                self.assertIs(bs.traf.actwp.reached, original)
                np.testing.assert_array_equal(active.turndist, before)
                self.assertEqual(active.calls, 0)
                self.assertTrue(audit["method_restored"])

    def test_native_exception_restores_cached_proxy_and_uncached_bound_method(self):
        for proxy in (True, False):
            with self.subTest(proxy=proxy):
                bs, active = fixture(proxy)
                active.fail = True
                original, had_member = bs.traf.actwp.reached, "reached" in vars(bs.traf.actwp)
                audit = navigation_audit("current_state_refresh")
                with self.assertRaisesRegex(RuntimeError, "native fixture error"):
                    with ordinary_flyby_guidance(bs, "current_state_refresh", audit, gravity_mps2=9.80665):
                        bs.traf.actwp.reached(np.array([90., 0.]), np.array([1000., 500.]))
                self.assertEqual(bs.traf.actwp.reached, original)
                self.assertEqual("reached" in vars(bs.traf.actwp), had_member)
                self.assertEqual(active.calls, 1)
                self.assertTrue(audit["method_restored"])
                self.assertFalse(audit["context_active"])

    def test_public_environment_step_owns_wrapper_only_until_return_or_exception(self):
        for mode in ("native_cached", "current_state_refresh"):
            for fail in (False, True):
                with self.subTest(mode=mode, fail=fail):
                    bs, active = fixture()
                    active.fail = fail
                    env = PaperEnvironment.__new__(PaperEnvironment)
                    env.bs, env.done, env.ordinary_flyby_guidance = bs, False, mode
                    env.navigation_audit = navigation_audit(mode)
                    original = bs.traf.actwp.reached
                    def transition(actions):
                        self.assertEqual(bs.traf.actwp.reached is original, mode == "native_cached")
                        return bs.traf.actwp.reached(np.array([90., 0.]), np.array([1000., 500.]))
                    env._step = transition
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "native fixture error"):
                            env.step(None)
                    else:
                        self.assertIs(env.step(None), active.return_value)
                    self.assertIs(bs.traf.actwp.reached, original)
                    self.assertFalse(env.navigation_audit["context_active"])
                    self.assertEqual(env.navigation_audit["context_entries"], int(mode == "current_state_refresh"))

    def test_new_configs_change_only_declared_guidance_and_training_lineage(self):
        root = Path(__file__).resolve().parents[1]
        old, old_parts = load_environment_config(root/"configs/paper_environment_execution.json")
        new, new_parts = load_environment_config(root/"configs/paper_environment_execution_refresh.json")
        self.assertNotIn("ordinary_flyby_guidance", old)
        self.assertEqual(new["ordinary_flyby_guidance"], "current_state_refresh")
        self.assertEqual(old_parts, new_parts)
        self.assertEqual({k: v for k, v in old.items() if k != "reconstruction_choices"},
                         {k: v for k, v in new.items() if k not in ("reconstruction_choices", "ordinary_flyby_guidance")})
        old_train = json.loads((root/"configs/paper_train_execution.json").read_text())
        new_train = json.loads((root/"configs/paper_train_execution_refresh.json").read_text())
        self.assertEqual(new_train["environment_config"], "configs/paper_environment_execution_refresh.json")
        self.assertEqual({k: v for k, v in old_train.items() if k not in ("environment_config", "reconstruction_choices")},
                         {k: v for k, v in new_train.items() if k not in ("environment_config", "reconstruction_choices")})


if __name__ == "__main__":
    unittest.main()
