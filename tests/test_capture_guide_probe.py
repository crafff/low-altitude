"""Pure guide geometry and scoped-hook restoration regressions; no native workloads."""
from types import SimpleNamespace
import unittest

from capture_guide_probe import GuideOverride, PassiveCaptureAudit, guide_geometry


def metadata():
    return dict(original_route_calls=0, initial_route_passthrough_calls=0,
        capture_route_calls=0, guide_skipped_calls=0, guide_applied_calls=0, captures=[],
        post_capture_samples_retained=0, per_fixture_physical_audit={})


class CaptureGuideProbeTests(unittest.TestCase):
    def test_fixed_parallel_guide_preserves_cap_and_lateral_position(self):
        cap = [100., 76.2]
        result = guide_geometry(cap, [0., 76.2], [2000., 76.2], [1., 0.])
        self.assertTrue(result['applicable'])
        self.assertEqual(result['guide_xy'], [600., 76.2])
        self.assertEqual(cap, [100., 76.2])
        mirrored = guide_geometry([100., -76.2], [0., -76.2], [2000., -76.2], [1., 0.])
        self.assertEqual(mirrored['guide_xy'], [600., -76.2])
        northbound = guide_geometry([-76.2, 100.], [-76.2, 0.], [-76.2, 2000.], [0., 1.])
        self.assertEqual(northbound['guide_xy'], [-76.2, 600.])

    def test_short_or_equal_remaining_leg_is_not_applicable_and_never_clamped(self):
        for end in (500., 600.):
            row = guide_geometry([100., 76.2], [0., 76.2], [end, 76.2], [1., 0.])
            self.assertFalse(row['applicable'])
            self.assertEqual(row['guide_xy'], [600., 76.2])
        row = guide_geometry([-600., 76.2], [0., 76.2], [2000., 76.2], [1., 0.])
        self.assertFalse(row['applicable'])
        with self.assertRaisesRegex(ValueError, 'unit vector'):
            guide_geometry([0., 0.], [0., 0.], [1000., 0.], [2., 0.])

    def test_original_registration_once_and_route_restoration_on_failure(self):
        calls = []
        class Controller:
            def _route(self, acid, record, capture=None, shifted=None):
                calls.append((acid, capture))
                return 'original'
            def action_mask(self, acid):
                return [True]*60
        original, original_mask = Controller._route, Controller.action_mask
        record = SimpleNamespace(geometry=SimpleNamespace(waypoints=[(0., 0.), (0., 1.)]),
                                 flight={'type': 'Mavic'})
        audit = metadata()
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            with GuideOverride(Controller, audit) as adapter:
                self.assertEqual(Controller()._route('F001', record), 'original')
                self.assertEqual(len(calls), 1)
                raise RuntimeError('injected')
        self.assertIs(Controller._route, original)
        self.assertIs(Controller.action_mask, original_mask)
        self.assertTrue(adapter.restored)
        self.assertEqual(audit['initial_route_passthrough_calls'], 1)

    def test_corner_rejected_before_original_route_mutation(self):
        class Controller:
            def _route(self, *args):
                raise AssertionError('Original route must not be called')
            def action_mask(self, acid):
                return [True]*60
        record = SimpleNamespace(geometry=SimpleNamespace(waypoints=[(0., 0.), (0., 1.), (1., 1.)]),
                                 flight={'type': 'Amzn'})
        with GuideOverride(Controller, metadata()):
            with self.assertRaisesRegex(ValueError, 'straight'):
                Controller()._route('F001', record)

    def test_passive_callback_calls_original_once_and_restores_on_error(self):
        calls = []
        class Audit:
            def capture(self, env):
                calls.append(env)
                raise RuntimeError('native failure')
        original, data = Audit.capture, metadata()
        with self.assertRaisesRegex(RuntimeError, 'native failure'):
            with PassiveCaptureAudit(Audit, data) as monitor:
                Audit().capture('env')
        self.assertEqual(calls, ['env'])
        self.assertIs(Audit.capture, original)
        self.assertTrue(monitor.restored)


if __name__ == '__main__':
    unittest.main()
