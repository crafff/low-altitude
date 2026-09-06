"""Focused wrapper regression; fake creation avoids an unrelated native rollout."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paper_environment import PaperEnvironment, load_environment_config


class AdmissionProgressTests(unittest.TestCase):
    def test_invalid_exit_tolerance_is_rejected_before_native_initialization(self):
        cfg, parts = load_environment_config('configs/paper_environment_shared_navigation.json')
        for value in (-1., float('nan'), float('inf'), True, '0.001'):
            with patch('paper_environment.initialise') as initialise:
                with self.assertRaises(ValueError):
                    PaperEnvironment(dict(cfg, exit_width_tolerance_m=value), parts)
                initialise.assert_not_called()

    def test_final_leg_activation_is_recorded_before_first_action_can_insert_capture(self):
        env = PaperEnvironment.__new__(PaperEnvironment)
        env.bs = SimpleNamespace(sim=SimpleNamespace(simt=0.), traf=SimpleNamespace(
            id=[], cre=lambda *args: True))
        native = {'final_nominal_active': True}
        env.actions = SimpleNamespace(register=lambda *args: None,
                                      state_fields=lambda acid: dict(native))
        env.scenario_cfg = {'altitude_ft':350.}
        env.types = {'test':{'nominal_tas_mps':20.}}
        env.flights = [{'id':'A', 'type':'test', 'corridor_id':'C', 'scheduled_entry_s':0.}]
        env.corridors = {'C':[(0., 0.), (0., .01)]}
        env.records = {'A':{'status':'pending', 'final_leg_activated':False}}
        env.next_flight = 0
        env._admit_due()
        self.assertTrue(env.records['A']['final_leg_activated'])
        # An inserted artificial capture changes active waypoint identity, not
        # the mission progress which has already occurred at registration.
        native['final_nominal_active'] = False
        self.assertFalse(env.actions.state_fields('A')['final_nominal_active'])
        self.assertTrue(env.records['A']['final_leg_activated'])
        self.assertEqual(env.next_flight, 1)


if __name__ == '__main__':
    unittest.main()
