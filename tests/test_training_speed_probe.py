"""Focused safety and CPU sampling-boundary checks; run inside lab."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

import paper_train
import training_speed_probe as probe


class ProbeTests(unittest.TestCase):
    def test_sampling_adapter_matches_existing_cpu_rng_and_outputs(self):
        model, _ = probe.model_and_optimizer(probe.config())
        obs = {name: dict(own=np.arange(7, dtype=np.float32) / 10,
                          intruders=np.ones((n, 10), dtype=np.float32) / 10,
                          action_mask=np.arange(60) % 3 == 0)
               for name, n in [('z', 0), ('a', 2), ('b', 1)]}
        for deterministic in (False, True):
            a, b = torch.Generator().manual_seed(104), torch.Generator().manual_seed(104)
            self.assertEqual(paper_train.select_actions(model, obs, a, deterministic=deterministic),
                             probe.select(model, obs, b, deterministic=deterministic))
            self.assertTrue(torch.equal(a.get_state(), b.get_state()))

    def test_missing_protected_pid_prevents_cuda_initialization(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch.object(probe, 'OUT', Path(temporary)), \
                 patch.object(probe, 'gpu_snapshot', return_value=dict(free_mib=4000, apps=[])), \
                 patch.object(torch.cuda.memory, 'set_per_process_memory_fraction') as cap:
                with self.assertRaisesRegex(RuntimeError, 'protected GPU PID'):
                    probe.GPUWatch()
                cap.assert_not_called()

    def test_guard_exits_own_process_even_when_error_log_write_fails(self):
        guard = probe.GPUWatch.__new__(probe.GPUWatch)
        guard.stop = Mock()
        guard.stop.wait.return_value = False
        guard.check = Mock(side_effect=RuntimeError('headroom'))
        with patch.object(probe, 'write', side_effect=OSError('disk full')), \
             patch.object(probe.os, '_exit', side_effect=SystemExit(74)) as leave:
            with self.assertRaises(SystemExit) as caught:
                guard.loop()
            self.assertEqual(caught.exception.code, 74)
            leave.assert_called_once_with(74)


if __name__ == '__main__':
    unittest.main()
