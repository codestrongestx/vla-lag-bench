"""The latency model reproduces the paper's fitted sensitivities and held-out errors."""
import unittest

from analysis.latency_model import refined
from analysis.latency_model.data import TASKS, load
from analysis.latency_model.speed_latency_model import schedule_features

# Table tab:sensitivities: (a per unit relative staleness, b) per task, release, set-down, grasping, articulated.
PAPER = {'pi0.5': ((0.04, 0.59), (0.33, 0.32), (0.23, 0.07), (-0.03, 1.41)),
         'MolmoAct2': ((-0.09, 0.80), (0.31, 0.09), (0.23, -0.06), (0.09, 1.37))}
BLIND = {'pi0.5': 5.7, 'MolmoAct2': 7.8}


class Model(unittest.TestCase):
    def test_schedule(self):
        # Extra mean observation age (steps) and idle steps per acting step at 100 ms (L = 2, H = 5).
        self.assertEqual(schedule_features('synchronous', 100), (2.0, 0.4))
        self.assertEqual(schedule_features('lazy', 100), (2.0, 0.0))
        self.assertEqual(schedule_features('continuous', 100), (0.5, 0.0))

    def test_sensitivities_and_blind_error(self):
        for policy, expected in PAPER.items():
            summary, _ = refined.blind_test(load(policy=policy), ('M1',))
            self.assertEqual(round(summary['M1']['blind'], 1), BLIND[policy])
            for task, (a, b) in zip(TASKS, expected):
                fitted = summary['M1']['tasks'][task]['all']
                self.assertAlmostEqual(round(2 * fitted['a'], 2), a, msg=f'{policy} {task} a')
                self.assertAlmostEqual(round(fitted['b'], 2), b, msg=f'{policy} {task} b')


if __name__ == '__main__':
    unittest.main()
