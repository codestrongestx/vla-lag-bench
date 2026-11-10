"""Minimal robosuite arm stand-ins for evaluator tests."""

import numpy as np


class FakeController:
    def __init__(self):
        self.ee_pos = np.array([0.1, 0.2, 0.3])
        self.ee_ori_mat = np.eye(3)
        self.goal_pos = np.array([9.0, 9.0, 9.0])
        self.goal_ori = np.diag([1.0, -1.0, -1.0])
        self.set_goal_calls = 0

    def update(self, force=False):
        pass

    def set_goal(self, action):
        self.set_goal_calls += 1


def fake_robots():
    return [type("Robot", (), {"controller": FakeController()})()]
