import numpy as np

from vlalagbench.benchmarks.libero.evaluator import idle_step
from tests.benchmarks.libero.robosuite_fakes import fake_robots


class _Robosuite:
    def __init__(self):
        self.robots = fake_robots()

    def step(self, action):
        # robosuite calls set_goal once per env.step
        self.robots[0].controller.set_goal(action)
        return None, 0.0, False, {}


class _Wrapper:
    def __init__(self, env):
        self.env = env

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        return self.env.step(action)


def test_idle_step_fixes_both_goals_where_the_wait_began():
    env = _Wrapper(_Robosuite())
    controller = env.env.robots[0].controller
    idle_step(env, [0.0] * 7, wait_started=True)
    assert controller.set_goal_calls == 0
    np.testing.assert_array_equal(controller.goal_pos, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(controller.goal_ori, np.eye(3))
    controller.ee_pos = np.array([0.5, 0.5, 0.5])
    idle_step(env, [0.0] * 7, wait_started=False)
    np.testing.assert_array_equal(controller.goal_pos, [0.1, 0.2, 0.3])
    assert controller.set_goal_calls == 0
    env.step([0.0] * 7)
    assert controller.set_goal_calls == 1
