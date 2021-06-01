import numpy as np
import matplotlib.pyplot as plt

from fym.utils.rot import quat2angle, angle2quat
import fym.logging
from fym.core import BaseEnv, BaseSystem

from ftc.models.multicopter import Multicopter
from ftc.agent.integral_slidingmode import IntegralSMC


class Env(BaseEnv):
    def __init__(self,
                 pos=np.zeros((3, 1)),
                 vel=np.zeros((3, 1)),
                 quat=np.vstack((1, 0, 0, 0)),
                 omega=np.zeros((3, 1)),
                 ref=np.vstack([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]),
                 rtype="hexa-x"):
        super().__init__(dt=0.001, max_t=10)
        self.plant = Multicopter(pos, vel, quat, omega, rtype)

    def reset(self):
        super().reset()
        return self.observe_flat()

    def get_forces(self, action):
        return action

    def step(self, action):
        self.action = action
        t = self.clock.get()
        states = self.observe_flat()
        info = self.observe_dict()

        forces = self.get_forces(action)
        rotors = self.plant.mixer(forces)
        *_, done = self.update()
        return t, states, info, rotors, done

    def set_dot(self, t):
        forces = self.get_forces(self.action)
        rotors = self.plant.mixer(forces)
        self.plant.set_dot(t, rotors)


def run(env, pos, quat, ref, dtype, agent=None):
    obs = env.reset()
    logger = fym.logging.Logger(path='data.h5')
    gamma_tune = np.array([1, 1, 1, 1])
    kd_tune = np.array([25, 1, 1, ])

    while True:
        env.render()

        if agent is None:
            action = 0
        else:
            action = agent.get_action(obs, ref, gamma_tune, kd_tune, dtype=dtype)

        t, next_obs, info, rotors, done = env.step(action)
        obs = next_obs
        logger.record(t=t, **info, rotors=rotors)

        if done:
            break

    env.close()
    logger.close()


def test_integralSMC(pos, quat, ref, dtype):
    env = Env(pos, quat)
    agent = IntegralSMC(env, ref)
    run(env, pos, quat, ref, agent, dtype)
    plot_var()
    plt.show()
