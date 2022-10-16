import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import argparse
import torch

import fym
from fym.utils.rot import angle2quat, quat2angle

import ftc
from ftc.utils import safeupdate
from ftc.models.LC62 import LC62
from ftc.sim_parallel import sim_parallel

np.seterr(all="raise")


class Env(fym.BaseEnv):
    def __init__(self, initial, gain):
        super().__init__(dt=0.01, max_t=10)
        pos, vel, angle, omega = initial
        quat = angle2quat(*angle.ravel()[::-1])
        env_config = {
            "init": {
                "pos": pos,
                "vel": vel,
                "quat": quat,
                "omega": omega,
            },
        }
        self.K1, self.K2 = gain
        self.plant = LC62(env_config)
        self.controller = ftc.make("INDI", self)
        self.u0 = self.controller.get_u0(self)

    def step(self):
        env_info, done = self.update()
        if not all(-50 < a < 50 for a in np.rad2deg(env_info["ang"][:2])):
            print(np.rad2deg(env_info["ang"][:2]))
            done = True
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t, *args):
        posd = np.vstack((0, 0, 0))
        posd_dot = np.vstack((0, 0, 0))
        refs = {"posd": posd, "posd_dot": posd_dot}
        return [refs[key] for key in args]

    def set_dot(self, t):
        ctrls0, controller_info = self.controller.get_control(t, self)
        ctrls = ctrls0
        bctrls = self.plant.saturate(ctrls0)

        """ set faults """
        lctrls = self.set_Lambda(t, bctrls)  # lambda * bctrls
        ctrls = self.plant.saturate(lctrls)

        FM = self.plant.get_FM(*self.plant.observe_list(), ctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            "posd": controller_info["posd"],
            "ang": controller_info["ang"],
            "angd": controller_info["angd"],
            "ctrls0": ctrls0,
            "ctrls": ctrls,
            "rc": self.running_cost(),
        }

        return env_info

    def get_Lambda(self, t):
        """Lambda function"""

        Lambda = np.ones((11, 1))
        return Lambda

    def set_Lambda(self, t, ctrls):
        Lambda = self.get_Lambda(t)
        return Lambda * ctrls

    def running_cost(self):
        pos, vel, quat, omega = self.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        rc = pos[2]**2 + vel[2]**2 + ang.T @ ang + omega.T @ omega
        return rc


def parsim(N=1, seed=0):
    np.random.seed(seed)
    # pos = np.random.uniform(0, 0, size=(N, 3, 1))
    # vel = np.random.uniform(0, 0, size=(N, 3, 1))
    # angle = np.random.uniform(*np.deg2rad((-10, 10)), size=(N, 3, 1))
    # omega = np.random.uniform(*np.deg2rad((-0, 0)), size=(N, 3, 1))
    # k1v = np.random.uniform(5, 20, size=(N, 1))
    # k1a = np.random.uniform(10, 50, size=(N, 2))
    # k1a_psi = np.random.uniform(0.1, 5, size=(N, 1))
    # k2v = np.random.uniform(5, 20, size=(N, 1))
    # k2a = np.random.uniform(10, 50, size=(N, 2))
    # k2a_psi = np.random.uniform(0.1, 5, size=(N, 1))
    # k1 = np.hstack((k1v, k1a, k1a_psi))
    # k2 = np.hstack((k2v, k2a, k2a_psi))

    test_result = torch.load("test_result.pt")
    dataset = test_result["dataset"]
    initial_state = dataset["condition"]  # d x n
    gain = dataset["decision"]  # d x m
    predicted_optimal_gain = dataset["predicted_optimal_decision"]  # d x m

    pos = np.zeros((1000, 3, 1))
    vel = np.zeros((1000, 3, 1))
    angle = np.zeros((1000, 3, 1))
    omega = np.zeros((1000, 3, 1))
    pos[:, 2, 0] = initial_state[:, 0]
    vel[:, 2, 0] = initial_state[:, 1]
    angle[:, :, 0] = initial_state[:, 2:5]
    omega[:, :, 0] = initial_state[:, 5:8]
    # k1 = gain[:, :4]
    # k2 = gain[:, 4:]
    k1 = predicted_optimal_gain[:, :4]
    k2 = predicted_optimal_gain[:, 4:]

    initials = np.stack((pos, vel, angle, omega), axis=1)
    gains = np.stack((k1, k2), axis=1)
    sim_parallel(N, initials, gains, Env)


if __name__ == "__main__":
    N = 1
    seed = 0
    parsim(N, seed)
