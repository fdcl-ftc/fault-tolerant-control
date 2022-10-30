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
        }

        return env_info

    def get_Lambda(self, t):
        """Lambda function"""

        Lambda = np.ones((11, 1))
        return Lambda

    def set_Lambda(self, t, ctrls):
        Lambda = self.get_Lambda(t)
        return Lambda * ctrls


def parsim(N=1, seed=0):
    """
    Generate data
    """
    np.random.seed(seed)
    pos = np.random.uniform(-0.5, 0.5, size=(N, 3, 1))
    vel = np.random.uniform(-0.5, 0.5, size=(N, 3, 1))
    angle = np.random.uniform(*np.deg2rad((-10, 10)), size=(N, 3, 1))
    omega = np.random.uniform(*np.deg2rad((-10, 10)), size=(N, 3, 1))
    k1v = np.random.uniform(20, 30, size=(N, 1))
    k1a = np.random.uniform(40, 60, size=(N, 2))
    k1a_psi = np.random.uniform(4, 6, size=(N, 1))
    k2v = np.random.uniform(8, 12, size=(N, 1))
    k2a = np.random.uniform(16, 24, size=(N, 2))
    k2a_psi = np.random.uniform(3.2, 4.8, size=(N, 1))
    k1 = np.hstack((k1v, k1a, k1a_psi))
    k2 = np.hstack((k2v, k2a, k2a_psi))

    """
    Evaluate data

    Variables:
        d: no. of data
        n: dimension of the condition variable
        m: dimension of the decision variable
    Notes:
        `gain` is merely the set of decision variables used for test.
        You may need `predicted_optimal_gain`.
    """
    # test_result = torch.load("test_result_last.pt")
    # dataset = test_result["dataset"]
    # initial_state = dataset["condition"]  # d x n
    # gain = dataset["decision"]  # d x m
    # predicted_optimal_gain = dataset["predicted_optimal_decision"]  # d x m

    # pos = np.zeros((1000, 3, 1))
    # vel = np.zeros((1000, 3, 1))
    # angle = np.zeros((1000, 3, 1))
    # omega = np.zeros((1000, 3, 1))
    # pos[:, 2, 0] = initial_state[:, 0]
    # vel[:, 2, 0] = initial_state[:, 1]
    # angle[:, :, 0] = initial_state[:, 2:5]
    # omega[:, :, 0] = initial_state[:, 5:8]
    # # k1 = gain[:, :4]
    # # k2 = gain[:, 4:]
    # k1 = predicted_optimal_gain[:, :4]
    # k2 = predicted_optimal_gain[:, 4:]

    """
    Paraller simulation
    """
    initials = np.stack((pos, vel, angle, omega), axis=1)
    gains = np.stack((k1, k2), axis=1)
    sim_parallel(N, initials, gains, Env)


if __name__ == "__main__":
    N = 1000
    seed = 0
    parsim(N, seed)

    # data = fym.load("data/env_00000.h5")["env"]
    # # dataopt = fym.load("dataopt/env_0002.h5")["env"]  # plse
    # # datatest = fym.load("datatest/env_0002.h5")["env"]  # random
    # # datafix2 = fym.load("datafix2/env_0002.h5")["env"]
    # # datafix3 = fym.load("datafix3/env_0002.h5")["env"]
    # # datafnn = fym.load("datafnn/env_0002.h5")["env"]

    # """ Figure 1 - States """
    # fig, axes = plt.subplots(2, 4, figsize=(18, 5), squeeze=False, sharex=True)

    # """ Column 1 """
    # ax = axes[0, 0]
    # ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "k-")
    # # ax.plot(dataopt["t"], dataopt["plant"]["pos"][:, 2].squeeze(-1), "k-")
    # # ax.plot(datatest["t"], datatest["plant"]["pos"][:, 2].squeeze(-1), "b-.")
    # # ax.plot(datafix2["t"], datafix2["plant"]["pos"][:, 2].squeeze(-1), "g:")
    # # ax.plot(datafix3["t"], datafix3["plant"]["pos"][:, 2].squeeze(-1), "m^")
    # # ax.plot(datafnn["t"], datafnn["plant"]["pos"][:, 2].squeeze(-1), "c.")
    # ax.plot(data["t"], data["posd"][:, 2].squeeze(-1), "r--")
    # ax.set_ylabel(r"$z$, m")
    # ax.legend(["Response", "Command"], loc="upper right")

    # ax.set_xlabel("Time, sec")

    # ax = axes[0, 1]
    # ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "k-")
    # # ax.plot(dataopt["t"], dataopt["plant"]["vel"][:, 2].squeeze(-1), "k-")
    # # ax.plot(datatest["t"], datatest["plant"]["vel"][:, 2].squeeze(-1), "b-.")
    # # ax.plot(datafix2["t"], datafix2["plant"]["vel"][:, 2].squeeze(-1), "g:")
    # # ax.plot(datafix3["t"], datafix3["plant"]["vel"][:, 2].squeeze(-1), "m^")
    # # ax.plot(datafnn["t"], datafnn["plant"]["vel"][:, 2].squeeze(-1), "c.")
    # ax.set_ylabel(r"$v_z$, m/s")

    # ax.set_xlabel("Time, sec")

    # ax = axes[0, 2]
    # ax.plot(data["t"], np.rad2deg(data["ang"][:, 0].squeeze(-1)), "k-")
    # # ax.plot(dataopt["t"], np.rad2deg(dataopt["ang"][:, 0].squeeze(-1)), "k-")
    # # ax.plot(datatest["t"], np.rad2deg(datatest["ang"][:, 0].squeeze(-1)), "b-.")
    # # ax.plot(datafix2["t"], np.rad2deg(datafix2["ang"][:, 0].squeeze(-1)), "g:")
    # # ax.plot(datafix3["t"], np.rad2deg(datafix3["ang"][:, 0].squeeze(-1)), "m^")
    # # ax.plot(datafnn["t"], np.rad2deg(datafnn["ang"][:, 0].squeeze(-1)), "c.")
    # ax.plot(data["t"], np.rad2deg(data["angd"][:, 0].squeeze(-1)), "r--")
    # ax.set_ylabel(r"$\phi$, deg")

    # ax = axes[0, 3]
    # ax.plot(data["t"], np.rad2deg(data["ang"][:, 1].squeeze(-1)), "k-")
    # # ax.plot(dataopt["t"], np.rad2deg(dataopt["ang"][:, 1].squeeze(-1)), "k-")
    # # ax.plot(datatest["t"], np.rad2deg(datatest["ang"][:, 1].squeeze(-1)), "b-.")
    # # ax.plot(datafix2["t"], np.rad2deg(datafix2["ang"][:, 1].squeeze(-1)), "g:")
    # # ax.plot(datafix3["t"], np.rad2deg(datafix3["ang"][:, 1].squeeze(-1)), "m^")
    # # ax.plot(datafnn["t"], np.rad2deg(datafnn["ang"][:, 1].squeeze(-1)), "c.")
    # ax.plot(data["t"], np.rad2deg(data["angd"][:, 1].squeeze(-1)), "r--")
    # ax.set_ylabel(r"$\theta$, deg")

    # """ Column 2 """
    # ax = axes[1, 0]
    # ax.plot(data["t"], np.rad2deg(data["ang"][:, 2].squeeze(-1)), "k-")
    # # ax.plot(dataopt["t"], np.rad2deg(dataopt["ang"][:, 2].squeeze(-1)), "k-")
    # # ax.plot(datatest["t"], np.rad2deg(datatest["ang"][:, 2].squeeze(-1)), "b-.")
    # # ax.plot(datafix2["t"], np.rad2deg(datafix2["ang"][:, 2].squeeze(-1)), "g:")
    # # ax.plot(datafix3["t"], np.rad2deg(datafix3["ang"][:, 2].squeeze(-1)), "m^")
    # # ax.plot(datafnn["t"], np.rad2deg(datafnn["ang"][:, 2].squeeze(-1)), "c.")
    # ax.plot(data["t"], np.rad2deg(data["angd"][:, 2].squeeze(-1)), "r--")
    # ax.set_ylabel(r"$\psi$, deg")

    # ax.set_xlabel("Time, sec")

    # ax = axes[1, 1]
    # ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 0].squeeze(-1)), "k-")
    # # ax.plot(dataopt["t"], np.rad2deg(dataopt["plant"]["omega"][:, 0].squeeze(-1)), "k-")
    # # ax.plot(datatest["t"], np.rad2deg(datatest["plant"]["omega"][:, 0].squeeze(-1)), "b-.")
    # # ax.plot(datafix2["t"], np.rad2deg(datafix2["plant"]["omega"][:, 0].squeeze(-1)), "g:")
    # # ax.plot(datafix3["t"], np.rad2deg(datafix3["plant"]["omega"][:, 0].squeeze(-1)), "m^")
    # # ax.plot(datafnn["t"], np.rad2deg(datafnn["plant"]["omega"][:, 0].squeeze(-1)), "c.")
    # ax.set_ylabel(r"$p$, deg/s")

    # ax = axes[1, 2]
    # ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "k-")
    # # ax.plot(dataopt["t"], np.rad2deg(dataopt["plant"]["omega"][:, 1].squeeze(-1)), "k-")
    # # ax.plot(datatest["t"], np.rad2deg(datatest["plant"]["omega"][:, 1].squeeze(-1)), "b-.")
    # # ax.plot(datafix2["t"], np.rad2deg(datafix2["plant"]["omega"][:, 1].squeeze(-1)), "g:")
    # # ax.plot(datafix3["t"], np.rad2deg(datafix3["plant"]["omega"][:, 1].squeeze(-1)), "m^")
    # # ax.plot(datafnn["t"], np.rad2deg(datafnn["plant"]["omega"][:, 1].squeeze(-1)), "c.")
    # ax.set_ylabel(r"$q$, deg/s")

    # ax = axes[1, 3]
    # ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 2].squeeze(-1)), "k-")
    # # ax.plot(dataopt["t"], np.rad2deg(dataopt["plant"]["omega"][:, 2].squeeze(-1)), "k-")
    # # ax.plot(datatest["t"], np.rad2deg(datatest["plant"]["omega"][:, 2].squeeze(-1)), "b-.")
    # # ax.plot(datafix2["t"], np.rad2deg(datafix2["plant"]["omega"][:, 2].squeeze(-1)), "g:")
    # # ax.plot(datafix3["t"], np.rad2deg(datafix3["plant"]["omega"][:, 2].squeeze(-1)), "m^")
    # # ax.plot(datafnn["t"], np.rad2deg(datafnn["plant"]["omega"][:, 2].squeeze(-1)), "c.")
    # ax.set_ylabel(r"$r$, deg/s")

    # ax.set_xlabel("Time, sec")

    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0.3)
    # fig.align_ylabels(axes)

    # """ Figure 2 - Rotor thrusts """
    # fig, axs = plt.subplots(3, 2, sharex=True)
    # ylabels = np.array((["Rotor 1", "Rotor 2"],
    #                     ["Rotor 3", "Rotor 4"],
    #                     ["Rotor 5", "Rotor 6"]))
    # for i, _ylabel in np.ndenumerate(ylabels):
    #     x, y = i
    #     ax = axs[i]
    #     ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 2*x+y], "k-", label="Response")
    #     # ax.plot(data["t"], dataopt["ctrls"].squeeze(-1)[:, 2*x+y], "k-", label="Res(PLSE)")
    #     # ax.plot(data["t"], datatest["ctrls"].squeeze(-1)[:, 2*x+y], "b-.", label="Res(Random)")
    #     # ax.plot(data["t"], datafix2["ctrls"].squeeze(-1)[:, 2*x+y], "g:", label="Res(Fix2)")
    #     # ax.plot(data["t"], datafix3["ctrls"].squeeze(-1)[:, 2*x+y], "m^", label="Res(Fix3)")
    #     # ax.plot(data["t"], datafnn["ctrls"].squeeze(-1)[:, 2*x+y], "c.", label="Res(FNN)")
    #     # ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, 2*x+y], "r--", label="Command")
    #     ax.grid()
    #     if i == (0, 1):
    #         ax.legend(loc="upper right")
    #     plt.setp(ax, ylabel=_ylabel)
    #     ax.set_ylim([1000-5, 2000+5])
    # plt.gcf().supxlabel("Time, sec")
    # plt.gcf().supylabel("Rotor Thrusts")

    # fig.tight_layout()
    # fig.subplots_adjust(wspace=0.5)
    # fig.align_ylabels(axs)

    # plt.show()
