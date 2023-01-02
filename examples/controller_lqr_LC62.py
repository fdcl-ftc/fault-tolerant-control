""" This is main file.
Should be placed in fault-tolerant-control/examples/
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse

import fym
from fym.utils.rot import angle2quat
import ftc
from ftc.models.LC62 import LC62
from ftc.utils import safeupdate

np.seterr(all="raise")

class MyEnv(fym.BaseEnv):
    ang = np.vstack((0,0,0))
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 100,
        },
        "plant": {
            "init": {
                "pos": np.vstack((5, 5, -10)),
                "vel": np.vstack((0,0,0)),
                "quat": angle2quat(ang[2], ang[1], ang[0]),
                "omega": np.vstack((0,0,0)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = LC62(env_config["plant"])
        self.Q = np.diag([100, 100, 100, 0, 0, 0, 1000, 1000, 1000, 0, 0, 0])
        self.R = np.diag([1, 1, 1, 1, 1, 1, 1000000, 1000000, 100000, 100000, 100000])
        self.controller = ftc.make("LQR-LC62", self)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t):
        posd = self.controller.x_trims[0:3]
        veld = self.controller.x_trims[3:6]
        angd = self.controller.x_trims[6:9]
        omegad = self.controller.x_trims[9:12]
        return posd, veld, angd, omegad
            

    def set_dot(self, t):
        ctrls, controller_info = self.controller.get_control(t, self)
        pos, vel, quat, omega = self.plant.observe_list()
        FM = self.plant.get_FM(pos, vel, quat, omega, ctrls)
        self.plant.set_dot(t, FM)


        pwms_rotors = ctrls[0:6]
        pwms_pusher = ctrls[6:8]
        dels = ctrls[8:11]

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
           # "pwms_rotors": pwms_rotors
           # "pwms_pusher": pwms_pusher
           # "dels": dels
        }

        return env_info

def run():
    env = MyEnv()
    flogger = fym.Logger("data.h5")

    env.reset()
    try:
        while True:
            env.render()

            done, env_info = env.step()
            flogger.record(env=env_info)

            if done:
                break

    finally:
        flogger.close()
        plot()


def plot():
    data = fym.load("data.h5")["env"]
    data = fym.load("data.h5")["env"]

    """ Figure 1 - States """
    fig, axes = plt.subplots(3, 2, figsize=(18, 5), squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$x$, m")
    # ax.legend(["Response", "Ref"], loc="upper right")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$y$, m")

    ax = axes[2, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$z$, m")

    ax.set_xlabel("Time, sec")

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_x$, m/s")
    # ax.legend(["Response", "Ref"], loc="upper right")

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 1].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_y$, m/s")

    ax = axes[2, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_z$, m/s")

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.align_ylabels(axes)

    plt.show()

def main(args):
    if args.only_plot:
        plot()
        return
    else:
       run()
       if args.plot:
           plot()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-P", "--only-plot", action="store_true")
    args = parser.parse_args()
    main(args)
