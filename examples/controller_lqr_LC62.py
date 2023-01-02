import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import angle2quat

import ftc
from ftc.models.LC62 import LC62
from ftc.utils import safeupdate

np.seterr(all="raise")


class MyEnv(fym.BaseEnv):
    ang = np.random.uniform(-np.deg2rad(0), np.deg2rad(0), size=(3, 1))
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 20,
        },
        "plant": {
            "init": {
                "pos": np.random.uniform(-1, 1, size=(3, 1)) + np.vstack((0, 0, -10)),
                "vel": np.zeros((3, 1)),
                "quat": angle2quat(ang[2], ang[1], ang[0]),
                "omega": np.zeros((3, 1)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = LC62(env_config["plant"])
        self.Q = 10 * np.diag([1, 1, 1, 0, 0, 0, 100, 100, 100, 0, 0, 0])
        self.R = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.controller = ftc.make("LQR-LC62", self)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t):
        posd = np.vstack((0, 0, 0))
        posd_dot = np.vstack((0, 0, 0))
        refs = {"posd": posd, "posd_dot": posd_dot}
        return [refs[key] for key in args]

    def set_dot(self, t):
        ctrls, controller_info = self.controller.get_control(t, self)
        rotors0 = ctrls
        rotors = rotors0

        pos, vel, quat, omega = self.plant.observe_list()
        FM = self.plant.get_FM(pos, vel, quat, omega, ctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "forces": np.zeros((4, 1)),
            "rotors0": rotors0,
            "rotors": rotors,
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
