import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np

import ftc
from ftc.models.LC62 import LC62
from ftc.utils import safeupdate

np.seterr(all="raise")

class MyEnv(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 20,
        },
        "plant": {
            "init": {
                "pos": np.vstack((0.0, 0.0, 0.0)),
                "vel": np.zeros((3, 1)),
                "quat": np.vstack((1, 0, 0, 0)),
                "omega": np.zeros((3, 1)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = LC62(env_config["plant"])
        self.env_config = env_config
        self.controller = ftc.make("Flat", self)
        self.fdi_delay = 0.1

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t, *args):
        # desired reference
        posd = np.vstack([np.sin(t), np.cos(t), -t])
        posd_1dot = np.vstack([np.cos(t), -np.sin(t), -1])
        posd_2dot = np.vstack([-np.sin(t), -np.cos(t), 0])
        posd_3dot = np.vstack([-np.cos(t), np.sin(t), 0])
        posd_4dot = np.stack([np.sin(t), np.cos(t), 0])
        refs = {"posd": posd, "posd_1dot": posd_1dot, "posd_2dot": posd_2dot, "posd_3dot": posd_3dot, "posd_4dot": posd_4dot}
        return [refs[key] for key in args]

    def set_dot(self, t):
        FM_traj, controller_info = self.controller.get_control(t, self)

        """ set faults """
        # Lambda = self.get_Lambda(t)
        # lctrls = np.vstack([
        #     (Lambda[:, None] * (ctrls[0:6] - 1000) / 1000) * 1000 + 1000,
        #     ctrls[6:11]
        # ])

        env_info = {
            "t": t,
            **controller_info,
            "FM": FM_traj,
            # "Lambda": self.get_Lambda(t),
        }

        return env_info

    def get_Lambda(self, t):
        """Lambda function"""
        if t > 5:
            W1 = 0.5
        else:
            W1 = 1
        if t > 7:
            W2 = 0.7
        else:
            W2 = 1
        Lambda = np.array([W1, W2, 1, 1, 1, 1])

        return Lambda

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

    """ Figure - Forces & Moments trajectories """
    fig, axes = plt.subplots(3, 2, squeeze=False, sharex=True)

    """ Column 1 - Forces """
    ax = axes[0, 0]
    ax.plot(data["t"], data["FM"][:, 0].squeeze(-1), "k-")
    ax.set_ylabel(r"$F_x$")
    ax.legend(["Response"], loc="upper right")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["FM"][:, 1].squeeze(-1), "k-")
    ax.set_ylabel(r"$F_y$")

    ax = axes[2, 0]
    ax.plot(data["t"], data["FM"][:, 2].squeeze(-1), "k-")
    ax.set_ylabel(r"$F_z$")

    ax.set_xlabel("Time, sec")

    """ Column 2 - Moments """
    ax = axes[0, 1]
    ax.plot(data["t"], data["FM"][:, 3].squeeze(-1), "k-")
    ax.set_ylabel(r"$M_x$")
    ax.legend(["Response"], loc="upper right")

    ax = axes[1, 1]
    ax.plot(data["t"], data["FM"][:, 4].squeeze(-1), "k-")
    ax.set_ylabel(r"$M_y$")

    ax = axes[2, 1]
    ax.plot(data["t"], data["FM"][:, 5].squeeze(-1), "k-")
    ax.set_ylabel(r"$M_z$")

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
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

