"""
LC62-50B LQR control
considering Flight mode

VTOL mode: use only rotors
FW   mode: use all actuators
"""
import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import angle2quat

import ftc
from ftc.models.LinearLC62 import LinearLC62
from ftc.utils import safeupdate

np.seterr(all="raise")


class MyEnv(fym.BaseEnv):
    ang = np.deg2rad(np.random.uniform(-0, 0, size=(3, 1)))
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 50,
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
        self.plant = LinearLC62(env_config["plant"])

        # Hovering
        self.x_trims_HV, self.u_trims_fixed_HV = self.plant.get_trim_fixed(
            fixed={"h": 10, "VT": 0}
        )
        self.u_trims_vtol_HV = self.plant.get_trim_vtol(
            fixed={"x_trims": self.x_trims_HV, "u_trims_fixed": self.u_trims_fixed_HV}
        )

        self.Q_HV = 10 * np.diag([100, 1, 1000, 50, 1, 1, 100, 100, 100, 0, 0, 0])
        self.R_HV = 100 * np.diag([1, 1, 1, 1, 1, 1])

        # FW
        self.x_trims_FW, self.u_trims_fixed_FW = self.plant.get_trim_fixed(
            fixed={"h": 10, "VT": 5}
        )
        self.u_trims_vtol_FW = self.plant.get_trim_vtol(
            fixed={"x_trims": self.x_trims_FW, "u_trims_fixed": self.u_trims_fixed_FW}
        )
        self.Q_FW = 10 * np.diag([100, 1, 2000, 2000, 1, 200, 100, 100, 100, 0, 0, 0])
        self.R_FW = np.diag([1, 1, 10, 1000, 10])

        self.controller = ftc.make("LQR-LC62", self)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t, *args):
        if 0 <= t <= 15:
            # Vertical takeoff + Hovering
            posd = np.vstack((0, 0, -10))
            posd_dot = np.vstack((0, 0, 0))
            w_r = 1
        elif 15 < t <= 35:
            # Level flight
            VT = 5
            posd_dot = np.vstack((VT, 0, 0))
            posd = np.vstack((0, 0, -10)) + (t - 15) * posd_dot
            w_r = 0
        elif 35 < t <= 40:
            # Hovering
            posd = np.vstack((100, 0, -10))
            posd_dot = np.vstack((0, 0, 0))
            w_r = 1
        elif 40 < t <= 50:
            # Vertical landing
            posd = np.vstack((100, 0, 0))
            posd_dot = np.vstack((0, 0, 0))
            w_r = 1

        refs = {"posd": posd, "posd_dot": posd_dot, "w_r": w_r}
        return [refs[key] for key in args]

    def set_dot(self, t):
        ctrls0, controller_info = self.controller.get_control(t, self)
        ctrls = ctrls0

        pos, vel, quat, omega = self.plant.observe_list()
        FM = self.plant.get_FM(pos, vel, quat, omega, ctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "FM": FM,
            "ctrls": ctrls,
            "ctrls0": ctrls0,
            "Lambda": np.ones((11, 1)),
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
    ax.set_xlabel("Time, sec")
    # ax.legend(["Response", "Ref"], loc="upper right")
    ax.set_xlim(data["t"][0], data["t"][-1])
    ax.axvspan(0, 10, color="g", alpha=0.25)
    ax.axvspan(10, 15, color="g", alpha=0.4)
    ax.axvspan(15, 35, color="b", alpha=0.25)
    ax.axvspan(35, 40, color="g", alpha=0.4)
    ax.axvspan(40, 50, color="g", alpha=0.25)
    ax.set_xticks(np.linspace(0, 50, 11))
    ax.grid()

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$y$, m")
    ax.set_xlabel("Time, sec")
    ax.axvspan(0, 10, color="g", alpha=0.25)
    ax.axvspan(10, 15, color="g", alpha=0.4)
    ax.axvspan(15, 35, color="b", alpha=0.25)
    ax.axvspan(35, 40, color="g", alpha=0.4)
    ax.axvspan(40, 50, color="g", alpha=0.25)
    ax.set_ylim(-5, 5)
    ax.set_xticks(np.linspace(0, 50, 11))
    ax.grid()

    ax = axes[2, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$z$, m")
    ax.set_xlabel("Time, sec")
    ax.axvspan(0, 10, color="g", alpha=0.25)
    ax.axvspan(10, 15, color="g", alpha=0.4)
    ax.axvspan(15, 35, color="b", alpha=0.25)
    ax.axvspan(35, 40, color="g", alpha=0.4)
    ax.axvspan(40, 50, color="g", alpha=0.25)
    ax.set_xticks(np.linspace(0, 50, 11))
    ax.grid()

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "k-")
    ax.plot(data["t"], data["veld"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_x$, m/s")
    ax.set_xlabel("Time, sec")
    ax.axvspan(0, 10, color="g", alpha=0.25)
    ax.axvspan(10, 15, color="g", alpha=0.4)
    ax.axvspan(15, 35, color="b", alpha=0.25)
    ax.axvspan(35, 40, color="g", alpha=0.4)
    ax.axvspan(40, 50, color="g", alpha=0.25)
    ax.set_xticks(np.linspace(0, 50, 11))
    ax.grid()

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["veld"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_y$, m/s")
    ax.set_xlabel("Time, sec")
    ax.set_ylim(-5, 5)
    ax.axvspan(0, 10, color="g", alpha=0.25)
    ax.axvspan(10, 15, color="g", alpha=0.4)
    ax.axvspan(15, 35, color="b", alpha=0.25)
    ax.axvspan(35, 40, color="g", alpha=0.4)
    ax.axvspan(40, 50, color="g", alpha=0.25)
    ax.set_xticks(np.linspace(0, 50, 11))
    ax.grid()

    ax = axes[2, 1]
    x = ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "k-")
    x_ref = ax.plot(data["t"], data["veld"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$v_z$, m/s")
    ax.set_xlabel("Time, sec")
    ax.set_ylim(-5, 5)

    ax.axvspan(0, 10, color="g", alpha=0.25)
    ax.axvspan(10, 15, color="g", alpha=0.4)
    ax.axvspan(15, 35, color="b", alpha=0.25)
    ax.axvspan(35, 40, color="g", alpha=0.4)
    ax.axvspan(40, 50, color="g", alpha=0.25)
    ax.set_xticks(np.linspace(0, 50, 11))
    ax.grid()

    """ Figure 2 - Rotor thrusts """
    fig, axs = plt.subplots(3, 2, sharex=True)
    ylabels = np.array(
        (["Rotor 1", "Rotor 2"], ["Rotor 3", "Rotor 4"], ["Rotor 5", "Rotor 6"])
    )
    for i, _ylabel in np.ndenumerate(ylabels):
        x, y = i
        ax = axs[i]
        ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 2 * x + y], "k-")
        ax.set_xlim(data["t"][0], data["t"][-1])
        ax.set_ylim([1400, 1700])
        ax.axvspan(0, 10, color="g", alpha=0.25)
        ax.axvspan(10, 15, color="g", alpha=0.4)
        ax.axvspan(15, 35, color="b", alpha=0.25)
        ax.axvspan(35, 40, color="g", alpha=0.4)
        ax.axvspan(40, 50, color="g", alpha=0.25)
        ax.grid()
        ax.set_xticks([0, 10, 15, 35, 40, 50])
        if i[0] == 2:
            ax.set_xlabel("Time, sec")

        plt.setp(ax, ylabel=_ylabel)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axs)

    """ Figure 3 - Pusher and Control surfaces """
    fig, axs = plt.subplots(5, 1, sharex=True)
    ylabels = np.array(
        ("Pusher 1", "Pusher 2", r"$\delta_a$", r"$\delta_e$", r"$\delta_r$")
    )
    for i, _ylabel in enumerate(ylabels):
        ax = axs[i]
        ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, i + 6], "k-")
        ax.set_xlim(data["t"][0], data["t"][-1])
        ax.set_xticks([0, 10, 15, 35, 40, 50])
        ax.grid()
        plt.setp(ax, ylabel=_ylabel)
        ax.axvspan(0, 10, color="g", alpha=0.25)
        ax.axvspan(10, 15, color="g", alpha=0.4)
        ax.axvspan(15, 35, color="b", alpha=0.25)
        ax.axvspan(35, 40, color="g", alpha=0.4)
        ax.axvspan(40, 50, color="g", alpha=0.25)

        if i == 4:
            ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.align_ylabels(axs)

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
