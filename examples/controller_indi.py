import numpy as np
import matplotlib.pyplot as plt
import argparse

import fym

import ftc
from ftc.utils import safeupdate
from ftc.models.LC62 import LC62

np.seterr(all="raise")


class ActuatorDynamics(fym.BaseSystem):
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def set_dot(self, ctrls, ctrls_cmd):
        self.dot = -1 / self.tau * (ctrls - ctrls_cmd)


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
        self.ndicontroller = ftc.make("NDI", self)
        self.controller = ftc.make("INDI", self)
        # self.rotor_dyn = ActuatorDynamics(tau=0.01, shape=(11, 1))

    def step(self, u0):
        env_info, done = self.update(u0=u0)
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t, *args):
        posd = np.vstack((1, 1, 0))
        posd_dot = np.vstack((0, 0, 0))
        refs = {"posd": posd, "posd_dot": posd_dot}
        return [refs[key] for key in args]

    def set_dot(self, t, u0):
        ctrls0, controller_info = self.controller.get_control(t, self, u0)
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
            **controller_info,
            "ctrls0": ctrls0,
            "ctrls": ctrls,
            "FM": FM,
            "Lambda": self.get_Lambda(t),
        }

        return env_info

    def get_Lambda(self, t):
        """Lambda function"""

        Lambda = np.ones((11, 1))
        return Lambda

    def set_Lambda(self, t, ctrls):
        Lambda = self.get_Lambda(t)
        return Lambda * ctrls


def run():
    env = MyEnv()
    flogger = fym.Logger("data.h5")

    env.reset()
    u0, _ = env.ndicontroller.get_control(0, env)
    try:
        while True:
            env.render()

            done, env_info = env.step(u0)
            flogger.record(env=env_info)
            u0 = env_info["ctrls0"]

            if done:
                break

    finally:
        flogger.close()
        plot()


def plot():
    data = fym.load("data.h5")["env"]

    """ Figure 1 - States """
    fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "k-")
    ax.plot(data["t"], data["posd"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$x$, m")
    ax.legend(["Response", "Command"], loc="upper right")
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

    ax = axes[1, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 1].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_y$, m/s")

    ax = axes[2, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 2].squeeze(-1), "k-")
    ax.set_ylabel(r"$v_z$, m/s")

    ax.set_xlabel("Time, sec")

    """ Column 3 - States: Euler angles """
    ax = axes[0, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 0].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 0].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\phi$, deg")

    ax = axes[1, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 1].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 1].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\theta$, deg")

    ax = axes[2, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 2].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 2].squeeze(-1)), "r--")
    ax.set_ylabel(r"$\psi$, deg")

    ax.set_xlabel("Time, sec")

    """ Column 4 - States: Angular rates """
    ax = axes[0, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 0].squeeze(-1)), "k-")
    ax.set_ylabel(r"$p$, deg/s")
    ax.legend(["Response", "Ref"], loc="upper right")

    ax = axes[1, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "k-")
    ax.set_ylabel(r"$q$, deg/s")

    ax = axes[2, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 2].squeeze(-1)), "k-")
    ax.set_ylabel(r"$r$, deg/s")

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.align_ylabels(axes)

    """ Figure 2 - Generalized forces """
    fig, axes = plt.subplots(3, 2, squeeze=False, sharex=True)

    """ Column 1 - Generalized forces: Forces """
    ax = axes[0, 0]
    ax.plot(data["t"], data["FM"][:, 0].squeeze(-1), "k-")
    ax.plot(data["t"], data["FM"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$F_x$")
    ax.legend(["Response", "Command"], loc="upper right")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["FM"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["FM"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$F_y$")

    ax = axes[2, 0]
    ax.plot(data["t"], data["FM"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["FM"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$F_z$")

    ax.set_xlabel("Time, sec")

    """ Column 2 - Generalized forces: Moments """
    ax = axes[0, 1]
    ax.plot(data["t"], data["FM"][:, 3].squeeze(-1), "k-")
    ax.plot(data["t"], data["FM"][:, 3].squeeze(-1), "r--")
    ax.set_ylabel(r"$M_x$")
    ax.legend(["Response", "Ref"], loc="upper right")

    ax = axes[1, 1]
    ax.plot(data["t"], data["FM"][:, 4].squeeze(-1), "k-")
    ax.plot(data["t"], data["FM"][:, 4].squeeze(-1), "r--")
    ax.set_ylabel(r"$M_y$")

    ax = axes[2, 1]
    ax.plot(data["t"], data["FM"][:, 5].squeeze(-1), "k-")
    ax.plot(data["t"], data["FM"][:, 5].squeeze(-1), "r--")
    ax.set_ylabel(r"$M_z$")

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axes)

    """ Figure 3 - Rotor thrusts """
    fig, axs = plt.subplots(3, 2, sharex=True)
    ylabels = np.array((["Rotor 1", "Rotor 2"],
                        ["Rotor 3", "Rotor 4"],
                        ["Rotor 5", "Rotor 6"]))
    for i, _ylabel in np.ndenumerate(ylabels):
        ax = axs[i]
        ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, sum(i)], "k-", label="Response")
        ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, sum(i)], "r--", label="Command")
        ax.grid()
        if i == (0, 1):
            ax.legend(loc="upper right")
        plt.setp(ax, ylabel=_ylabel)
        ax.set_ylim([1000-5, 2000+5])
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Rotor Thrusts")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axs)

    """ Figure 4 - Pusher and Control surfaces """
    fig, axs = plt.subplots(5, 1, sharex=True)
    ylabels = np.array(("Pusher 1", "Pusher 2",
                        r"$\delta_a$", r"$\delta_e$", r"$\delta_r$"))
    for i, _ylabel in enumerate(ylabels):
        ax = axs[i]
        ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, i+6], "k-", label="Response")
        ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, i+6], "r--", label="Command")
        ax.grid()
        plt.setp(ax, ylabel=_ylabel)
        # if i < 2:
        #     ax.set_ylim([1000-5, 2000+5])
        if i == 0:
            ax.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Pusher and Control Surfaces")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axs)

    """ Figure 5 - LPF """
    fig, axs = plt.subplots(4, 1, sharex=True)
    ylabels = np.array(("ddz", "dp", "dq", "dr"))
    for i, _ylabel in enumerate(ylabels):
        ax = axs[i]
        ax.plot(data["t"], data["dxi"].squeeze(-1)[:, i], "k-", label="Response")
        ax.plot(data["t"], data["dxic"].squeeze(-1)[:, i], "r--", label="Command")
        ax.grid()
        plt.setp(ax, ylabel=_ylabel)
        if i == 0:
            ax.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Filter")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
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
