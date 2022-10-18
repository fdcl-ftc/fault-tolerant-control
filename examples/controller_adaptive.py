import sys
import traceback
from pathlib import Path

import click
import fym
import ipdb
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from numpy import cos, sin

from ftc.models.LC62 import LC62
from ftc.utils import make, safeupdate


class Scenario:
    """Position control scenario

    Source:
        https://www.notion.so/nrfteams/3-8252622d923d45a7af4cf84e9c5f0fd0#9e0660872bdd45159f48f6e23cd24aec

    """

    LAMBDA_SCENARIOS = [
        {
            "rotor1": {"t": 4, "level": 1.0},
            "rotor2": {"t": 8, "level": 1.0},
            "rotor3": {"t1": 12, "t2": 12, "level": 1.0},
            "rotor4": {"t": 15, "level": 1.0},
        },
        {
            "rotor1": {"t": 3, "level": 0.1},
            "rotor2": {"t": 7, "level": 0.5},
            "rotor3": {"t1": 10, "t2": 13, "level": 0.7},
            "rotor4": {"t": 15, "level": 0.6},
        },
    ]

    def __init__(self, scenario_config={}):
        self.init = {
            "pos": np.vstack((0.0, 0.0, 0.0)),
            "vel": np.zeros((3, 1)),
            "quat": np.vstack((1.0, 0.0, 0.0, 0.0)),
            "omega": np.zeros((3, 1)),
        }
        self.Lambda_scenario = self.LAMBDA_SCENARIOS[scenario_config["index"]]

    def posd(self, t):
        return np.vstack((5.0, 0.0, -0.0))

    def posd_dot(self, t):
        return np.zeros((3, 1))

    def posd_ddot(self, t):
        return np.zeros((3, 1))

    def psid(self, t):
        return 0.0

    def psid_dot(self, t):
        return 0.0

    def psid_ddot(self, t):
        return 0.0

    def get_disturbance(self, t):
        dv = np.vstack((0.3 * sin(3 * t), 0.3 * cos(2 * t), 0.3 * sin(t)))
        domega = np.vstack((0.2 * sin(1.5 * t), 0.2 * cos(2.5 * t), 0.2 * sin(t)))
        return dv, domega

    def get_Lambda(self, t):
        ls = self.Lambda_scenario

        Lambda = np.ones((6, 1))
        Lambda[0] = 1 if t < ls["rotor1"]["t"] else ls["rotor1"]["level"]
        Lambda[1] = 1 if t < ls["rotor2"]["t"] else ls["rotor2"]["level"]
        Lambda[2] = (
            1
            if t < ls["rotor3"]["t1"]
            else 1
            - (1 - ls["rotor3"]["level"])
            * np.sin(
                (np.pi / 2)
                * (t - ls["rotor3"]["t1"])
                / (ls["rotor3"]["t2"] - ls["rotor3"]["t1"])
            )
            if t < ls["rotor3"]["t2"]
            else ls["rotor3"]["level"]
        )
        Lambda[3] = 1 if t < ls["rotor4"]["t"] else ls["rotor4"]["level"]
        return Lambda

    def get_delta(self, t):
        return np.zeros((4, 1))


class Env(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 20,
        },
        "scenario_config": {"index": 0},
        "break": False,
    }

    def __init__(self, env_config={}):
        self.env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**self.env_config["fkw"])

        self.scenario = Scenario(self.env_config["scenario_config"])
        self.plant = LC62({"init": self.scenario.init})
        self.controller = make(id="Adaptive", env=self)

        self.brk = self.env_config["break"]

    def set_dot(self, t):
        control_input, controller_info = self.controller.get_control(t, self)
        FM = self.plant.get_FM(*self.plant.observe_list(), ctrls=control_input)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **controller_info,
        }
        return env_info


def _run(env_config):
    np.seterr(all="raise")

    env = Env(env_config)

    # save logger path
    path = Path("data.h5")
    env.logger = fym.Logger(path)
    env.logger.set_info(**env_config)
    logger.info(f"Data file will be saved in {path}")

    # simulate
    env.reset()

    try:
        while True:
            env.render()

            _, done = env.update()

            if done:
                break

    except:
        *_, tb = sys.exc_info()
        traceback.print_exc()
        if env_config["break"]:
            ipdb.post_mortem(tb)

    finally:
        env.close()
        return path


def selected_plots(path, t_range=(0, None), show=True):
    data = dict(fym.load(path))

    t0 = t_range[0]
    tf = t_range[1] or max(data["t"])

    fig, axes = plt.subplots(3, 2, figsize=(8, 6), squeeze=False, sharex=True)

    cax = axes[:, 0]

    ax = cax[0]
    ax.plot(data["t"], data["pos"][:, 0, 0], "k-", label=r"$x$")
    ax.plot(data["t"], data["posd"][:, 0, 0], "r--", label=r"$x_d$")
    ax.legend(loc="upper right")

    ax = cax[1]
    ax.plot(data["t"], data["pos"][:, 1, 0], "k-", label=r"$y$")
    ax.plot(data["t"], data["posd"][:, 1, 0], "r--", label=r"$y_d$")
    ax.legend(loc="upper right")

    ax = cax[2]
    ax.plot(data["t"], data["pos"][:, 2, 0], "k-", label=r"$z$")
    ax.plot(data["t"], data["posd"][:, 2, 0], "r--", label=r"$z_d$")
    ax.legend(loc="upper right")

    ax = cax[-1]
    ax.set_xlabel("Time, sec")
    ax.set_xlim(t0, tf)

    for ax in cax:
        ax.set_ylim(-6, 6)

    cax = axes[:, 1]

    ax = cax[0]
    ax.plot(data["t"], (data["pos"] - data["posd"])[:, 0, 0], "k-", label=r"$e_x$")
    ax.legend(loc="upper right")

    ax = cax[1]
    ax.plot(data["t"], (data["pos"] - data["posd"])[:, 1, 0], "k-", label=r"$e_y$")
    ax.legend(loc="upper right")

    ax = cax[2]
    ax.plot(data["t"], (data["pos"] - data["posd"])[:, 2, 0], "k-", label=r"$e_z$")
    ax.legend(loc="upper right")

    ax = cax[-1]
    ax.set_xlabel("Time, sec")

    for ax in cax:
        ax.set_ylim(-6, 6)

    ax = cax[-1]
    ax.set_xlabel("Time, sec")

    fig.tight_layout()

    """ Plot 2 """

    fig, axes = plt.subplots(3, 3, figsize=(9.5, 6), squeeze=False, sharex=True)

    cax = axes[:, 0]

    ax = cax[0]
    ax.plot(data["t"], np.rad2deg(data["anglesd"][:, 0, 0]), "r--", label=r"$\phi_d$")
    ax.plot(data["t"], np.rad2deg(data["angles"][:, 0, 0]), "k-", label=r"$\phi$")
    ax.legend(loc="upper right")

    ax = cax[1]
    ax.plot(data["t"], np.rad2deg(data["anglesd"][:, 1, 0]), "r--", label=r"$\theta_d$")
    ax.plot(data["t"], np.rad2deg(data["angles"][:, 1, 0]), "k-", label=r"$\theta$")
    ax.legend(loc="upper right")

    ax = cax[2]
    ax.plot(data["t"], np.rad2deg(data["anglesd"][:, 2, 0]), "r--", label=r"$\psi_d$")
    ax.plot(data["t"], np.rad2deg(data["angles"][:, 2, 0]), "k-", label=r"$\psi$")
    ax.legend(loc="upper right")

    for ax in cax:
        ax.set_ylim(-5, 5)

    ax = cax[-1]
    ax.set_xlabel("Time, sec")

    cax = axes[:, 1]

    ax = cax[0]
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 0, 0]), "r--", label=r"$p_d$")
    ax.plot(data["t"], np.rad2deg(data["omega"][:, 0, 0]), "k-", label=r"$p$")
    ax.legend(loc="upper right")

    ax = cax[1]
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 1, 0]), "r--", label=r"$q_d$")
    ax.plot(data["t"], np.rad2deg(data["omega"][:, 1, 0]), "k-", label=r"$q$")
    ax.legend(loc="upper right")

    ax = cax[2]
    ax.plot(data["t"], np.rad2deg(data["omegad"][:, 2, 0]), "r--", label=r"$r_d$")
    ax.plot(data["t"], np.rad2deg(data["omega"][:, 2, 0]), "k-", label=r"$r$")
    ax.legend(loc="upper right")

    for ax in cax:
        ax.set_ylim(-90, 90)

    ax = cax[-1]
    ax.set_xlabel("Time, sec")

    cax = axes[:, 2]

    ax = cax[0]
    ax.plot(
        data["t"],
        np.rad2deg(data["omega"] - data["omegad"])[:, 0, 0],
        "k-",
        label=r"$e_p$",
    )
    ax.legend(loc="upper right")

    ax = cax[1]
    ax.plot(
        data["t"],
        np.rad2deg(data["omega"] - data["omegad"])[:, 1, 0],
        "k-",
        label=r"$e_q$",
    )
    ax.legend(loc="upper right")

    ax = cax[2]
    ax.plot(
        data["t"],
        np.rad2deg(data["omega"] - data["omegad"])[:, 2, 0],
        "k-",
        label=r"$e_r$",
    )
    ax.legend(loc="upper right")

    for ax in cax:
        ax.set_ylim(-90, 90)

    ax = cax[-1]
    ax.set_xlabel("Time, sec")

    fig.tight_layout()

    """ Plot 3 """

    fig, axes = plt.subplots(4, 2, figsize=(8, 6), squeeze=False, sharex=True)

    cax = axes[:, 0]

    ax = cax[0]
    ax.plot(data["t"], data["uc"][:, 0, 0], "r--", label=r"$f_{1 c}$")
    ax.plot(data["t"], data["u"][:, 0, 0], "k-", label=r"$f_1$")
    ax.legend(loc="upper right")

    ax = cax[1]
    ax.plot(data["t"], data["uc"][:, 1, 0], "r--", label=r"$f_{2 c}$")
    ax.plot(data["t"], data["u"][:, 1, 0], "k-", label=r"$f_2$")
    ax.legend(loc="upper right")

    ax = cax[2]
    ax.plot(data["t"], data["uc"][:, 2, 0], "r--", label=r"$f_{3 c}$")
    ax.plot(data["t"], data["u"][:, 2, 0], "k-", label=r"$f_3$")
    ax.legend(loc="upper right")

    ax = cax[3]
    ax.plot(data["t"], data["uc"][:, 3, 0], "r--", label=r"$f_{4 c}$")
    ax.plot(data["t"], data["u"][:, 3, 0], "k-", label=r"$f_4$")
    ax.legend(loc="upper right")

    for ax in cax:
        ax.set_ylim(900, 2100)

    ax = cax[-1]
    ax.set_xlabel("Time, sec")

    cax = axes[:, 1]

    ax = cax[0]
    ax.plot(data["t"], data["Lambda"][:, 0, 0], "k-", label=r"$\lambda_1$")
    ax.legend(loc="upper right")

    ax = cax[1]
    ax.plot(data["t"], data["Lambda"][:, 1, 0], "k-", label=r"$\lambda_2$")
    ax.legend(loc="upper right")

    ax = cax[2]
    ax.plot(data["t"], data["Lambda"][:, 2, 0], "k-", label=r"$\lambda_3$")
    ax.legend(loc="upper right")

    ax = cax[3]
    ax.plot(data["t"], data["Lambda"][:, 3, 0], "k-", label=r"$\lambda_4$")
    ax.legend(loc="upper right")

    for ax in cax:
        ax.set_ylim(-0.1, 1.1)

    ax = cax[-1]
    ax.set_xlabel("Time, sec")

    fig.tight_layout()

    """ Plot 4 - 3D Trajectory """

    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    idx = np.abs(data["t"] - tf).argmin()
    x = data["pos"][:idx, 0, 0]
    y = data["pos"][:idx, 1, 0]
    h = -data["pos"][:idx, 2, 0]

    xd = data["posd"][:idx, 0, 0]
    yd = data["posd"][:idx, 1, 0]
    hd = -data["posd"][:idx, 2, 0]

    ax.plot(xd, yd, hd, "b-")
    ax.plot(x, y, h, "r--")

    ax.set_xlim(-1, 6)
    ax.set_ylim(-1, 6)
    ax.set_zlim(-3, 3)
    limits = np.array([getattr(ax, f"get_{axis}lim")() for axis in "xyz"])
    ax.set_box_aspect(np.ptp(limits, axis=1))

    plt.show()


@click.group()
def main():
    pass


@main.command()
@click.option("-p", "--plot", is_flag=True)
@click.option("-P", "--only-plot", is_flag=True)
@click.option("-t", "--max-t", type=float, default=20)
@click.option("--dt", type=float, default=0.01)
@click.option("-b", "--break", is_flag=True)
@click.option("--scenario-index", type=int, default=0)
def run(**kwargs):
    if kwargs["only_plot"]:
        path = Path("data.h5")
        selected_plots(path, show=True)
        return

    env_config = {
        "fkw": {
            "max_t": kwargs["max_t"],
            "dt": kwargs["dt"],
        },
        "scenario_config": {"index": kwargs["scenario_index"]},
        "break": kwargs["break"],
    }
    path = _run(env_config)

    if kwargs["plot"]:
        selected_plots(path, show=True)


if __name__ == "__main__":
    main()
