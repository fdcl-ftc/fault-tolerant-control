import argparse

import fym
import matplotlib.pyplot as plt
import numpy as np
from fym.utils.rot import hat
from scipy.spatial.transform import Rotation

import ftc
from ftc.utils import safeupdate

np.seterr(all="raise")


def cross(x, y):
    return np.cross(x, y, axis=0)


class Quad(fym.BaseEnv):
    """Quadrotor model
    - Target: DJI M100
    - Reference:
        Shen, Z., Li, F., Cao, X., & Guo, C. (2021). Prescribed performance dynamic surface control for trajectory tracking of quadrotor UAV with uncertainties and input constraints. International Journal of Control, 94(11), 2945–2955. https://doi.org/10.1080/00207179.2020.1743366
    - Reference for kappa and tau:
        Chen, F., Jiang, R., Zhang, K., Jiang, B., & Tao, G. (2016). Robust Backstepping Sliding Mode Control and Observer-based Fault Estimation for a Quadrotor UAV. IEEE Transactions on Industrial Electronics, 1–1. https://doi.org/10.1109/TIE.2016.2552151

    """

    # Actual paramters
    m = 2.5
    g = 9.8
    l = 0.325
    J = np.diag([0.082, 0.082, 0.149])
    Jinv = np.linalg.inv(J)
    Kv = np.diag([0.17, 0.17, 0.17])
    Komega = np.diag([0.17, 0.17, 0.17])
    rfmax = 14
    kappa = 2.98e-6
    tau = 1.14e-7
    B = np.array(
        [
            [1, 1, 1, 1],
            [0, -l, 0, l],
            [l, 0, -l, 0],
            [-tau / kappa, tau / kappa, -tau / kappa, tau / kappa],
        ]
    )

    ENV_CONFIG = {
        "init": {
            "pos": np.zeros((3, 1)),
            "vel": np.zeros((3, 1)),
            "R": np.eye(3),
            "omega": np.zeros((3, 1)),
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__()
        self.pos = fym.BaseSystem(env_config["init"]["pos"])
        self.vel = fym.BaseSystem(env_config["init"]["vel"])
        self.R = fym.BaseSystem(env_config["init"]["R"])
        self.omega = fym.BaseSystem(env_config["init"]["omega"])

        self.e3 = np.vstack((0, 0, 1))

    def set_dot(self, t, rfs0):
        """
        Parameters:
            rfs0: rotor forces
        """
        _, vel, R, omega = self.observe_list()

        brfs = self.saturate(rfs0)  # bounded rfs0
        lrfs = self.set_Lambda(t, brfs)  # lambda * brfs
        rfs = self.saturate(lrfs)
        fTM = self.B @ rfs  # total thrust and moments
        fT, M = fTM[0:1], fTM[1:]

        """ state dependent disturbances """
        Omega_v = -self.Kv @ vel
        Omega_omega = self.Jinv @ (-np.linalg.norm(omega) * self.Komega @ omega)

        """ state independent disturbances """
        dv = np.zeros((3, 1))
        domega = self.Jinv @ np.zeros((3, 1))

        """ dynamics """
        self.pos.dot = vel
        self.vel.dot = self.g * self.e3 - 1 / self.m * R @ self.e3 * fT + Omega_v + dv
        self.R.dot = R @ hat(omega)
        self.omega.dot = (
            self.Jinv @ (M - cross(omega, self.J @ omega)) + Omega_omega + domega
        )

        quad_info = {
            "rfs": rfs,
            "rfs0": rfs0,
            "brfs": brfs,
            "lrfs": lrfs,
            "fT": fT,
            "M": M,
            "Lambda": self.get_Lambda(t),
        }
        return quad_info

    def saturate(self, rfs):
        """Saturation function"""
        return np.clip(rfs, 0, self.rfmax)

    def get_Lambda(self, t):
        """Lambda function"""

        _Lambda = np.vstack([0.05, 0.9, 0.9, 1])  # final Lambda

        if t < (t1 := 12):
            Lambda = np.ones((4, 1))
        elif t < (t2 := 24):
            Lf0 = np.ones((4, 1))
            Lf = np.vstack([_Lambda[0], 1, 1, 1])
            Lambda = (Lf0 - 1 * Lf) * ((t2 - t) / (t2 - t1)) ** 2 + Lf
        elif t < (t3 := 36):
            Lf0 = np.vstack([_Lambda[0], 1, 1, 1])
            Lf = np.vstack([_Lambda[0], _Lambda[1], 1, 1])
            Lambda = (Lf0 - 1 * Lf) * ((t3 - t) / (t3 - t2)) ** 2 + Lf
        elif t < (t4 := 48):
            Lf0 = np.vstack([_Lambda[0], _Lambda[1], 1, 1])
            Lf = np.vstack([_Lambda[0], _Lambda[1], _Lambda[2], 1])
            Lambda = (Lf0 - 1 * Lf) * ((t4 - t) / (t4 - t3)) ** 2 + Lf
        else:
            Lambda = _Lambda

        return Lambda

    def set_Lambda(self, t, brfs):
        Lambda = self.get_Lambda(t)
        return Lambda * brfs - 0.0 * (np.sin(2 * np.pi * 1 * t) + 1)

    def get_angles(self, R):
        return Rotation.from_matrix(R).as_euler("ZYX")[::-1]


class ExtendedQuadEnv(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 60,
        },
        "quad": {
            "init": {
                "pos": np.vstack((0.5, 0.5, 0.0)),
                "vel": np.zeros((3, 1)),
                "R": np.eye(3),
                "omega": np.zeros((3, 1)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        # quad
        self.plant = Quad(env_config["quad"])
        # controller
        self.controller = ftc.make("PPC-DSC", self)

    def step(self, action):
        obs = self.observation()

        env_info, done = self.update()

        next_obs = self.observation()
        reward = self.get_reward(obs, action, next_obs)
        return next_obs, reward, done, env_info

    def observation(self):
        return self.observe_flat()

    def get_reward(self, obs, action, next_obs):
        return 0

    def get_ref(self, t, *args):
        # a = 0.3
        # b = 1
        # posd = np.vstack((5, 5, 0)) + b * np.vstack((sin(a * t), 1 - cos(a * t), 0))
        # posd_dot = b * np.vstack((a * cos(a * t), a * sin(a * t), 0))

        # posd = np.vstack((5.0, 5.0, -0.5))
        # posd_dot = np.vstack((0.0, 0.0, 0.0))

        posd = np.vstack(
            (
                1 * (np.tanh(t - 10) - np.tanh(t - 30)) / 2,
                1 * (np.tanh(t - 20) - np.tanh(t - 40)) / 2,
                -0.5,
            )
        )
        posd_dot = np.vstack(
            (
                1 * ((1 - np.tanh(t - 10) ** 2) - (1 - np.tanh(t - 30) ** 2)) / 2,
                1 * ((1 - np.tanh(t - 20) ** 2) - (1 - np.tanh(t - 40) ** 2)) / 2,
                0,
            )
        )
        refs = {"posd": posd, "posd_dot": posd_dot}
        return [refs[key] for key in args]

    def set_dot(self, t):
        rfs0, controller_info = self.controller.get_control(t, self)
        # quad
        quad_info = self.plant.set_dot(t, rfs0)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **quad_info,
            **controller_info,
        }

        return env_info


class Agent:
    def get_action(self, obs):
        return obs, {}


def run():
    env = ExtendedQuadEnv()
    agent = Agent()
    flogger = fym.Logger("data.h5")

    obs = env.reset()
    try:
        while True:
            env.render()

            action, agent_info = agent.get_action(obs)

            next_obs, reward, done, env_info = env.step(action=action)
            flogger.record(reward=reward, env=env_info, agent=agent_info)

            if done:
                break

            obs = next_obs

    finally:
        flogger.close()
        plot()


def plot():
    data = fym.load("data.h5")["env"]

    fig, axes = plt.subplots(4, 4, figsize=(18, 5.8), squeeze=False, sharex=True)

    """ Column 1 - States """

    ax = axes[0, 0]
    ax.plot(data["t"], data["ep"].squeeze(-1))
    ax.plot(data["t"], (data["ppc_min"] - data["posd"]).squeeze(-1), "r--")
    ax.plot(data["t"], (data["ppc_max"] - data["posd"]).squeeze(-1), "r--")
    # ax.axhline(-2, color="r", ls="--")
    ax.set_ylabel("Position, m")
    ax.legend([r"$x$", r"$y$", r"$z$", "Bounds"], loc="upper right")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["vel"].squeeze(-1))
    ax.set_ylabel("Velocity, m/s")
    ax.legend([r"$v_x$", r"$v_y$", r"$v_z$"])

    ax = axes[2, 0]
    angles = Rotation.from_matrix(data["plant"]["R"]).as_euler("ZYX")[:, ::-1]
    ax.plot(data["t"], np.rad2deg(angles))
    ax.set_ylabel("Angles, deg")
    ax.legend([r"$\phi$", r"$\theta$", r"$\psi$"])

    ax = axes[3, 0]
    ax.plot(data["t"], data["plant"]["omega"].squeeze(-1))
    ax.set_ylabel("Angular velocity, rad/s")
    ax.legend([r"$p$", r"$q$", r"$r$"])

    ax.set_xlabel("Time, sec")

    """ Column 2 - Rotor forces """

    ax = axes[0, 1]
    ax.plot(data["t"], data["rfs0"].squeeze(-1)[:, 0], "r--")
    ax.plot(data["t"], data["rfs"].squeeze(-1)[:, 0], "k-")
    ax.set_ylabel("Rotor 1 thrust, N")

    ax = axes[1, 1]
    ax.plot(data["t"], data["rfs0"].squeeze(-1)[:, 1], "r--")
    ax.plot(data["t"], data["rfs"].squeeze(-1)[:, 1], "k-")
    ax.set_ylabel("Rotor 2 thrust, N")

    ax = axes[2, 1]
    ax.plot(data["t"], data["rfs0"].squeeze(-1)[:, 2], "r--")
    ax.plot(data["t"], data["rfs"].squeeze(-1)[:, 2], "k-")
    ax.set_ylabel("Rotor 3 thrust, N")

    ax = axes[3, 1]
    ax.plot(data["t"], data["rfs0"].squeeze(-1)[:, 3], "r--")
    ax.plot(data["t"], data["rfs"].squeeze(-1)[:, 3], "k-")
    ax.set_ylabel("Rotor 4 thrust, N")
    ax.legend(["Command"])

    ax.set_xlabel("Time, sec")
    for ax in axes[:, 1]:
        ax.set_ylim(-1, 15)

    """ Column 3 - Faults """

    ax = axes[0, 2]
    ax.plot(data["t"], data["Lambda"].squeeze(-1)[:, 0], "k")
    ax.set_ylabel("Lambda 1")

    ax = axes[1, 2]
    ax.plot(data["t"], data["Lambda"].squeeze(-1)[:, 1], "k")
    ax.set_ylabel("Lambda 2")

    ax = axes[2, 2]
    ax.plot(data["t"], data["Lambda"].squeeze(-1)[:, 2], "k")
    ax.set_ylabel("Lambda 3")

    ax = axes[3, 2]
    ax.plot(data["t"], data["Lambda"].squeeze(-1)[:, 3], "k")
    ax.set_ylabel("Lambda 4")

    """ Column 4 - Faults """

    ax = axes[0, 3]
    ax.plot(data["t"], data["zp"].squeeze(-1))
    ax.set_ylabel(r"$z_p$")
    ax.legend([r"$z_1$", r"$z_2$", r"$z_3$"], loc="upper right")

    ax = axes[1, 3]
    ax.plot(data["t"], data["zv"].squeeze(-1))
    ax.set_ylabel(r"$z_v$")
    ax.legend([r"$z_{v 1}$", r"$z_{v 2}$", r"$z_{v 3}$"], loc="upper right")

    ax = axes[2, 3]
    ax.plot(data["t"], data["zXi"].squeeze(-1))
    ax.set_ylabel(r"$z_\Xi$")
    ax.legend([r"$z_{\Xi 1}$", r"$z_{\Xi 2}$"], loc="upper right")

    ax = axes[3, 3]
    ax.plot(data["t"], data["zchi"].squeeze(-1))
    ax.set_ylabel(r"$z_\chi$")
    ax.legend(
        [r"$z_{\chi 1}$", r"$z_{\chi 2}$", r"$z_{\chi 3}$", r"$z_{\chi 4}$"],
        loc="upper right",
    )

    # """ Column 4 - Auxs """

    # ax = axes[0, 3]
    # ax.plot(data["t"], data["varphi"].squeeze(-1))
    # ax.set_ylabel(r"$\varphi$")

    # ax = axes[1, 3]
    # ax.plot(data["t"], data["eta"].squeeze(-1))
    # ax.set_ylabel(r"$\eta$")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.3)
    fig.align_ylabels(axes)

    """ FIGURE 2 """
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")

    x = data["plant"]["pos"][:, 0, 0]
    y = data["plant"]["pos"][:, 1, 0]
    h = -data["plant"]["pos"][:, 2, 0]

    xd = data["posd"][:, 0, 0]
    yd = data["posd"][:, 1, 0]
    hd = -data["posd"][:, 2, 0]

    ax.plot(xd, yd, hd, "b-")
    ax.plot(x, y, h, "r--")

    fig.tight_layout()

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
