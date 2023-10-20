import argparse
from pathlib import Path

import fym
import matplotlib.pyplot as plt
import numdifftools as nd
import numpy as np
from fym.utils.rot import angle2quat

import ftc
from ftc.mfa import MFA
from ftc.models.LC62 import LC62
from ftc.sim_parallel import evaluate_mfa_success_rate, sim_parallel
from ftc.utils import safeupdate

np.seterr(all="raise")


def shrink(u_min, u_max, scaling_factor=1.0):
    mean = (u_min + u_max) / 2
    width = (u_max - u_min) / 2
    u_min = mean - scaling_factor * width
    u_max = mean + scaling_factor * width
    return u_min, u_max


class Env(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 10,
        },
    }

    def __init__(self, initial, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        pos, vel, angle, omega = initial
        quat = angle2quat(*angle.ravel()[::-1])
        plant_init = {
            "init": {
                "pos": pos,
                "vel": vel,
                "quat": quat,
                "omega": omega,
            },
        }
        self.plant = LC62(plant_init)
        self.controller = ftc.make("INDI", self)

        self.posd = lambda t: np.vstack((0, 0, 0))
        self.posd_dot = nd.Derivative(self.posd, n=1)

        pwm_min, pwm_max = self.plant.control_limits["pwm"]
        self.mfa = MFA(
            pwm_min * np.ones(6),
            pwm_max * np.ones(6),
            predictor=ftc.make("Flat", self),
            distribute=self.distribute,
            is_success=lambda polynus: all(
                polytope.contains(nu) for polytope, nu in polynus
            ),
        )

        self.u0 = self.controller.get_u0(self)

        dx1, dx2, dx3 = self.plant.dx1, self.plant.dx2, self.plant.dx3
        dy1, dy2 = self.plant.dy1, self.plant.dy2
        c, self.c_th = 0.0338, 128  # tq / th, th / rcmds
        self.B_r2f = np.array(
            (
                [-1, -1, -1, -1, -1, -1],
                [-dy2, dy1, dy1, -dy2, -dy2, dy1],
                [-dx2, -dx2, dx1, -dx3, dx1, -dx3],
                [-c, c, -c, c, c, -c],
            )
        )

    def distribute(self, t, state, pwms_rotor):
        nu = self.B_r2f @ (pwms_rotor - 1000) / 1000 * self.c_th
        return nu

    def step(self, mfa_predict_prev, Lambda_prev):
        t = self.clock.get()

        Lambda = self.get_Lambda(t)
        if np.allclose(Lambda, Lambda_prev):
            mfa_predict = mfa_predict_prev
        else:
            lmbd = Lambda[:6]
            tspan = self.clock.tspan
            tspan = tspan[tspan >= t][::20]
            loe = lambda u_min, u_max: (
                lmbd * (u_min - 1000) + 1000,
                lmbd * (u_max - 1000) + 1000,
            )
            mfa_predict = self.mfa.predict(tspan, [loe, shrink])

        env_info, done = self.update()

        return done, env_info | {"mfa": mfa_predict}

    def observation(self):
        return self.observe_flat()

    def psid(self, t):
        return 0

    def get_ref(self, t, *args):
        refs = {
            "posd": self.posd(t),
            "posd_dot": self.posd_dot(t),
        }
        return [refs[key] for key in args]

    def set_dot(self, t):
        ctrls0, controller_info = self.controller.get_control(t, self)
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

        Lambda = np.ones(11)
        if t >= 3:
            Lambda[1] = 0.3
        return Lambda

    def set_Lambda(self, t, ctrls):
        Lambda = self.get_Lambda(t)
        ctrls[:6] = np.diag(Lambda[:6]) @ (ctrls[:6] - 1000) + 1000
        return ctrls


def sim(i, initial, Env, dirpath="data"):
    loggerpath = Path(dirpath, f"env_{i:04d}.h5")
    env = Env(initial)
    flogger = fym.Logger(loggerpath)

    env.reset()

    # initialization
    Lambda_prev = env.get_Lambda(env.clock.get())
    mfa_predict_prev = True
    while True:
        env.render(mode=None)

        done, env_info = env.step(mfa_predict_prev, Lambda_prev)
        flogger.record(env=env_info, initial=initial)

        Lambda_prev = env_info["Lambda"]
        mfa_predict_prev = env_info["mfa"]

        if done:
            break

    flogger.close()


def parsim(N=1, seed=0):
    np.random.seed(seed)
    pos = np.random.uniform(0, 0, size=(N, 3, 1))
    vel = np.random.uniform(0, 0, size=(N, 3, 1))
    angle = np.random.uniform(*np.deg2rad((-5, 5)), size=(N, 3, 1))
    omega = np.random.uniform(*np.deg2rad((-1, 1)), size=(N, 3, 1))

    initials = np.stack((pos, vel, angle, omega), axis=1)
    sim_parallel(sim, N, initials, Env)


def plot(i):
    loggerpath = Path("data", f"env_{i:04d}.h5")
    data = fym.load(loggerpath)["env"]

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

    ax = axes[1, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "k-")
    ax.set_ylabel(r"$q$, deg/s")

    ax = axes[2, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 2].squeeze(-1)), "k-")
    ax.set_ylabel(r"$r$, deg/s")

    ax.set_xlabel("Time, sec")

    plt.tight_layout()
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
    ax.legend(["Response", "Command"], loc="upper right")

    ax = axes[1, 1]
    ax.plot(data["t"], data["FM"][:, 4].squeeze(-1), "k-")
    ax.plot(data["t"], data["FM"][:, 4].squeeze(-1), "r--")
    ax.set_ylabel(r"$M_y$")

    ax = axes[2, 1]
    ax.plot(data["t"], data["FM"][:, 5].squeeze(-1), "k-")
    ax.plot(data["t"], data["FM"][:, 5].squeeze(-1), "r--")
    ax.set_ylabel(r"$M_z$")

    ax.set_xlabel("Time, sec")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axes)

    """ Figure 3 - Rotor thrusts """
    fig, axs = plt.subplots(3, 2, sharex=True)
    ylabels = np.array(
        (["Rotor 1", "Rotor 2"], ["Rotor 3", "Rotor 4"], ["Rotor 5", "Rotor 6"])
    )
    for i, _ylabel in np.ndenumerate(ylabels):
        ax = axs[i]
        ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, sum(i)], "k-", label="Response")
        ax.plot(
            data["t"], data["ctrls0"].squeeze(-1)[:, sum(i)], "r--", label="Command"
        )
        ax.grid()
        if i == (0, 1):
            ax.legend(loc="upper right")
        plt.setp(ax, ylabel=_ylabel)
        ax.set_ylim([1000 - 5, 2000 + 5])
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Rotor Thrusts")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axs)

    """ Figure 4 - Pusher and Control surfaces """
    fig, axs = plt.subplots(5, 1, sharex=True)
    ylabels = np.array(
        ("Pusher 1", "Pusher 2", r"$\delta_a$", r"$\delta_e$", r"$\delta_r$")
    )
    for i, _ylabel in enumerate(ylabels):
        ax = axs[i]
        ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, i + 6], "k-", label="Response")
        ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, i + 6], "r--", label="Command")
        ax.grid()
        plt.setp(ax, ylabel=_ylabel)
        # if i < 2:
        #     ax.set_ylim([1000-5, 2000+5])
        if i == 0:
            ax.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Pusher and Control Surfaces")

    plt.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axs)

    plt.show()


def main(args, N, seed, i):
    if args.only_plot:
        plot(i)
        return
    else:
        parsim(N, seed)
        evaluate_mfa_success_rate(
            N, time_from=2, error_type=None, threshold=np.ones(3), verbose=True
        )

        if args.plot:
            plot(i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-P", "--only-plot", action="store_true")
    args = parser.parse_args()
    main(args, N=10, seed=0, i=0)
