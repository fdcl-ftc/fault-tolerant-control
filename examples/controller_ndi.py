from copy import deepcopy
from functools import reduce

import fym
import matplotlib.pyplot as plt
import numpy as np

import ftc
from ftc.models.LC62 import LC62

np.seterr(all="raise")


def safeupdate(*configs):
    assert len(configs) > 1

    def _merge(base, new):
        assert isinstance(base, dict), f"{base} is not a dict"
        assert isinstance(new, dict), f"{new} is not a dict"
        out = deepcopy(base)
        for k, v in new.items():
            # assert k in out, f"{k} not in {base}"
            if isinstance(v, dict):
                if "grid_search" in v:
                    out[k] = v
                else:
                    out[k] = _merge(out[k], v)
            else:
                out[k] = v

        return out

    return reduce(_merge, configs)


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
        self.controller = ftc.make("NDI")
        m, g = self.plant.m, self.plant.g
        self.trim_FM = np.vstack([0, 0, m * g, 0, 0, 0])

    def step(self):
        env_info, done = self.update()
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
        FM = self.plant.get_FM(self.plant.observe_list, ctrls)
        self.plant.set_dot(t, FM)
        F, M = FM[0:3], FM[3:]

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "ctrls0": ctrls0,
            "ctrls": ctrls,
            "F": F,
            "M": M,
            "Lambda": self.get_Lambda(t),
        }

        return env_info

    def get_Lambda(self, t):
        """Lambda function"""

        Lambda = np.eye(6)
        return Lambda

    def set_Lambda(self, t):
        Lambda = self.get_Lambda(t)
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

    fig, axes = plt.subplots(4, 4, figsize=(18, 5.8), squeeze=False, sharex=True)

    """ Column 1 - States """

    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"].squeeze(-1))
    ax.plot(data["t"], data["posd"].squeeze(-1), "r--")
    ax.set_ylabel("Position, m")
    ax.legend([r"$x$", r"$y$", r"$z$", "Bounds"], loc="upper right")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["vel"].squeeze(-1))
    ax.set_ylabel("Velocity, m/s")
    ax.legend([r"$v_x$", r"$v_y$", r"$v_z$"])

    ax = axes[2, 0]
    ax.plot(data["t"], np.rad2deg(data["plant"]["ang"].squeeze(-1)))
    ax.set_ylabel("Angles, deg")
    ax.legend([r"$\phi$", r"$\theta$", r"$\psi$"])

    ax = axes[3, 0]
    ax.plot(data["t"], data["plant"]["omega"].squeeze(-1))
    ax.set_ylabel("Angular velocity, rad/s")
    ax.legend([r"$p$", r"$q$", r"$r$"])

    ax.set_xlabel("Time, sec")

    """ Column 2 - Rotor forces """

    ax = axes[0, 1]
    ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, 0], "r--")
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 0], "k-")
    ax.set_ylabel("Rotor 1 thrust, N")

    ax = axes[1, 1]
    ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, 1], "r--")
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 1], "k-")
    ax.set_ylabel("Rotor 2 thrust, N")

    ax = axes[2, 1]
    ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, 2], "r--")
    ax.plot(data["t"], data["ctrl"].squeeze(-1)[:, 2], "k-")
    ax.set_ylabel("Rotor 3 thrust, N")

    ax = axes[3, 1]
    ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, 3], "r--")
    ax.plot(data["t"], data["ctrl"].squeeze(-1)[:, 3], "k-")
    ax.set_ylabel("Rotor 4 thrust, N")

    ax = axes[4, 1]
    ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, 4], "r--")
    ax.plot(data["t"], data["ctrl"].squeeze(-1)[:, 4], "k-")
    ax.set_ylabel("Rotor 5 thrust, N")

    ax = axes[5, 1]
    ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, 5], "r--")
    ax.plot(data["t"], data["ctrl"].squeeze(-1)[:, 5], "k-")
    ax.set_ylabel("Rotor 6 thrust, N")
    ax.legend(["Command"])

    ax.set_xlabel("Time, sec")
    for ax in axes[:, 1]:
        ax.set_ylim(-1, 15)

    """ Column 3 - Control surfaces """

    ax = axes[0, 2]
    ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, 6], "r--")
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 6], "k-")
    ax.set_ylabel("Pusher 1")

    ax = axes[1, 2]
    ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, 7], "r--")
    ax.plot(data["t"], data["ctrls"].squeeze(-1)[:, 7], "k-")
    ax.set_ylabel("Pusher 2")

    ax = axes[2, 2]
    ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, 8], "r--")
    ax.plot(data["t"], data["ctrl"].squeeze(-1)[:, 8], "k-")
    ax.set_ylabel("Aileron deflection, deg")

    ax = axes[3, 2]
    ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, 9], "r--")
    ax.plot(data["t"], data["ctrl"].squeeze(-1)[:, 9], "k-")
    ax.set_ylabel("Elevator deflection, deg")

    ax = axes[4, 2]
    ax.plot(data["t"], data["ctrls0"].squeeze(-1)[:, 10], "r--")
    ax.plot(data["t"], data["ctrl"].squeeze(-1)[:, 10], "k-")
    ax.set_ylabel("Rudder deflection, deg")
    ax.legend(["Command"])

    ax.set_xlabel("Time, sec")
    for ax in axes[:, 1]:
        ax.set_ylim(-1, 15)

#     """ Column 3 - Faults """

#     ax = axes[0, 2]
#     ax.plot(data["t"], data["Lambda"].squeeze(-1)[:, 0], "k")
#     ax.set_ylabel("Lambda 1")

#     ax = axes[1, 2]
#     ax.plot(data["t"], data["Lambda"].squeeze(-1)[:, 1], "k")
#     ax.set_ylabel("Lambda 2")

#     ax = axes[2, 2]
#     ax.plot(data["t"], data["Lambda"].squeeze(-1)[:, 2], "k")
#     ax.set_ylabel("Lambda 3")

#     ax = axes[3, 2]
#     ax.plot(data["t"], data["Lambda"].squeeze(-1)[:, 3], "k")
#     ax.set_ylabel("Lambda 4")

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


if __name__ == "__main__":
    run()
