from ray import tune
import os
import json
from ray.air import CheckpointConfig, RunConfig
from ray.tune.search.hyperopt import HyperOptSearch
from ray.tune import CLIReporter

import numpy as np
import matplotlib.pyplot as plt
import argparse

import fym
from fym.utils.rot import angle2quat, quat2angle

import ftc
from ftc.utils import safeupdate
from ftc.models.LC62 import LC62


class ActuatorDynamics(fym.BaseSystem):
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def set_dot(self, ctrls, ctrls_cmd):
        self.dot = -1 / self.tau * (ctrls - ctrls_cmd)


class MyEnv(fym.BaseEnv):
    euler = np.random.uniform(-np.deg2rad(5), np.deg2rad(5), size=(3, 1))
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
                # "pos": np.random.uniform(-1, 1, size=(3, 1)),
                # "vel": np.random.uniform(-1, 1, size=(3, 1)),
                # "quat": angle2quat(euler[0], euler[1], euler[2]),
                # "omega": np.random.uniform(-np.deg2rad(5), np.deg2rad(5), size=(3, 1)),
            },
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = LC62(env_config["plant"])
        self.env_config = env_config
        self.controller = ftc.make("BLF-LC62", self)
        self.fdi_delay = 0.1
        # self.rotor_dyn = ActuatorDynamics(tau=0.01, shape=(11, 1))

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_ref(self, t, *args):
        posd = np.vstack([np.sin(t), np.cos(t), -t])
        posd_dot = np.vstack([np.cos(t), -np.sin(t), -1])
        refs = {"posd": posd, "posd_dot": posd_dot}
        return [refs[key] for key in args]

    def set_dot(self, t):
        ctr_forces, ctrls0, controller_info = self.controller.get_control(t, self)
        ctrls = ctrls0

        """ set faults """
        Lambda = self.get_Lambda(t)
        lctrls = np.vstack([
            (Lambda[:, None] * (ctrls[0:6] - 1000) / 1000) * 1000 + 1000,
            ctrls[6:11]
        ])

        FM = self.plant.get_FM(*self.plant.observe_list(), lctrls)
        self.plant.set_dot(t, FM)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "ctrls0": ctrls0,
            "ctrls": ctrls,
            "lctrls": lctrls,
            "FM": FM,
            "Lambda": self.get_Lambda(t),
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


def run(config):
    env = MyEnv(config)
    flogger = fym.Logger("data.h5")

    env.reset()
    # try:
    while True:
        env.render()

        done, env_info = env.step()
        flogger.record(env=env_info)

        if done:
            break

    # finally:
    flogger.close()
    plot()


def plot():
    data = fym.load("data.h5")["env"]

    """ Figure 1 - States """
    fig, axes = plt.subplots(3, 4, figsize=(18, 5), squeeze=False, sharex=True)

    """ Column 1 - States: Position """
    ax = axes[0, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1)-data["posd"][:, 0].squeeze(-1), "k-")
    ax.plot(data["t"], data["obs_pos"][:, 0].squeeze(-1), "b--")
    ax.plot(data["t"], data["bound_err"], "r:")
    ax.plot(data["t"], -data["bound_err"], "r:")
    # ax.plot(data["t"], data["plant"]["pos"][:, 0].squeeze(-1), "k-")
    # ax.plot(data["t"], data["posd"][:, 0].squeeze(-1), "r--")
    # ax.plot(data["t"], data["obs_pos"][:, 0].squeeze(-1)+data["posd"][:, 0].squeeze(-1), "b--")
    ax.set_ylabel(r"$x$, m")
    ax.legend(["Response", "Command", "Estimation"], loc="upper right")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1)-data["posd"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["obs_pos"][:, 1].squeeze(-1), "b--")
    ax.plot(data["t"], data["bound_err"], "r:")
    ax.plot(data["t"], -data["bound_err"], "r:")
    # ax.plot(data["t"], data["plant"]["pos"][:, 1].squeeze(-1), "k-")
    # ax.plot(data["t"], data["posd"][:, 1].squeeze(-1), "r--")
    # ax.plot(data["t"], data["obs_pos"][:, 1].squeeze(-1)+data["posd"][:, 1].squeeze(-1), "b--")
    ax.set_ylabel(r"$y$, m")

    ax = axes[2, 0]
    ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1)-data["posd"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["obs_pos"][:, 2].squeeze(-1), "b--")
    ax.plot(data["t"], data["bound_err"], "r:")
    ax.plot(data["t"], -data["bound_err"], "r:")
    # ax.plot(data["t"], data["plant"]["pos"][:, 2].squeeze(-1), "k-")
    # ax.plot(data["t"], data["posd"][:, 2].squeeze(-1), "r--")
    # ax.plot(data["t"], data["obs_pos"][:, 2].squeeze(-1)+data["posd"][:, 2].squeeze(-1), "b--")
    ax.set_ylabel(r"$z$, m")

    ax.set_xlabel("Time, sec")

    """ Column 2 - States: Velocity """
    ax = axes[0, 1]
    ax.plot(data["t"], data["plant"]["vel"][:, 0].squeeze(-1), "k-")
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
    ax.plot(data["t"], np.rad2deg(data["obs_ang"][:, 0].squeeze(-1)), "b--")
    ax.plot(data["t"], np.rad2deg(data["bound_ang"][:, 0]), "r:")
    ax.plot(data["t"], np.rad2deg(-data["bound_ang"][:, 0]), "r:")
    ax.set_ylabel(r"$\phi$, deg")

    ax = axes[1, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 1].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 1].squeeze(-1)), "r--")
    ax.plot(data["t"], np.rad2deg(data["obs_ang"][:, 1].squeeze(-1)), "b--")
    ax.plot(data["t"], np.rad2deg(data["bound_ang"][:, 0]), "r:")
    ax.plot(data["t"], np.rad2deg(-data["bound_ang"][:, 0]), "r:")
    ax.set_ylabel(r"$\theta$, deg")

    ax = axes[2, 2]
    ax.plot(data["t"], np.rad2deg(data["ang"][:, 2].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["angd"][:, 2].squeeze(-1)), "r--")
    ax.plot(data["t"], np.rad2deg(data["obs_ang"][:, 2].squeeze(-1)), "b--")
    ax.plot(data["t"], np.rad2deg(data["bound_ang"][:, 0]), "r:")
    ax.plot(data["t"], np.rad2deg(-data["bound_ang"][:, 0]), "r:")
    ax.set_ylabel(r"$\psi$, deg")

    ax.set_xlabel("Time, sec")

    """ Column 4 - States: Angular rates """
    ax = axes[0, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 0].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["bound_ang"][:, 1]), "r:")
    ax.plot(data["t"], np.rad2deg(-data["bound_ang"][:, 1]), "r:")
    ax.set_ylabel(r"$p$, deg/s")
    ax.legend(["Response", "Ref"], loc="upper right")

    ax = axes[1, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 1].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["bound_ang"][:, 1]), "r:")
    ax.plot(data["t"], np.rad2deg(-data["bound_ang"][:, 1]), "r:")
    ax.set_ylabel(r"$q$, deg/s")

    ax = axes[2, 3]
    ax.plot(data["t"], np.rad2deg(data["plant"]["omega"][:, 2].squeeze(-1)), "k-")
    ax.plot(data["t"], np.rad2deg(data["bound_ang"][:, 1]), "r:")
    ax.plot(data["t"], np.rad2deg(-data["bound_ang"][:, 1]), "r:")
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
    ax.plot(data["t"], data["ctr_forces"][:, 0].squeeze(-1), "r--")
    ax.set_ylabel(r"$F_x$")
    ax.legend(["Response"], loc="upper right")
    ax.set_xlim(data["t"][0], data["t"][-1])

    ax = axes[1, 0]
    ax.plot(data["t"], data["FM"][:, 1].squeeze(-1), "k-")
    ax.plot(data["t"], data["ctr_forces"][:, 1].squeeze(-1), "r--")
    ax.set_ylabel(r"$F_y$")

    ax = axes[2, 0]
    ax.plot(data["t"], data["FM"][:, 2].squeeze(-1), "k-")
    ax.plot(data["t"], data["ctr_forces"][:, 2].squeeze(-1), "r--")
    ax.set_ylabel(r"$F_z$")

    ax.set_xlabel("Time, sec")

    """ Column 2 - Generalized forces: Moments """
    ax = axes[0, 1]
    ax.plot(data["t"], data["FM"][:, 3].squeeze(-1), "k-")
    ax.plot(data["t"], data["ctr_forces"][:, 3].squeeze(-1), "r--")
    ax.set_ylabel(r"$M_x$")
    ax.legend(["Response"], loc="upper right")

    ax = axes[1, 1]
    ax.plot(data["t"], data["FM"][:, 4].squeeze(-1), "k-")
    ax.plot(data["t"], data["ctr_forces"][:, 4].squeeze(-1), "r--")
    ax.set_ylabel(r"$M_y$")

    ax = axes[2, 1]
    ax.plot(data["t"], data["FM"][:, 5].squeeze(-1), "k-")
    ax.plot(data["t"], data["ctr_forces"][:, 5].squeeze(-1), "r--")
    ax.set_ylabel(r"$M_z$")

    ax.set_xlabel("Time, sec")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.5)
    fig.align_ylabels(axes)

    """ Figure 3 - Rotor thrusts """
    plt.figure()

    ax = plt.subplot(321)
    for i in range(6):
        if i != 0:
            plt.subplot(321+i, sharex=ax)
        plt.ylim([1000-5, 2000+5])
        plt.plot(data["t"], data["lctrls"].squeeze(-1)[:, i], "k-", label="Response")
        plt.plot(data["t"], data["ctrls"].squeeze(-1)[:, i], "r--", label="Command")
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Rotor Thrusts")
    plt.tight_layout()

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

    # dist
    plt.figure()

    ax = plt.subplot(611)
    for i, _label in enumerate([r"$d_x$", r"$d_y$", r"$d_z$",
                                r"$d_\phi$", r"$d_\theta$", r"$d_\psi$"]):
        if i != 0:
            plt.subplot(611+i, sharex=ax)
        plt.plot(data["t"], data["dist"][:, i, 0], "k", label=" distarbance")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supylabel("dist")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

    plt.show()


def main(args):
    if args.only_plot:
        plot()
        return
    elif args.with_ray:
        def objective(config):
            np.seterr(all="raise")

            env = MyEnv(config)

            env.reset()
            tf = 0
            try:
                while True:
                    done, env_info = env.step()
                    tf = env_info["t"]

                    if done:
                        break

            finally:
                return {"tf": tf}

        config = {
            "k11": 2/300,
            "k12": 0.1,
            "k13": 0,
            "k21": tune.uniform(0.000001, 100),
            "k22": tune.uniform(0.000001, 100),
            "k23": 0,
            "k31": tune.uniform(0.1, 100),
            "k32": tune.uniform(0.1, 100),
            "k33": 0,
            "k41": 500/40,
            "k42": 40,
            "k43": 0,
            "eps11": 2,
            "eps12": 2,
            "eps13": 25,
            "eps21": 25,
            "eps22": 25,
            "eps23": 25,
        }
        current_best_params = [{
            "k11": 2/300,
            "k12": 0.1,
            "k13": 0,
            "k21": 0,
            "k22": 0,
            "k23": 0,
            "k31": 0.8,
            "k32": 10,
            "k33": 0,
            "k41": 500/40,
            "k42": 40,
            "k43": 0,
            "eps11": 1,
            "eps12": 1,
            "eps13": 25,
            "eps21": 25,
            "eps22": 25,
            "eps23": 25,
        }]
        search = HyperOptSearch(
            metric="tf",
            mode="max",
            points_to_evaluate=current_best_params,
        )
        tuner = tune.Tuner(
            tune.with_resources(
                objective,
                resources={"cpu": os.cpu_count()},
            ),
            param_space=config,
            tune_config=tune.TuneConfig(
                num_samples=1000,
                search_alg=search,
            ),
            run_config=RunConfig(
                name="train_run",
                local_dir="data/ray_results",
                verbose=1,
                progress_reporter=CLIReporter(
                    parameter_columns=list(config.keys())[:3],
                    max_progress_rows=3,
                    metric="tf",
                    mode="max",
                    sort_by_metric=True,
                ),
                checkpoint_config=CheckpointConfig(
                    num_to_keep=5,
                    checkpoint_score_attribute="tf",
                    checkpoint_score_order="max",
                ),
            ),
        )
        results = tuner.fit()
        config = results.get_best_result(metric="tf", mode="max").config
        with open("data/ray_results/train_run/best_config.json", "w") as f:
            json.dump(config, f)  # json file은 cat cmd로 볼 수 있다
        return
    else:
        params = {
            "k01": 1,
            "k02": 0.1,
            "k03": 0,
            "k11": 1.3,
            "k12": 0.1,
            "k13": 0,
            "k51": 1,
            "k52": 0.05,
            "k53": 0,
            "k21": 500/40,
            "k22": 40,
            "k23": 0,
            "k31": 500/40,
            "k32": 40,
            "k33": 0,
            "k41": 500/40,
            "k42": 40,
            "k43": 0,
            "eps11": 5,
            "eps12": 5,
            "eps13": 20,
            "eps21": 30,
            "eps22": 30,
            "eps23": 30,
        }
        run(params)

        if args.plot:
            plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-P", "--only-plot", action="store_true")
    parser.add_argument("-r", "--with-ray", action="store_true")
    args = parser.parse_args()
    main(args)
