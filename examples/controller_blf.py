import argparse
import json
import os
from copy import deepcopy
from functools import reduce

import fym
import matplotlib.pyplot as plt
import numpy as np
from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import angle2quat, dcm2quat, quat2angle, quat2dcm
from ray import tune
from ray.air import CheckpointConfig, RunConfig
from ray.tune.search.hyperopt import HyperOptSearch

import ftc
from ftc.utils import safeupdate

np.seterr(all="raise")


class Quad(BaseEnv):
    # Actual parameters
    g = 9.81
    m = 0.65
    d = 0.23
    c = 0.75 * 1e-6
    b = 0.0000313
    J = np.diag([0.0075, 0.0075, 0.013])
    Jinv = np.linalg.inv(J)
    rotor_min = 0
    rotor_max = 1e6
    B = np.array(
        [[b, b, b, b], [0, -b * d, 0, b * d], [b * d, 0, -b * d, 0], [-c, c, -c, c]]
    )
    fault_delay = 0.1

    ENV_CONFIG = {
        "init": {
            "pos": np.zeros((3, 1)),
            "vel": np.zeros((3, 1)),
            "quat": np.vstack([1, 0, 0, 0]),
            "omega": np.zeros((3, 1)),
        },
    }
    COND = {
        "ext_unc": True,
        "int_unc": False,
        "gyro": False,
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__()
        self.pos = fym.BaseSystem(env_config["init"]["pos"])
        self.vel = fym.BaseSystem(env_config["init"]["vel"])
        self.quat = fym.BaseSystem(env_config["init"]["quat"])
        self.omega = fym.BaseSystem(env_config["init"]["omega"])

        self.prev_rotors = np.zeros((4, 1))

    def set_dot(self, t, rotors):
        m, g, J = self.m, self.g, self.J
        e3 = np.vstack((0, 0, 1))

        pos, vel, quat, omega = self.observe_list()

        fT, M1, M2, M3 = self.B.dot(rotors)
        M = np.vstack((M1, M2, M3))
        self.prev_rotors = rotors

        # uncertainty
        ext_pos, ext_vel, ext_euler, ext_omega = self.get_ext_uncertainties(t)
        int_pos, int_vel, int_euler, int_omega = self.get_int_uncertainties(t, vel)
        gyro = self.get_gyro(omega, rotors, self.prev_rotors)

        self.pos.dot = vel + ext_pos + int_pos
        dcm = quat2dcm(quat)
        self.vel.dot = g * e3 - fT * dcm.T.dot(e3) / m + ext_vel + int_vel
        # DCM integration (Note: dcm; I to B) [1]
        p, q, r = np.ravel(omega)
        # unit quaternion integration [4]
        dquat = 0.5 * np.array(
            [[0.0, -p, -q, -r], [p, 0.0, r, -q], [q, -r, 0.0, p], [r, q, -p, 0.0]]
        ).dot(quat)
        eps = 1 - (quat[0] ** 2 + quat[1] ** 2 + quat[2] ** 2 + quat[3] ** 2)
        k = 1
        self.quat.dot = (
            dquat
            + k * eps * quat
            + angle2quat(
                ext_euler[2] + int_euler[2],
                ext_euler[1] + int_euler[1],
                ext_euler[0] + int_euler[0],
            )
        )
        self.omega.dot = (
            self.Jinv.dot(M - np.cross(omega, J.dot(omega), axis=0) + gyro)
            + ext_omega
            + int_omega
        )

        quad_info = {
            # "rotors_cmd": rotors,
            "rotors": rotors,
            "fT": fT,
            "M": M,
            "Lambda": self.get_Lambda(t),
            "ref_dist": self.get_sumOfDist(t),
        }
        return quad_info

    def get_int_uncertainties(self, t, vel):
        if self.COND["int_unc"] is True:
            int_pos = np.zeros((3, 1))
            int_vel = np.vstack(
                [
                    np.sin(vel[0]) * vel[1],
                    np.sin(vel[1]) * vel[2],
                    np.exp(-t) * np.sin(t + np.pi / 4),
                ]
            )
            int_euler = np.zeros((3, 1))
            int_omega = np.zeros((3, 1))
        else:
            int_pos = np.zeros((3, 1))
            int_vel = np.zeros((3, 1))
            int_euler = np.zeros((3, 1))
            int_omega = np.zeros((3, 1))
        return int_pos, int_vel, int_euler, int_omega

    def get_ext_uncertainties(self, t):
        upos = np.zeros((3, 1))
        uvel = np.zeros((3, 1))
        uomega = np.zeros((3, 1))
        ueuler = np.zeros((3, 1))
        if self.COND["ext_unc"] is True:
            upos = np.vstack(
                [
                    0.1 * np.cos(0.2 * t),
                    0.2 * np.sin(0.5 * np.pi * t),
                    0.3 * np.cos(t),
                ]
            )
            uvel = np.vstack(
                [
                    0.1 * np.sin(t),
                    0.2 * np.sin(np.pi * t),
                    0.2 * np.sin(3 * t) - 0.1 * np.sin(0.5 * np.pi * t),
                ]
            )
            ueuler = np.vstack(
                [
                    0.2 * np.sin(t),
                    0.1 * np.cos(np.pi * t + np.pi / 4),
                    0.2 * np.sin(0.5 * np.pi * t),
                ]
            )
            uomega = np.vstack(
                [
                    -0.2 * np.sin(0.5 * np.pi * t),
                    0.1 * np.cos(np.sqrt(2) * t),
                    0.1 * np.cos(2 * t + 1),
                ]
            )
        return upos, uvel, ueuler, uomega

    def get_sumOfDist(self, t):
        pi = np.pi
        ref_dist = np.zeros((6, 1))
        ref_dist[0] = -(
            -pi / 5 * np.cos(t / 2) * np.sin(pi * t / 5)
            - (1 / 4 + pi**2 / 25) * np.sin(t / 2) * np.cos(pi * t / 5)
        )
        ref_dist[1] = -(
            pi / 5 * np.cos(t / 2) * np.cos(pi * t / 5)
            - (1 / 4 + pi**2 / 25) * np.sin(t / 2) * np.sin(pi * t / 5)
        )

        if self.COND["ext_unc"] is True:
            ext_dist = np.zeros((6, 1))
            m1, m2, m3, m4 = self.get_ext_uncertainties(t)
            ext_dist[0:3] = m2
            ext_dist[3:6] = m4
            int_dist = np.vstack(
                [
                    -0.1 * 0.2 * np.sin(0.2 * t),
                    0.2 * 0.5 * pi * np.cos(0.5 * pi * t),
                    -0.3 * np.sin(t),
                    0.2 * np.cos(t),
                    -0.1 * pi * np.sin(pi * t + pi / 4),
                    0.2 * 0.5 * pi * np.cos(0.5 * pi * t),
                ]
            )
            ref_dist = ref_dist + ext_dist + int_dist
        return ref_dist

    def get_gyro(self, omega, rotors, prev_rotors):
        # propeller gyro effect
        if self.COND["gyro"] is True:
            p, q, r = omega.ravel()
            Omega = rotors ** (1 / 2)
            Omega_r = -Omega[0] + Omega[1] - Omega[2] + Omega[3]
            prev_Omega = prev_rotors ** (1 / 2)
            prev_Omega_r = (
                -prev_Omega[0] + prev_Omega[1] - prev_Omega[2] + prev_Omega[3]
            )
            gyro = np.vstack(
                [
                    self.Jr * q * Omega_r,
                    -self.Jr * p * Omega_r,
                    self.Jr * (Omega_r - prev_Omega_r),
                ]
            )
        else:
            gyro = np.zeros((3, 1))
        return gyro

    def groundEffect(self, u1):
        h = -self.pos.state[2]

        if h == 0:
            ratio = 2
        else:
            ratio = 1 / (1 - (self.R / 4 / self.pos.state[2]) ** 2)

        if ratio > self.max_IGE_ratio:
            u1_d = self.max_IGE_ratio * u1
        else:
            u1_d = ratio * u1
        return u1_d

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
        if t > 13:
            W3 = 0.4
        elif t > 11:
            W3 = 0.8
        else:
            W3 = 1

        W = np.diag([W1, W2, W3, 1])
        return W

    def set_Lambda(self, t, brfs):
        Lambda = self.get_Lambda(t)
        return Lambda.dot(brfs)


class ExtendedQuadEnv(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 20,
        },
        "quad": {
            "init": {
                "pos": np.vstack([0.0, 0.0, 0.0]),
                "vel": np.zeros((3, 1)),
                "quat": np.vstack([1, 0, 0, 0]),
                "omega": np.zeros((3, 1)),
            },
        },
        "rtype": "Quad",
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        # quad
        self.plant = Quad(env_config["quad"])
        # controller
        self.env_config = env_config
        self.controller = ftc.make("BLF", self)

    def step(self):
        env_info, done = self.update()
        return done, env_info

    def observation(self):
        return self.observe_flat()

    def get_reward(self, obs, action, next_obs):
        return 0

    def get_ref(self, t, *args):
        posd = np.vstack(
            [
                np.sin(t / 2) * np.cos(np.pi * t / 5),
                np.sin(t / 2) * np.sin(np.pi * t / 5),
                -t,
            ]
        )
        refs = {"posd": posd}
        return [refs[key] for key in args]

    def set_dot(self, t):
        forces, controller_info = self.controller.get_control(t, self)
        rotors_cmd = np.linalg.pinv(
            self.plant.B.dot(self.plant.get_Lambda(t - self.plant.fault_delay))
        ).dot(forces)
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)
        rotors = self.plant.get_Lambda(t).dot(rotors)
        quad_info = self.plant.set_dot(t, rotors)

        # caculate f
        J = np.diag(self.plant.J)
        p_, q_, r_ = self.plant.omega.state
        f = np.array(
            [
                (J[1] - J[2]) / J[0] * q_ * r_,
                (J[2] - J[0]) / J[1] * p_ * r_,
                (J[0] - J[1]) / J[2] * p_ * q_,
            ]
        )

        env_info = {
            "t": t,
            **self.observe_dict(),
            **quad_info,
            **controller_info,
            "rotors_cmd": rotors_cmd,
            "posd": self.get_ref(t, "posd"),
            "f": f,
        }

        return env_info


def run(config):
    env = ExtendedQuadEnv(config)
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
        return


def plot():
    data = fym.load("data.h5")["env"]
    rotor_min = 0
    rotor_max = 1e6

    # Rotor
    plt.figure()

    ax = plt.subplot(221)
    for i in range(data["rotors_cmd"].shape[1]):
        if i != 0:
            plt.subplot(221 + i, sharex=ax)
        plt.ylim([rotor_min - 5, np.sqrt(rotor_max) + 5])
        plt.plot(data["t"], np.sqrt(data["rotors"][:, i]), "k-", label="Response")
        plt.plot(data["t"], np.sqrt(data["rotors_cmd"][:, i]), "r--", label="Command")
        if i == 0:
            plt.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Angular rate of each rotor")
    plt.tight_layout()

    # Position
    plt.figure()
    # plt.ylim([-5, 5])

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["x", "y", "z"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311 + i, sharex=ax)
        plt.plot(
            data["t"],
            data["obs_pos"][:, i, 0] + data["posd"].squeeze()[:, i],
            "b-",
            label="Estimated",
        )
        plt.plot(data["t"], data["plant"]["pos"][:, i, 0], "k-.", label="Real")
        plt.plot(data["t"], data["posd"].squeeze()[:, i], "r--", label="Desired")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Position, m")
    plt.tight_layout()

    # velocity
    # plt.figure()
    # plt.ylim([-5, 5])

    # ax = plt.subplot(311)
    # for i, (_label, _ls) in enumerate(zip(["Vx", "Vy", "Vz"], ["-", "--", "-."])):
    #     if i != 0:
    #         plt.subplot(311+i, sharex=ax)
    #     plt.plot(data["t"], data["plant"]["vel"][:, i, 0], "k"+_ls, label=_label)
    #     plt.ylabel(_label)
    # plt.gcf().supxlabel("Time, sec")
    # plt.gcf().supylabel("Velocity, m/s")
    # plt.tight_layout()

    # observation: position error
    plt.figure()

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["ex", "ey", "ez"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311 + i, sharex=ax)
        plt.plot(data["t"], data["obs_pos"][:, i, 0], "b-", label="Estimated")
        plt.plot(
            data["t"],
            data["plant"]["pos"][:, i, 0]
            - data["posd"].squeeze(-1)[:, :, i].squeeze(-1),
            "k-.",
            label="Real",
        )
        plt.plot(data["t"], data["bound_err"], "c")
        plt.plot(data["t"], -data["bound_err"], "c")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Error observation, m/s")
    plt.tight_layout()

    # euler angles
    plt.figure()

    ax = plt.subplot(311)
    angles = np.vstack(
        [
            quat2angle(data["plant"]["quat"][j, :, 0])
            for j in range(len(data["plant"]["quat"][:, 0, 0]))
        ]
    )
    ax = plt.subplot(311)
    for i, _label in enumerate([r"$\phi$", r"$\theta$", r"$\psi$"]):
        if i != 0:
            plt.subplot(311 + i, sharex=ax)
        plt.plot(
            data["t"], np.rad2deg(data["obs_ang"][:, i, 0]), "b-", label="Estimated"
        )
        plt.plot(data["t"], np.rad2deg(angles[:, 2 - i]), "k-.", label="Real")
        plt.plot(data["t"], np.rad2deg(data["eulerd"][:, i, 0]), "r--", label="Desired")
        plt.plot(data["t"], np.rad2deg(data["bound_ang"][:, 0]), "c", label="bound")
        plt.plot(data["t"], -np.rad2deg(data["bound_ang"][:, 0]), "c")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc="upper right")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Euler angles, deg")
    plt.tight_layout()
    # plt.savefig("lpeso_angle.png", dpi=300)

    # angular rates
    plt.figure()

    for i, (_label, _ls) in enumerate(zip(["p", "q", "r"], ["-.", "--", "-"])):
        plt.plot(
            data["t"],
            np.rad2deg(data["plant"]["omega"][:, i, 0]),
            "k" + _ls,
            label=_label,
        )
    plt.plot(data["t"], np.rad2deg(data["bound_ang"][:, 1]), "c", label="bound")
    plt.plot(data["t"], -np.rad2deg(data["bound_ang"][:, 1]), "c")
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Angular rates, deg/s")
    plt.tight_layout()
    plt.legend(loc="upper right")

    # virtual control
    plt.figure()

    ax = plt.subplot(411)
    for i, _label in enumerate([r"$F$", r"$M_{\phi}$", r"$M_{\theta}$", r"$M_{\psi}$"]):
        if i != 0:
            plt.subplot(411 + i, sharex=ax)
        plt.plot(data["t"], data["forces"][:, i], "k-", label=_label)
        plt.ylabel(_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Generalized forces")
    plt.tight_layout()
    # plt.savefig("lpeso_forces.png", dpi=300)

    # disturbance
    plt.figure()

    ax = plt.subplot(611)
    for i, _label in enumerate(
        [r"$d_x$", r"$d_y$", r"$d_z$", r"$d_\phi$", r"$d_\theta$", r"$d_\psi$"]
    ):
        if i != 0:
            plt.subplot(611 + i, sharex=ax)
        plt.plot(data["t"], data["ref_dist"][:, i, 0], "r-", label="true")
        plt.plot(data["t"], data["dist"][:, i, 0], "k", label=" distarbance")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc="upper right")
    plt.gcf().supylabel("dist")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

    # q
    # plt.figure()

    # ax = plt.subplot(311)
    # for i, _label in enumerate([r"$q_x$", r"$q_y$", r"$q_z$"]):
    #     if i != 0:
    #         plt.subplot(311+i, sharex=ax)
    #     plt.plot(data["t"], data["q"][:, i, 0], "k-")
    #     plt.ylabel(_label)
    # plt.gcf().supylabel("observer control input")
    # plt.gcf().supxlabel("Time, sec")
    # plt.tight_layout()

    plt.show()


def main(args):
    if args.only_plot:
        plot()
        return
    elif args.with_ray:

        def objective(config):
            np.seterr(all="raise")

            env = ExtendedQuadEnv(config)

            env.reset()
            tf = 0
            try:
                while True:
                    done, env_info = env.step()
                    tf = env.info["t"]

                    if done:
                        break

            finally:
                return {"tf": tf}

        config = {
            "k11": tune.uniform(1.01, 5),
            "k12": tune.uniform(20, 40),
            "k13": tune.uniform(0, 2),
            "k21": tune.uniform(10, 30),
            "k22": tune.uniform(10, 30),
            "k23": tune.uniform(0, 2),
            "eps1": tune.uniform(1.01, 5),
            "eps2": tune.uniform(15, 35),
        }
        current_best_params = [
            {
                "k11": 2,
                "k12": 30,
                "k13": 5 / 30 / (0.5) ** 2,
                "k21": 500 / 30,
                "k22": 30,
                "k23": 5 / 30 / np.deg2rad(45) ** 2,
                "eps1": 5,
                "eps2": 25,
            }
        ]
        search = HyperOptSearch(
            metric="tf",
            mode="max",
            points_to_evaluate=current_best_params,
        )
        tuner = tune.Tuner(
            tune.with_resources(
                objective,
                resources={"cpu": 12},
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
        current_best_params = {
            "k11": 2,
            "k12": 30,
            "k13": 5 / 30 / (0.5) ** 2,
            "k21": 500 / 30,
            "k22": 30,
            "k23": 5 / 30 / np.deg2rad(45) ** 2,
            "eps1": 5,
            "eps2": 25,
        }
        run(current_best_params)
        plot()

        if args.plot:
            plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--plot", action="store_true")
    parser.add_argument("-P", "--only-plot", action="store_true")
    parser.add_argument("-r", "--with-ray", action="store_true")
    args = parser.parse_args()
    main(args)
