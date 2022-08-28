import numpy as np
import matplotlib.pyplot as plt
import argparse
from copy import deepcopy
from functools import reduce

import fym
from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import dcm2quat, quat2dcm, angle2quat, quat2angle

import ftc

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
        [[b, b, b, b],
         [0, -b*d, 0, b*d],
         [b*d, 0, -b*d, 0],
         [-c, c, -c, c]]
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
        "int_unc": True,
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

        rotors_sat = self.saturate(rotors)
        lrotors = self.set_Lambda(t-self.fault_delay, rotors_sat)
        lrotors_sat = self.saturate(lrotors)
        fT, M1, M2, M3 = self.B.dot(lrotors_sat)
        M = np.vstack((M1, M2, M3))
        self.prev_rotors = lrotors_sat

        # uncertainty
        ext_pos, ext_vel, ext_euler, ext_omega = self.get_ext_uncertainties(t)
        int_pos, int_vel, int_euler, int_omega = self.get_int_uncertainties(t, vel)
        gyro = self.get_gyro(omega, rotors, self.prev_rotors)

        self.pos.dot = vel + ext_pos + int_pos
        dcm = quat2dcm(quat)
        self.vel.dot = (g*e3 - fT*dcm.T.dot(e3)/m
                        + ext_vel + int_vel
                        )
        # DCM integration (Note: dcm; I to B) [1]
        p, q, r = np.ravel(omega)
        # unit quaternion integration [4]
        dquat = 0.5 * np.array([[0., -p, -q, -r],
                                [p, 0., r, -q],
                                [q, -r, 0., p],
                                [r, q, -p, 0.]]).dot(quat)
        eps = 1 - (quat[0]**2+quat[1]**2+quat[2]**2+quat[3]**2)
        k = 1
        self.quat.dot = (dquat + k*eps*quat
                         + angle2quat(ext_euler[2]+int_euler[2],
                                      ext_euler[1]+int_euler[1],
                                      ext_euler[0]+int_euler[0]))
        self.omega.dot = self.Jinv.dot(
            M
            - np.cross(omega, J.dot(omega), axis=0)
            + gyro
        ) \
            + ext_omega + int_omega

        quad_info = {
            "rotors": rotors,
            "rotors_sat": rotors_sat,
            "lrotors": lrotors,
            "lrotors_sat": lrotors_sat,
            "fT": fT,
            "M": M,
            "Lambda": self.get_Lambda(t),
            "ref_dist": self.get_sumOfDist(t),
        }
        return quad_info

    def saturate(self, rfs):
        """Saturation function"""
        return np.clip(rfs, 0, self.rotor_max)

    def get_int_uncertainties(self, t, vel):
        if self.COND["int_unc"] is True:
            int_pos = np.zeros((3, 1))
            int_vel = np.vstack([
                np.sin(vel[0])*vel[1],
                np.sin(vel[1])*vel[2],
                np.exp(-t)*np.sin(t+np.pi/4)
            ])
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
            upos = np.vstack([
                0.1*np.cos(2*np.pi*t),
                0.2*np.sin(0.5*np.pi*t),
                0.3*np.cos(t),
            ])
            uvel = np.vstack([
                0.1*np.sin(t),
                0.2*np.sin(np.pi*t),
                0.2*np.sin(3*t) - 0.1*np.sin(0.5*np.pi*t)
            ])
            ueuler = np.vstack([
                0.3*np.sin(t),
                0.1*np.cos(np.pi*t+np.pi/4),
                0.2*np.sin(0.5*np.pi*t),
            ])
            uomega = np.vstack([
                - 0.2*np.sin(0.5*np.pi*t),
                0.1*np.cos(np.sqrt(2)*t),
                0.1*np.cos(2*t+1)
            ])
        return upos, uvel, ueuler, uomega

    def get_sumOfDist(self, t):
        pi = np.pi
        ref_dist = np.zeros((6, 1))
        ref_dist[0] = - (- pi/10*np.cos(t/2)*np.sin(pi*t/10)
                         - (1/4 + pi**2/100)*np.sin(t/2)*np.cos(pi*t/10))
        ref_dist[1] = - (pi/5*np.cos(t/2)*np.cos(pi*t/5)
                         - (1/4 + pi**2/25)*np.sin(t/2)*np.sin(pi*t/5))

        ext_dist = np.zeros((6, 1))
        m1, m2, m3, m4 = self.get_ext_uncertainties(t)
        ext_dist[0:3] = m2
        ext_dist[3:6] = m4
        int_dist = np.vstack([- 0.1*2*pi*np.sin(2*pi*t),
                              0.2*0.5*pi*np.cos(0.5*pi*t),
                              - 0.3*np.sin(t),
                              0.3*np.cos(t),
                              - 0.1*pi*np.sin(pi*t+pi/4),
                              0.2*0.5*pi*np.cos(0.5*pi*t)])
        ref_dist = ref_dist + ext_dist + int_dist
        return ref_dist

    def get_gyro(self, omega, rotors, prev_rotors):
        # propeller gyro effect
        if self.COND["gyro"] is True:
            p, q, r = omega.ravel()
            Omega = rotors ** (1/2)
            Omega_r = - Omega[0] + Omega[1] - Omega[2] + Omega[3]
            prev_Omega = prev_rotors ** (1/2)
            prev_Omega_r = (- prev_Omega[0] + prev_Omega[1]
                            - prev_Omega[2] + prev_Omega[3])
            gyro = np.vstack([self.Jr * q * Omega_r,
                              - self.Jr * p * Omega_r,
                              self.Jr * (Omega_r - prev_Omega_r)])
        else:
            gyro = np.zeros((3, 1))
        return gyro

    def groundEffect(self, u1):
        h = - self.pos.state[2]

        if h == 0:
            ratio = 2
        else:
            ratio = 1 / (1 - (self.R/4/self.pos.state[2])**2)

        if ratio > self.max_IGE_ratio:
            u1_d = self.max_IGE_ratio * u1
        else:
            u1_d = ratio * u1
        return u1_d

    def get_Lambda(self, t):
        """Lambda function"""
        if t > 20:
            W1 = 0.4
        elif t > 3:
            W1 = (- 40/17**2 * (t+14) * (t-20) + 40) * 0.01
        else:
            W1 = 1

        if t > 11:
            W2 = 0.7
        elif t > 6:
            W2 = (6/5 * (t-11)**2 + 70) * 0.01
        else:
            W2 = 1

        if t > 10:
            W3 = 0.9
        else:
            W3 = 1

        if t > 25:
            W4 = 0.5
        else:
            W4 = 1
        W = np.diag([W1, W2, W3, W4])
        return W

    def set_Lambda(self, t, brfs):
        Lambda = self.get_Lambda(t)
        return Lambda.dot(brfs)


class ExtendedQuadEnv(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 40,
        },
        "quad": {
            "init": {
                "pos": np.vstack((0., 0., 0.0)),
                "vel": np.zeros((3, 1)),
                "R": np.vstack([1, 0, 0, 0]),
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
        self.controller = ftc.make("BLF")

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
        posd = np.vstack([np.sin(t/2)*np.cos(np.pi*t/10),
                          np.sin(t/2)*np.sin(np.pi*t/5),
                          -t])
        refs = {"posd": posd}
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
    rotor_min = data["plant"]["rotor_min"]
    rotor_max = data["plant"]["rotor_max"]

    # Rotor
    plt.figure()

    ax = plt.subplot(221)
    for i in range(data["rotors"].shape[1]):
        if i != 0:
            plt.subplot(221+i, sharex=ax)
        plt.ylim([rotor_min-5, np.sqrt(rotor_max)+5])
        plt.plot(data["t"], np.sqrt(data["rotors"][:, i]), "k-", label="Response")
        plt.plot(data["t"], np.sqrt(data["rotors_cmd"][:, i]), "r--", label="Command")
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Angular rate of each rotor")
    plt.tight_layout()

    # Position
    plt.figure()
    # plt.ylim([-5, 5])

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["x", "y", "z"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], data["obs_pos"][:, i, 0]+data["ref"][:, i, 0], "b-", label="Estimated")
        plt.plot(data["t"], data["plant"]["pos"][:, i, 0], "k-.", label="Real")
        plt.plot(data["t"], data["ref"][:, i, 0], "r--", label="Desired")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Position, m")
    plt.tight_layout()

    # velocity
    plt.figure()
    plt.ylim([-5, 5])

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["Vx", "Vy", "Vz"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], data["plant"]["vel"][:, i, 0], "k"+_ls, label=_label)
        plt.ylabel(_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Velocity, m/s")
    plt.tight_layout()

    # observation: position error
    plt.figure()

    ax = plt.subplot(311)
    for i, (_label, _ls) in enumerate(zip(["ex", "ey", "ez"], ["-", "--", "-."])):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], data["obs_pos"][:, i, 0], "b-", label="Estimated")
        plt.plot(data["t"], data["plant"]["pos"][:, i, 0]-data["ref"][:, i, 0], "k-.", label="Real")
        plt.plot(data["t"], data["bound_err"], "c")
        plt.plot(data["t"], -data["bound_err"], "c")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Error observation, m/s")
    plt.tight_layout()

    # euler angles
    plt.figure()

    ax = plt.subplot(311)
    angles = np.vstack([quat2angle(data["x"]["quat"][j, :, 0]) for j in range(len(data["x"]["quat"][:, 0, 0]))])
    ax = plt.subplot(311)
    for i, _label in enumerate([r"$\phi$", r"$\theta$", r"$\psi$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], np.rad2deg(data["obs_ang"][:, i, 0]), "b-", label="Estimated")
        plt.plot(data["t"], np.rad2deg(angles[:, 2-i]), "k-.", label="Real")
        plt.plot(data["t"], np.rad2deg(data["eulerd"][:, i, 0]), "r--", label="Desired")
        plt.plot(data["t"], data["bound_ang"][:, 0], "c", label="bound")
        plt.plot(data["t"], -data["bound_ang"][:, 0], "c")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Euler angles, deg")
    plt.tight_layout()
    # plt.savefig("lpeso_angle.png", dpi=300)

    # angular rates
    plt.figure()

    for i, (_label, _ls) in enumerate(zip(["p", "q", "r"], ["-.", "--", "-"])):
        plt.plot(data["t"], np.rad2deg(data["x"]["omega"][:, i, 0]), "k"+_ls, label=_label)
    plt.plot(data["t"], data["bound_ang"][:, 1], 'c', label="bound")
    plt.plot(data["t"], -data["bound_ang"][:, 1], 'c')
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Angular rates, deg/s")
    plt.tight_layout()
    plt.legend(loc='upper right')

    # virtual control
    plt.figure()

    ax = plt.subplot(411)
    for i, _label in enumerate([r"$F$", r"$M_{\phi}$", r"$M_{\theta}$", r"$M_{\psi}$"]):
        if i != 0:
            plt.subplot(411+i, sharex=ax)
        plt.plot(data["t"], data["virtual_u"][:, i], "k-", label=_label)
        plt.ylabel(_label)
    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Generalized forces")
    plt.tight_layout()
    # plt.savefig("lpeso_forces.png", dpi=300)

    # disturbance
    plt.figure()

    ax = plt.subplot(611)
    for i, _label in enumerate([r"$d_x$", r"$d_y$", r"$d_z$",
                                r"$d_\phi$", r"$d_\theta$", r"$d_\psi$"]):
        if i != 0:
            plt.subplot(611+i, sharex=ax)
        plt.plot(data["t"], data["ref_dist"][:, i, 0], "r-", label="true")
        plt.plot(data["t"], data["dist"][:, i, 0], "k", label=" distarbance")
        plt.ylabel(_label)
        if i == 0:
            plt.legend(loc='upper right')
    plt.gcf().supylabel("dist")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

    # q
    plt.figure()

    ax = plt.subplot(311)
    for i, _label in enumerate([r"$q_x$", r"$q_y$", r"$q_z$"]):
        if i != 0:
            plt.subplot(311+i, sharex=ax)
        plt.plot(data["t"], data["q"][:, i, 0], "k-")
        plt.ylabel(_label)
    plt.gcf().supylabel("observer control input")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

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
