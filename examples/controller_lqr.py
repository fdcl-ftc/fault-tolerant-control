from copy import deepcopy
from functools import reduce

import fym
import matplotlib.pyplot as plt
import numpy as np

import ftc
from ftc.models.multicopter import Multicopter
from ftc.utils import safeupdate

np.seterr(all="raise")


class MyEnv(fym.BaseEnv):
    ENV_CONFIG = {
        "fkw": {
            "dt": 0.01,
            "max_t": 20,
        },
        "init": {
            "pos": np.vstack((0.5, 0.5, 0.0)),
            "vel": np.zeros((3, 1)),
            "quat": np.vstack((1, 0, 0, 0)),
            "omega": np.zeros((3, 1)),
        },
    }

    def __init__(self, env_config={}):
        env_config = safeupdate(self.ENV_CONFIG, env_config)
        super().__init__(**env_config["fkw"])
        self.plant = Multicopter(env_config["init"]["pos"],
                                 env_config["init"]["vel"],
                                 env_config["init"]["quat"],
                                 env_config["init"]["omega"],
                                 )
        self.controller = ftc.make("LQR")
        Jinv = self.plant.Jinv
        m, g = self.plant.m, self.plant.g
        self.trim_forces = np.vstack([m * g, 0, 0, 0])

        A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        B = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [-1/m, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, Jinv[0, 0], 0, 0],
                      [0, 0, Jinv[1, 1], 0],
                      [0, 0, 0, Jinv[2, 2]]])
        Q = np.diag([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
        R = np.diag([1, 1, 1, 1])

        self.K, *_ = fym.agents.LQR.clqr(A, B, Q, R)

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
        forces, controller_info = self.controller.get_control(t, self)
        rotors = self.plant.mixer(forces)
        self.plant.set_dot(t, rotors)

        env_info = {
            "t": t,
            **self.observe_dict(),
            **controller_info,
            "rotors": rotors,
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

    fig = plt.figure()

    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)

    ax1.plot(data['t'], data['plant']['pos'].squeeze())
    ax2.plot(data['t'], data['plant']['vel'].squeeze())
    ax3.plot(data['t'], data['plant']['quat'].squeeze())
    ax4.plot(data['t'], data['plant']['omega'].squeeze())

    ax1.set_ylabel('Position')
    ax1.legend([r'$x$', r'$y$', r'$z$'])
    ax1.grid(True)

    ax2.set_ylabel('Velocity')
    ax2.legend([r'$u$', r'$v$', r'$w$'])
    ax2.grid(True)

    ax3.set_ylabel('Quaternion')
    ax3.legend([r'$q_0$', r'$q_1$', r'$q_2$', r'$q_3$'])
    ax3.grid(True)

    ax4.set_ylabel('Angular Velocity')
    ax4.legend([r'$p$', r'$q$', r'$r$'])
    ax4.set_xlabel('Time [sec]')
    ax4.grid(True)

    plt.tight_layout()

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(6, 1, 1)
    ax2 = fig2.add_subplot(6, 1, 2, sharex=ax1)
    ax3 = fig2.add_subplot(6, 1, 3, sharex=ax1)
    ax4 = fig2.add_subplot(6, 1, 4, sharex=ax1)
    ax5 = fig2.add_subplot(6, 1, 5, sharex=ax1)
    ax6 = fig2.add_subplot(6, 1, 6, sharex=ax1)

    ax1.plot(data['t'], data['rotors'].squeeze()[:, 0])
    ax2.plot(data['t'], data['rotors'].squeeze()[:, 1])
    ax3.plot(data['t'], data['rotors'].squeeze()[:, 2])
    ax4.plot(data['t'], data['rotors'].squeeze()[:, 3])
    ax5.plot(data['t'], data['rotors'].squeeze()[:, 4])
    ax6.plot(data['t'], data['rotors'].squeeze()[:, 5])

    ax1.set_ylabel('rotor1')
    ax1.grid(True)
    ax2.set_ylabel('rotor2')
    ax2.grid(True)
    ax3.set_ylabel('rotor3')
    ax3.grid(True)
    ax4.set_ylabel('rotor4')
    ax4.grid(True)
    ax5.set_ylabel('rotor5')
    ax5.grid(True)
    ax6.set_ylabel('rotor6')
    ax6.grid(True)
    ax6.set_xlabel('Time [sec]')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    run()
