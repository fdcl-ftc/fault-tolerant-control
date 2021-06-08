import numpy as np
from numpy import arcsin
import matplotlib.pyplot as plt

from fym.utils.rot import quat2angle, angle2quat
import fym.logging
from fym.core import BaseEnv, BaseSystem

from copter import Copter_nonlinear
from ISMC import IntegralSMC


class Env(BaseEnv):
    def __init__(self):
        super().__init__(solver="odeint", max_t=10, dt=5, ode_step_len=100)
        self.plant = Copter_nonlinear()
        ic = np.vstack((0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        self.pos_des = np.vstack((-1, 1, -2))
        self.vel_des = np.vstack((0, 0, 0))
        self.quat_des = np.vstack((1, 0, 0, 0))
        self.omega_des = np.vstack((0, 0, 0))
        ref0 = np.vstack((self.pos_des, self.vel_des, self.quat_des, self.omega_des))

        self.controller = IntegralSMC(self.plant.J,
                                      self.plant.m,
                                      self.plant.g,
                                      self.plant.d,
                                      ic,
                                      ref0)

    def step(self):
        *_, done = self.update()
        return done

    def control_allocation(self, forces):
        rotors = np.linalg.pinv(self.plant.mixer.B).dot(forces)
        rotors = np.clip(rotors, 0, self.plant.rotor_max)

        return rotors

    def get_ref(self, t, x):
        pos_des = np.vstack((-1, 1, -2))
        vel_des = np.vstack((0, 0, 0))
        # pos_des = np.vstack((cos(t), sin(t), -t))
        # vel_des = np.vstack((-sin(t), cos(t), -1))
        quat_des = np.vstack((1, 0, 0, 0))
        omega_des = np.zeros((3, 1))
        ref = np.vstack((pos_des, vel_des, quat_des, omega_des))

        return ref

    def _get_derivs(self, t, x, p):
        ref = self.get_ref(t, x)

        K = np.array([[25, 20],
                      [200, 20],
                      [200, 20],
                      [25, 10]])
        Kc = np.vstack((1, 1, 1, 1))
        PHI = np.vstack([1] * 4)

        forces = self.controller.get_FM(x, ref, p, K, Kc, PHI, t)
        rotors = self.control_allocation(forces)

        return rotors, ref

    def set_dot(self, t):
        x = self.plant.state
        p = self.controller.state
        rotors, ref = self._get_derivs(t, x, p)

        self.plant.set_dot(t, rotors)
        self.controller.set_dot(x, ref)

    def logger_callback(self, i, t, y, *args):
        states = self.observe_dict(y)
        x_flat = self.plant.observe_vec(y[self.plant.flat_index])
        p = self.controller.observe_list(y[self.controller.flat_index])
        rotors, ref = self._get_derivs(t, x_flat, p)
        return dict(t=t, **states, rotors=rotors, ref=ref)


def run():
    env = Env()
    env.logger = fym.logging.Logger("data.h5")

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            break

    env.close()


def exp1():
    run()


def exp1_plot():
    data = fym.logging.load("data.h5")

    fig = plt.figure()
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig.add_subplot(4, 1, 4, sharex=ax1)

    ax1.plot(data['t'], data['plant']['pos'].squeeze(), label="plant")
    ax1.plot(data["t"], data["ref"][:, 0, 0], "r--", label="x (cmd)")
    ax1.plot(data["t"], data["ref"][:, 1, 0], "r--", label="y (cmd)")
    ax1.plot(data["t"], data["ref"][:, 2, 0], "r--", label="z (cmd)")
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
    ax1 = fig2.add_subplot(4, 1, 1)
    ax2 = fig2.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig2.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig2.add_subplot(4, 1, 4, sharex=ax1)

    ax1.plot(data['t'], data['rotors'].squeeze()[:, 0])
    ax2.plot(data['t'], data['rotors'].squeeze()[:, 1])
    ax3.plot(data['t'], data['rotors'].squeeze()[:, 2])
    ax4.plot(data['t'], data['rotors'].squeeze()[:, 3])

    ax1.set_ylabel('rotor1')
    ax1.grid(True)
    ax2.set_ylabel('rotor2')
    ax2.grid(True)
    ax3.set_ylabel('rotor3')
    ax3.grid(True)
    ax4.set_ylabel('rotor4')
    ax4.grid(True)
    ax4.set_xlabel('Time [sec]')

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    exp1()
    exp1_plot()
