import numpy as np
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging
from fym.utils.rot import angle2quat

from ftc.models.multicopter import Multicopter
from ftc.agents.CA import Grouping, CA
from ftc.agents.fdi import SimpleFDI
from ftc.faults.actuator import LoE
from ftc.agents.AdaptiveISMC import AdaptiveISMController
from copy import deepcopy


class Env(BaseEnv):
    def __init__(self):
        super().__init__(solver="odeint", max_t=20, dt=10, ode_step_len=100)
        self.plant = Multicopter()
        n = self.plant.mixer.B.shape[1]

        # Define faults
        self.sensor_faults = []
        self.actuator_faults = [
            LoE(time=3, index=0, level=0.),
        ]

        # Define FDI
        self.fdi = SimpleFDI(no_act=n, tau=0.1)

        # Define initial condition and reference at t=0
        ic = np.vstack((0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0))
        ref0 = np.vstack((1, -1, -2, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0))

        # Define agents
        self.controller = AdaptiveISMController(self.plant.J,
                                                self.plant.m,
                                                self.plant.g,
                                                self.plant.d,
                                                ic,
                                                ref0)

        self.detection_time = [[] for _ in range(len(self.actuator_faults))]

    def step(self):
        *_, done = self.update()
        return done

    def get_ref(self, t, x):
        pos_des = np.vstack((1, -1, -2))
        vel_des = np.vstack((0, 0, 0))
        quat_des = np.vstack((1, 0, 0, 0))
        omega_des = np.zeros((3, 1))
        ref = np.vstack((pos_des, vel_des, quat_des, omega_des))

        return ref

    def _get_derivs(self, t, x, What, p, gamma):
        # Set sensor faults
        for sen_fault in self.sensor_faults:
            x = sen_fault(t, x)

        fault_index = self.fdi.get_index(What)
        ref = self.get_ref(t, x)

        K = np.array([[2, 1],
                      [20, 2],
                      [20, 2],
                      [2, 1]])
        Kc = np.vstack((5, 9, 5, 5))
        PHI = np.vstack([1] * 4)

        forces, sliding = self.controller.get_FM(x, ref, p, gamma, K, Kc, PHI, t)

        rotors_cmd = np.linalg.pinv(self.plant.mixer.B).dot(forces)
        _rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)
        rotors = deepcopy(_rotors)

        # Set actuator faults
        for act_fault in self.actuator_faults:
            rotors = act_fault(t, rotors)

        _rotors[fault_index] = 1
        W = self.fdi.get_true(rotors, _rotors)

        return rotors_cmd, W, rotors, forces, ref, sliding

    def set_dot(self, t):
        x = self.plant.state
        What = self.fdi.state
        p, gamma = self.controller.observe_list()

        rotors_cmd, W, rotors, forces, ref, sliding = \
            self._get_derivs(t, x, What, p, gamma)

        self.plant.set_dot(t, rotors)
        self.controller.set_dot(x, ref, sliding)
        self.fdi.set_dot(W)

    def logger_callback(self, i, t, y, *args):
        states = self.observe_dict(y)
        x_flat = self.plant.observe_vec(y[self.plant.flat_index])
        ctrl_flat = self.controller.observe_list(y[self.controller.flat_index])
        x = states["plant"]
        What = states["fdi"]
        rotors_cmd, W, rotors, forces, ref, sliding = \
            self._get_derivs(t, x_flat, What, ctrl_flat[0], ctrl_flat[1])

        return dict(t=t, x=x, What=What, rotors=rotors,
                    rotors_cmd=rotors_cmd, W=W, ref=ref, gamma=ctrl_flat[1])


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

    # FDI
    plt.figure()

    ax = plt.subplot(321)
    plt.plot(data["t"], data["W"][:, 0, 0], "r--", label="Actual")
    plt.plot(data["t"], data["What"][:, 0, 0], "k-", label="Estimated")
    plt.ylim([-0.1, 1.1])
    plt.legend()

    plt.subplot(322, sharex=ax)
    plt.plot(data["t"], data["W"][:, 1, 1], "r--")
    plt.plot(data["t"], data["What"][:, 1, 1], "k-")
    plt.ylim([-0.1, 1.1])

    plt.subplot(323, sharex=ax)
    plt.plot(data["t"], data["W"][:, 2, 2], "r--")
    plt.plot(data["t"], data["What"][:, 2, 2], "k-")
    plt.ylim([-0.1, 1.1])

    plt.subplot(324, sharex=ax)
    plt.plot(data["t"], data["W"][:, 3, 3], "r--")
    plt.plot(data["t"], data["What"][:, 3, 3], "k-")
    plt.ylim([-0.1, 1.1])

    plt.subplot(325, sharex=ax)
    plt.plot(data["t"], data["W"][:, 4, 4], "r--")
    plt.plot(data["t"], data["What"][:, 4, 4], "k-")
    plt.ylim([-0.1, 1.1])

    plt.subplot(326, sharex=ax)
    plt.plot(data["t"], data["W"][:, 5, 5], "r--")
    plt.plot(data["t"], data["What"][:, 5, 5], "k-")
    plt.ylim([-0.1, 1.1])

    plt.gcf().supylabel("FDI")
    plt.gcf().supxlabel("Time, sec")
    plt.tight_layout()

    # Rotor
    plt.figure()

    ax = plt.subplot(321)
    plt.plot(data["t"], data["rotors"][:, 0], "k-", label="Response")
    plt.plot(data["t"], data["rotors_cmd"][:, 0], "r--", label="Command")
    plt.ylim([-5.1, 12.1])
    plt.legend()

    plt.subplot(322, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 1], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 1], "r--")
    plt.ylim([-5.1, 12.1])

    plt.subplot(323, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 2], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 2], "r--")
    plt.ylim([-5.1, 12.1])

    plt.subplot(324, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 3], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 3], "r--")
    plt.ylim([-5.1, 12.1])

    plt.subplot(325, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 4], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 4], "r--")
    plt.ylim([-5.1, 12.1])

    plt.subplot(326, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 5], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 5], "r--")
    plt.ylim([-5.1, 12.1])

    plt.gcf().supxlabel("Time, sec")
    plt.gcf().supylabel("Rotor force")
    plt.tight_layout()

    plt.figure()

    plt.plot(data["t"], data["ref"][:, 0, 0], "r-", label="x (cmd)")
    plt.plot(data["t"], data["x"]["pos"][:, 0, 0], "k-", label="x")

    plt.plot(data["t"], data["ref"][:, 1, 0], "r--", label="y (cmd)")
    plt.plot(data["t"], data["x"]["pos"][:, 1, 0], "k--", label="y")

    plt.plot(data["t"], data["ref"][:, 2, 0], "r-.", label="z (cmd)")
    plt.plot(data["t"], data["x"]["pos"][:, 2, 0], "k-.", label="z")

    # plt.axvspan(3, 3.042, alpha=0.2, color="b")
    # plt.axvline(3.042, alpha=0.8, color="b", linewidth=0.5)

    # plt.axvspan(6, 6.011, alpha=0.2, color="b")
    # plt.axvline(6.011, alpha=0.8, color="b", linewidth=0.5)

    # plt.annotate("Rotor 0 fails", xy=(3, 0), xytext=(3.5, 0.5),
    #              arrowprops=dict(arrowstyle='->', lw=1.5))
    # plt.annotate("Rotor 2 fails", xy=(6, 0), xytext=(7.5, 0.2),
    #              arrowprops=dict(arrowstyle='->', lw=1.5))

    plt.xlabel("Time, sec")
    plt.ylabel("Position")
    plt.legend(loc="right")
    plt.tight_layout()

    fig3 = plt.figure()
    ax1 = fig3.add_subplot(4, 1, 1)
    ax2 = fig3.add_subplot(4, 1, 2, sharex=ax1)
    ax3 = fig3.add_subplot(4, 1, 3, sharex=ax1)
    ax4 = fig3.add_subplot(4, 1, 4, sharex=ax1)

    ax1.plot(data['t'], data['gamma'].squeeze()[:, 0])
    ax2.plot(data['t'], data['gamma'].squeeze()[:, 1])
    ax3.plot(data['t'], data['gamma'].squeeze()[:, 2])
    ax4.plot(data['t'], data['gamma'].squeeze()[:, 3])

    ax1.set_ylabel('gamma1')
    ax1.grid(True)
    ax2.set_ylabel('gamma2')
    ax2.grid(True)
    ax3.set_ylabel('gamma3')
    ax3.grid(True)
    ax4.set_ylabel('gamma4')
    ax4.grid(True)
    ax4.set_xlabel('Time [sec]')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    exp1()
    exp1_plot()
