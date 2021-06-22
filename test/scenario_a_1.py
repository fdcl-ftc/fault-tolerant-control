import numpy as np
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging
from fym.utils.rot import angle2quat, quat2angle

from ftc.models.multicopter import Multicopter
from ftc.agents.CA import ConstrainedCA
from ftc.agents.fdi import SimpleFDI
from ftc.faults.actuator import LoE, LiP, Float
import ftc.agents.lqr as lqr
from copy import deepcopy


class ActuatorDynamcs(BaseSystem):
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def set_dot(self, rotors, rotors_cmd):
        self.dot = - 1 / self.tau * (rotors - rotors_cmd)


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=20)
        self.plant = Multicopter()
        self.trim_forces = np.vstack([self.plant.m * self.plant.g, 0, 0, 0])
        n = self.plant.mixer.B.shape[1]

        # Define actuator dynamics
        # self.act_dyn = ActuatorDynamcs(tau=0.01, shape=(n, 1))

        # Define faults
        self.sensor_faults = []
        self.actuator_faults = [
            LoE(time=3, index=0, level=0.),  # scenario a
            # LoE(time=6, index=2, level=0.),  # scenario b
        ]

        # Define FDI
        self.fdi = SimpleFDI(no_act=n, tau=0.1, threshold=0.1)

        # Define agents
        self.CCA = ConstrainedCA(self.plant.mixer.B)
        self.controller = lqr.LQRController(self.plant.Jinv,
                                            self.plant.m,
                                            self.plant.g)
        # self.controller2 = SecondController()

        self.detection_time = [[] for _ in range(len(self.actuator_faults))]

    def step(self):
        *_, done = self.update()
        return done

    def control_allocation(self, forces, What):
        fault_index = self.fdi.get_index(What)

        if len(fault_index) == 0:
            rotors = np.linalg.pinv(self.plant.mixer.B.dot(What)).dot(forces)
        else:
            rotors = self.CCA.solve_lp(fault_index, forces,
                                       self.plant.rotor_min,
                                       self.plant.rotor_max)
        return rotors

    def get_ref(self, t):
        pos_des = np.vstack([-1, 1, 2])
        vel_des = np.vstack([0, 0, 0])
        quat_des = np.vstack([1, 0, 0, 0])
        omega_des = np.vstack([0, 0, 0])
        ref = np.vstack([pos_des, vel_des, quat_des, omega_des])

        return ref

    def _get_derivs(self, t, x, What):
        # Set sensor faults
        for sen_fault in self.sensor_faults:
            x = sen_fault(t, x)

        fault_index = self.fdi.get_index(What)
        ref = self.get_ref(t)

        forces = self.controller.get_FM(x, ref)

        # Controller
        if len(fault_index) == 0:
            rotors_cmd = self.control_allocation(forces, What)

        # Switching logic
        elif len(fault_index) >= 1:
            if len(self.detection_time[len(fault_index) - 1]) == 0:
                # print(t)
                self.detection_time[len(fault_index) - 1] == [t]
            rotors_cmd = self.control_allocation(forces, What)

        # actuator saturation
        _rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)
        rotors = deepcopy(_rotors)

        # Set actuator faults
        for act_fault in self.actuator_faults:
            rotors = act_fault(t, rotors)

        _rotors[fault_index] = 1
        W = self.fdi.get_true(rotors, _rotors)

        return rotors_cmd, W, rotors

    def set_dot(self, t):
        mult_states = self.plant.state
        What = self.fdi.state
        ref = self.get_ref(t)
        # rotors = states["act_dyn"]

        rotors_cmd, W, rotors = self._get_derivs(t, mult_states, What)

        self.plant.set_dot(t, rotors)
        self.fdi.set_dot(W)

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref)


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
    plt.ylim([-2.1, 40.1])
    plt.legend()

    plt.subplot(322, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 1], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 1], "r--")
    plt.ylim([-2.1, 40.1])

    plt.subplot(323, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 2], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 2], "r--")
    plt.ylim([-2.1, 40.1])

    plt.subplot(324, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 3], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 3], "r--")
    plt.ylim([-2.1, 40.1])

    plt.subplot(325, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 4], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 4], "r--")
    plt.ylim([-2.1, 40.1])

    plt.subplot(326, sharex=ax)
    plt.plot(data["t"], data["rotors"][:, 5], "k-")
    plt.plot(data["t"], data["rotors_cmd"][:, 5], "r--")
    plt.ylim([-2.1, 40.1])

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

    plt.show()


if __name__ == "__main__":
    exp1()
    exp1_plot()
