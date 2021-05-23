import numpy as np
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging
from fym.utils.rot import angle2quat, quat2angle
from fym.utils.linearization import jacob_analytic
from fym.agents.LQR import LQR

from ftc.models.multicopter import Multicopter
from ftc.agents.CA import Grouping
from ftc.agents.CA import CA
from ftc.agents.fdi import SimpleFDI
from ftc.faults.actuator import LoE, LiP, Float
import ftc.agents.lqr as lqr


class ActuatorDynamcs(BaseSystem):
    def __init__(self, tau, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau

    def set_dot(self, rotors, rotors_cmd):
        self.dot = - 1 / self.tau * (rotors - rotors_cmd)


class LQRLibrary:
    def __init__(self, plant):
        self.plant = plant
        self.xtrim, self.utrim = self.get_trims(alt=1)

        # Get the linear model
        self.A = jacob_analytic(self.deriv, 0)(self.xtrim, self.utrim)
        self.B = jacob_analytic(self.deriv, 1)(self.xtrim, self.utrim)

        # Get the optimal gain
        self.K, self.P = LQR.clqr(self.A, self.B, cfg.agent.Q, cfg.agent.R)

    def deriv(self, x, u):
        pos, vel, angle, omega = x[:3], x[3:6], x[6:9], x[9:12]
        R = self.plant.angle2R(angle)
        dpos, dvel, _, domega = self.plant.deriv(pos, vel, R, omega, u)
        phi, theta, _ = angle.ravel()
        dangle = self.omega2dangle(omega, phi, theta)
        xdot = np.vstack((dpos, dvel, dangle, domega))
        return xdot

    def transform(self, y):
        return np.vstack((y[0:6], np.vstack(quat2angle(y[6:10])[::-1]), y[10:]))

    def get_forces(self, obs, ref):
        pass

    def get_trims(self, alt=1):
        pos = np.vstack((0, 0, -alt))
        vel = angle = omega = np.zeros((3, 1))
        u = np.vstack([self.plant.m * self.plant.g / 4] * 4)
        return np.vstack((pos, vel, angle, omega)), u

    def omega2dangle(self, omega, phi, theta):
        dangle = np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ]) @ omega
        return dangle


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
        self.fdi = SimpleFDI(no_act=n, tau=0.1)

        # Define agents
        self.grouping = Grouping(self.plant.mixer.B)
        self.CA = CA(self.plant.mixer.B)
        self.controller = lqr.LQRController(self.plant.Jinv,
                                            self.plant.m,
                                            self.plant.g)

        self.controller2 = LQRLibrary(self.plant)

    def step(self):
        *_, done = self.update()
        return done

    def control_allocation(self, forces, What):
        fault_index = self.fdi.get_index(What)

        if len(fault_index) == 0:
            rotors = np.linalg.pinv(self.plant.mixer.B.dot(What)).dot(forces)
        else:
            BB = self.CA.get(fault_index)
            rotors = np.linalg.pinv(BB.dot(What)).dot(forces)

        # actuator saturation
        rotors = np.clip(rotors, 0, self.plant.rotor_max)

        return rotors

    def get_ref(self, t):
        pos_des = np.vstack([0, 0, 0])
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

        # Controller
        forces = self.controller.get_forces(x, ref)

        # Switching logic
        if len(fault_index) >= 1:
            forces = self.controller2.get_forces(x, ref, fault_index)

        rotors = rotors_cmd = self.control_allocation(forces, What)

        # Set actuator faults
        for act_fault in self.actuator_faults:
            rotors = act_fault(t, rotors)

        W = self.fdi.get_true(rotors, rotors_cmd)
        # it works on failure only
        W[fault_index, fault_index] = 0

        return rotors_cmd, W, rotors

    def set_dot(self, t):
        x = self.plant.state
        What = self.fdi.state
        # rotors = self.act_dyn.state

        rotors_cmd, W, rotors = self._get_derivs(t, x, What)

        self.plant.set_dot(t, rotors)
        self.fdi.set_dot(W)
        # self.act_dyn.set_dot(rotors, rotors_cmd)

    def logger_callback(self, i, t, y, *args):
        states = self.observe_dict(y)
        x_flat = self.plant.observe_vec(y[self.plant.flat_index])
        x = states["plant"]
        What = states["fdi"]
        # rotors = states["act_dyn"]

        rotors_cmd, W, rotors = self._get_derivs(t, x_flat, What)
        return dict(t=t, x=x, What=What, rotors=rotors, rotors_cmd=rotors_cmd,
                    W=W)


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
    plt.plot(data["t"], data["W"][:, 0, 0], "r--", label="Fault")
    plt.plot(data["t"], data["What"][:, 0, 0], "k-", label="estimated")
    plt.legend()

    plt.subplot(322, sharex=ax)
    plt.plot(data["t"], data["W"][:, 1, 1], "r--")
    plt.plot(data["t"], data["What"][:, 1, 1], "k-")

    plt.subplot(323, sharex=ax)
    plt.plot(data["t"], data["W"][:, 2, 2], "r--")
    plt.plot(data["t"], data["What"][:, 2, 2], "k-")
    plt.ylabel("FDI")

    plt.subplot(324, sharex=ax)
    plt.plot(data["t"], data["W"][:, 3, 3], "r--")
    plt.plot(data["t"], data["What"][:, 3, 3], "k-")

    plt.subplot(325, sharex=ax)
    plt.plot(data["t"], data["W"][:, 4, 4], "r--")
    plt.plot(data["t"], data["What"][:, 4, 4], "k-")

    plt.subplot(326, sharex=ax)
    plt.plot(data["t"], data["W"][:, 5, 5], "r--")
    plt.plot(data["t"], data["What"][:, 5, 5], "k-")

    # Rotor
    plt.figure()

    ax = plt.subplot(321)
    plt.plot(data["t"], data["rotors_cmd"][:, 0], "r--")
    plt.plot(data["t"], data["rotors"][:, 0], "k-")

    plt.subplot(322, sharex=ax)
    plt.plot(data["t"], data["rotors_cmd"][:, 1], "r--")
    plt.plot(data["t"], data["rotors"][:, 1], "k-")

    plt.subplot(323, sharex=ax)
    plt.plot(data["t"], data["rotors_cmd"][:, 2], "r--")
    plt.plot(data["t"], data["rotors"][:, 2], "k-")
    plt.ylabel("Rotors")

    plt.subplot(324, sharex=ax)
    plt.plot(data["t"], data["rotors_cmd"][:, 3], "r--")
    plt.plot(data["t"], data["rotors"][:, 3], "k-")

    plt.subplot(325, sharex=ax)
    plt.plot(data["t"], data["rotors_cmd"][:, 4], "r--")
    plt.plot(data["t"], data["rotors"][:, 4], "k-")

    plt.subplot(326, sharex=ax)
    plt.plot(data["t"], data["rotors_cmd"][:, 5], "r--")
    plt.plot(data["t"], data["rotors"][:, 5], "k-")

    plt.figure()

    plt.plot(data["t"], data["x"]["pos"][:, 0, 0], "k-", label="x")  # x
    plt.plot(data["t"], data["x"]["pos"][:, 1, 0], "k--", label="y")  # y
    plt.plot(data["t"], -data["x"]["pos"][:, 2, 0], "k-.", label="z")  # z
    plt.legend()

    plt.show()


if __name__ == "__main__":
    exp1()
    exp1_plot()
