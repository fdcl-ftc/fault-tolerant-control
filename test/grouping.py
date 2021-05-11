import numpy as np
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import angle2quat
import fym.logging

from ftc.models.multicopter import Multicopter
from ftc.agents.backstepping import BacksteppingController
from ftc.agents.grouping import Grouping
from ftc.agents.fdi import SimpleFDI
from ftc.faults.actuator import LoE, LiP, Float


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=20)
        self.plant = Multicopter()

        # Define faults
        self.sensor_faults = []
        self.actuator_faults = [
            LoE(time=3, index=0, level=0.5),
            LoE(time=5, index=1, level=0.2),
            LoE(time=7, index=2, level=0.5),
        ]

        # Define FDI
        self.fdi = SimpleFDI(no_act=self.plant.mixer.B.shape[1], tau=0.1)
        self.grouping = Grouping(self.plant.mixer.B)

    def step(self):
        *_, done = self.update()
        return done

    def set_dot(self, t):
        x = self.plant.state
        What = self.fdi.state

        u, W, *_ = self._get_derivs(t, x, What)

        self.plant.set_dot(t, u)
        self.fdi.set_dot(W)

    def get_forces(self, x):
        return np.vstack((self.plant.m * self.plant.g, 0, 0, 0))

    def control_allocation(self, f, What):
        fault_index = self.fdi.get_index(What)
        if len(fault_index) == 0:
            return np.linalg.pinv(self.plant.mixer.B.dot(What)).dot(f)
        else:
            G = self.grouping.get(fault_index[0])
            return np.linalg.pinv(G.dot(What)).dot(f)

    def _get_derivs(self, t, x, What):
        # Set sensor faults
        for sen_fault in self.sensor_faults:
            x = sen_fault(t, x)

        f = self.get_forces(x)
        u = u_command = self.control_allocation(f, What)

        # Set actuator faults
        for act_fault in self.actuator_faults:
            u = act_fault(t, u)

        W = self.fdi.get_true(u, u_command)

        return u, W, u_command

    def logger_callback(self, i, t, y, *args):
        states = self.observe_dict(y)
        x = states["plant"]
        What = states["fdi"]
        u, W, uc = self._get_derivs(t, x, What)
        return dict(t=t, x=x, What=What, u=u, uc=uc, W=W)


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

    plt.figure()
    plt.plot(data["t"], np.diagonal(data["W"], axis1=1, axis2=2), "r--")
    plt.plot(data["t"], np.diagonal(data["What"], axis1=1, axis2=2), "k-")

    plt.figure()
    plt.plot(data["t"], data["x"]["pos"][:, 0, 0], "k-", label="x")  # x
    plt.plot(data["t"], data["x"]["pos"][:, 1, 0], "k--", label="y")  # y
    plt.plot(data["t"], -data["x"]["pos"][:, 2, 0], "k-.", label="z")  # z
    plt.legend()

    plt.show()


if __name__ == "__main__":
    exp1()
    exp1_plot()
