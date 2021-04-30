import numpy as np
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging

from ftc.models.multicopter import Multicopter
from ftc.faults.actuator import LoE, LiP, Float


class FDI(BaseSystem):
    def __init__(self, no_act, tau):
        super().__init__(np.eye(no_act))
        self.tau = tau

    def get_true(self, u, uc):
        w = np.hstack([
            ui / uci
            if (ui != 0 and uci != 0) else 1
            if (ui == 0 and uci == 0) else 0
            for ui, uci in zip(u, uc)])
        return np.diag(w)

    def set_dot(self, W):
        What = self.state
        self.dot = - 1 / self.tau * (What - W)


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=20)
        self.plant = Multicopter()

        # Define faults
        self.sensor_faults = []
        self.actuator_faults = [
            LoE(time=3, index=1, level=0.5),
            LoE(time=5, index=2, level=0.2),
            LoE(time=7, index=1, level=0.1),
            Float(time=10, index=0),
        ]

        # Define FDI
        self.fdi = FDI(no_act=self.plant.mixer.B.shape[1], tau=0.1)

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
        return np.vstack((50, 0, 0, 0))

    def control_allocation(self, f, What):
        return np.linalg.pinv(self.plant.mixer.B.dot(What)).dot(f)

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

    plt.show()


if __name__ == "__main__":
    exp1()
    exp1_plot()
