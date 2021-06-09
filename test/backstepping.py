import numpy as np
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import angle2quat, quat2angle
import fym.logging

from ftc.models.multicopter import Multicopter
from ftc.agents.backstepping import IndirectBacksteppingController, DirectBacksteppingController
from ftc.agents.grouping import Grouping
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

    def get_index(self, W):
        fault_index = np.where(np.diag(W) != 1)[0]
        return fault_index

    def set_dot(self, W):
        What = self.state
        self.dot = - 1 / self.tau * (What - W)


class Env(BaseEnv):
    def __init__(self, method):
        super().__init__(dt=0.01, max_t=20)
        self.method = method
        pos0 = np.zeros((3, 1))
        self.plant = Multicopter(pos=pos0)
        if self.method == "indirect":
            controller = IndirectBacksteppingController(
                self.plant.pos.state,
                self.plant.m,
                self.plant.g,
            )
        elif self.method == "direct":
            controller = DirectBacksteppingController(
                self.plant.pos.state,
                self.plant.m,
                self.plant.g,
            )
        self.controller = controller
        # Define faults
        self.sensor_faults = []
        self.actuator_faults = [
            LoE(time=3, index=0, level=0.0),
            LoE(time=5, index=1, level=0.0),
            # Float(time=10, index=0),
        ]

        # Define FDI
        self.fdi = FDI(no_act=self.plant.mixer.B.shape[1], tau=0.1)
        self.grouping = Grouping(self.plant.mixer.B)

    def step(self):
        *_, done = self.update()
        return done

    def set_dot(self, t):
        x = self.plant.state
        What = self.fdi.state

        if self.method == "indirect":
            u, W, _, Td_dot, pos_c = self._get_derivs(t, x, What)
        elif self.method == "direct":
            u, W, _, Td_dot, Theta_hat_dot, pos_c = self._get_derivs(t, x, What)

        self.plant.set_dot(t, u)
        self.fdi.set_dot(W)
        if self.method == "indirect":
            self.controller.set_dot(Td_dot, pos_c)
        elif self.method == "direct":
            self.controller.set_dot(Td_dot, Theta_hat_dot, pos_c)

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

        pos_c = np.vstack((-1, 1, 2))  # TODO: position commander
        if self.method == "indirect":
            FM, Td_dot = self.controller.command(
                *self.plant.observe_list(), *self.controller.observe_list(),
                self.plant.m, self.plant.J, np.vstack((0, 0, self.plant.g)),
            )
            u_command = self.control_allocation(FM, What)
        elif self.method == "direct":
            FM, Td_dot, Theta_hat_dot = self.controller.command(
                *self.plant.observe_list(), *self.controller.observe_list(),
                self.plant.m, self.plant.J, np.vstack((0, 0, self.plant.g)), self.plant.mixer.B,
            )
            Theta_hat = self.controller.Theta_hat.state
            u_command = (self.plant.mixer.Binv + Theta_hat) @ FM
        u = np.clip(u_command, self.plant.rotor_min, self.plant.rotor_max)

        # Set actuator faults
        for act_fault in self.actuator_faults:
            u = act_fault(t, u)

        W = self.fdi.get_true(u, u_command)

        if self.method == "indirect":
            return u, W, u_command, Td_dot, pos_c
        elif self.method == "direct":
            return u, W, u_command, Td_dot, Theta_hat_dot, pos_c

    def logger_callback(self, i, t, y, *args):
        states = self.observe_dict(y)
        x = states["plant"]
        What = states["fdi"]
        x_controller = states["controller"]
        if self.method == "indirect":
            u, W, uc, Td_dot, pos_c, *_ = self._get_derivs(t, x, What)
            return dict(
                t=t,
                x=x,
                What=What,
                u=u,
                uc=uc,
                W=W,
                x_controller=x_controller,
                pos_c=pos_c
            )
        elif self.method == "direct":
            u, W, uc, Td_dot, Theta_hat_dot, pos_c, *_ = self._get_derivs(t, x, What)
            return dict(
                t=t,
                x=x,
                What=What,
                u=u,
                uc=uc,
                W=W,
                x_controller=x_controller,
                Theta_hat_dot=Theta_hat_dot,
                pos_c=pos_c
            )


def run(method):
    env = Env(method)
    env.logger = fym.logging.Logger("data.h5")

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            env_info = {
                "rotor_min": env.plant.rotor_min,
                "rotor_max": env.plant.rotor_max,
            }
            env.logger.set_info(**env_info)
            break

    env.close()


def exp_indirect():
    run("indirect")


def exp_indirect_plot():
    data, info = fym.logging.load("data.h5", with_info=True)

    plt.figure()
    plt.plot(data["t"], data["x"]["pos"][:, :, 0], "r--", label="pos")  # position
    plt.plot(data["t"], data["pos_c"][:, :, 0], "k--", label="position command")  # position command

    plt.legend()
    plt.show()


def exp_direct():
    run("direct")


def exp_direct_plot():
    data, info = fym.logging.load("data.h5", with_info=True)
    # position
    plt.figure()
    plt.title("position")
    plt.plot(data["t"], data["x"]["pos"][:, :, 0], "r--", label="pos")  # position
    plt.plot(data["t"], data["pos_c"][:, :, 0], "k--", label="position command")  # position command
    plt.legend()
    plt.savefig("position.png")
    # rotor
    plt.figure()
    plt.ylim([info["rotor_min"], info["rotor_max"]])
    plt.title("rotor input")
    for i in range(data["u"].shape[1]):
        plt.plot(data["t"], data["u"][:, i, 0], label=f"u_{i}")
    plt.legend()
    plt.savefig("rotor_input.png")
    # Euler angles
    plt.figure()
    plt.title("Euler angles")
    plt.ylabel("Euler angles (deg)")
    _shape = data["x"]["quat"][:, :, 0].shape
    angles = np.zeros((_shape[0], 3))
    for i in range(_shape[0]):
        _angle = quat2angle(data["x"]["quat"][i, :, 0])
        angles[i, :] = _angle
    plt.plot(data["t"], np.rad2deg(angles[:, 0]), "r--", label="yaw")
    plt.plot(data["t"], np.rad2deg(angles[:, 1]), "g--", label="pitch")
    plt.plot(data["t"], np.rad2deg(angles[:, 2]), "b--", label="roll")
    plt.legend()
    plt.savefig("angles.png")


if __name__ == "__main__":
    # exp_indirect()
    # exp_indirect_plot()
    exp_direct()
    exp_direct_plot()
