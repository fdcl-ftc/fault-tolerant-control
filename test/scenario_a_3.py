import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

from fym.core import BaseEnv, BaseSystem
import fym.logging
from fym.utils.rot import angle2quat, quat2angle

from ftc.models.multicopter import Multicopter
from ftc.agents.fdi import SimpleFDI
from ftc.faults.actuator import LoE, LiP, Float
from ftc.agents.backstepping import DirectBacksteppingController


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
            LoE(time=6, index=2, level=0.),  # scenario b
        ]

        # Define FDI
        self.fdi = SimpleFDI(no_act=n, tau=0.1)

        # Define agents
        self.controller = DirectBacksteppingController(
            self.plant.pos.state,
            self.plant.m,
            self.plant.g,
        )

    def step(self):
        *_, done = self.update()
        return done

    def get_ref(self, t):
        # pos_des = np.vstack([0, 0, 0])
        # vel_des = np.vstack([0, 0, 0])
        # quat_des = np.vstack([1, 0, 0, 0])
        # omega_des = np.vstack([0, 0, 0])
        # ref = np.vstack([pos_des, vel_des, quat_des, omega_des])
        pos_des = np.vstack([-1, 1, 2])
        ref = {"pos": pos_des,}

        return ref

    def _get_derivs(self, t, x, What):
        # Set sensor faults
        for sen_fault in self.sensor_faults:
            x = sen_fault(t, x)

        fault_index = self.fdi.get_index(What)
        ref = self.get_ref(t)

        # Controller
        FM, Td_dot, Theta_hat_dot = self.controller.command(
            *self.plant.observe_list(), *self.controller.observe_list(),
            self.plant.m, self.plant.J, np.vstack((0, 0, self.plant.g)), self.plant.mixer.B,
        )

        Theta_hat = self.controller.Theta_hat.state
        rotors_cmd = (self.plant.mixer.Binv + Theta_hat) @ FM
        rotors = np.clip(rotors_cmd, self.plant.rotor_min, self.plant.rotor_max)

        # actuator saturation
        _rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)
        rotors = deepcopy(_rotors)
        # Set actuator faults
        for act_fault in self.actuator_faults:
            rotors = act_fault(t, rotors)

        _rotors[fault_index] = 1
        W = self.fdi.get_true(rotors, _rotors)

        # return rotors_cmd, W, rotors
        return rotors_cmd, W, rotors, Td_dot, Theta_hat_dot, ref["pos"]
        # Set actuator faults
        # for act_fault in self.actuator_faults:
        #     rotors = act_fault(t, rotors)

        # W = self.fdi.get_true(rotors, rotors_cmd)
        # # it works on failure only
        # W[fault_index, fault_index] = 0

        # # return rotors_cmd, W, rotors
        # return rotors_cmd, W, rotors, Td_dot, Theta_hat_dot, ref["pos"]

    def set_dot(self, t):
        x = self.plant.state
        What = self.fdi.state
        # rotors = self.act_dyn.state

        # rotors_cmd, W, rotors = self._get_derivs(t, x, What)
        rotors_cmd, W, rotors, Td_dot, Theta_hat_dot, pos_cmd = self._get_derivs(t, x, What)

        self.plant.set_dot(t, rotors)
        self.fdi.set_dot(W)
        # self.act_dyn.set_dot(rotors, rotors_cmd)
        self.controller.set_dot(Td_dot, Theta_hat_dot, pos_cmd)

    def logger_callback(self, i, t, y, *args):
        states = self.observe_dict(y)
        x_flat = self.plant.state
        x = states["plant"]
        What = states["fdi"]
        # rotors = states["act_dyn"]
        Theta_hat = self.controller.Theta_hat.state

        # rotors_cmd, W, rotors = self._get_derivs(t, x_flat, What)
        rotors_cmd, W, rotors, Td_dot, Theta_hat_dot, pos_cmd = self._get_derivs(t, x, What)
        return dict(
            t=t, x=x, What=What, rotors=rotors, rotors_cmd=rotors_cmd, W=W,
            pos_cmd=pos_cmd, adaptation_params=Theta_hat.flatten(),
        )


def run():
    env = Env()
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


def exp1():
    run()


def exp1_plot():
    data, info = fym.logging.load("data.h5", with_info=True)

    # FDI
    plt.figure()
    plt.suptitle("FDI")

    ax = plt.subplot(321)
    for i in range(data["W"].shape[1]):
        if i is not 0:
            plt.subplot(321+i, sharex=ax)
        plt.ylim([0-0.1, 1+0.1])
        plt.plot(data["t"], data["W"][:, i, i], "r--", label="Actual")
        plt.plot(data["t"], data["What"][:, i, i], "k-", label="Estimated")
        if i == 0:
            plt.legend()
        if i == 2:
            plt.ylabel("Effectiveness")

    # # rotor
    fig, axes = plt.subplots(3, 2)
    ax = plt.subplot(321)
    for i in range(data["rotors"].shape[1]):
        if i is not 0:
            plt.subplot(321+i, sharex=ax)
        plt.ylim([info["rotor_min"]-5, info["rotor_max"]+5])
        plt.plot(data["t"], data["rotors"][:, i], "k-", label="Response")
        plt.plot(data["t"], data["rotors_cmd"][:, i], "r--", label="Command")

    fig.suptitle("Rotor inputs")
    fig.supxlabel("Time (sec)")
    fig.supylabel("Rotor input")
    plt.legend()

    # adaptation parameter
    fig, axes = plt.subplots(6, 4)
    ax = plt.subplot(6, 4, 1)
    for i in range(data["adaptation_params"].shape[1]):
        if i is not 0:
            plt.subplot(6, 4, 1+i, sharex=ax)
        plt.plot(data["t"], data["adaptation_params"][:, i], "k-")

    fig.suptitle("Adaptation parameter")
    fig.supxlabel("Time (sec)")
    fig.supylabel("Adaptation parameter")
    plt.legend()

    # position
    plt.figure()
    plt.title("Position")
    plt.xlabel("Time (sec)")
    plt.ylabel("Position (m)")
    plt.ylim([-5, 5])

    for (i, _label, _ls) in zip(range(data["x"]["pos"].shape[1]), ["x", "y", "z"], ["-", "--", "-."]):
        plt.plot(data["t"], data["x"]["pos"][:, i, 0], "k"+_ls, label=_label)
        plt.plot(data["t"], data["pos_cmd"][:, i, 0], "r"+_ls, label=_label+" (cmd)")
    plt.legend()
    # velocity
    plt.figure()
    plt.title("Velocity")
    plt.xlabel("Time (sec)")
    plt.ylabel("Velocity (m/s)")
    plt.ylim([-5, 5])

    for (i, _label, _ls) in zip(range(data["x"]["vel"].shape[1]), ["x", "y", "z"], ["-", "--", "-."]):
        plt.plot(data["t"], data["x"]["vel"][:, i, 0], "k"+_ls, label=_label)
    plt.legend()
    # euler angles
    plt.figure()
    plt.title("Euler angles")
    plt.xlabel("Time (sec)")
    plt.ylabel("Euler angles (deg)")
    plt.ylim([-40, 40])

    angles = np.vstack([quat2angle(data["x"]["quat"][j, :, 0]) for j in range(len(data["x"]["quat"][:, 0, 0]))])
    for (i, _label, _ls) in zip(range(angles.shape[1]), ["yaw", "pitch", "roll"], ["-", "--", "-."]):
        plt.plot(data["t"], np.rad2deg(angles[:, i]), "k"+_ls, label=_label)
    plt.legend()
    # angular rates
    plt.figure()
    plt.title("Angular rates")
    plt.xlabel("Time (sec)")
    plt.ylabel("Angular rates (deg/s)")
    plt.ylim([-10, 10])

    for (i, _label, _ls) in zip(range(data["x"]["omega"].shape[1]), ["x", "y", "z"], ["-", "--", "-."]):
        plt.plot(data["t"], np.rad2deg(data["x"]["omega"][:, i, 0]), "k"+_ls, label=_label)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    exp1()
    exp1_plot()
