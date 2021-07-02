import numpy as np
import matplotlib.pyplot as plt

import fym
from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import angle2quat, quat2angle

from ftc.models.multicopter import Multicopter
from ftc.agents.CA import ConstrainedCA
from ftc.agents.fdi import SimpleFDI
from ftc.faults.actuator import LoE, LiP, Float
import ftc.agents.lqr as lqr
from ftc.plotting import exp_plot


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
        self.fdi = SimpleFDI(self.actuator_faults,
                             no_act=n, delay=0., threshold=0.1)

        # Define agents
        self.CCA = ConstrainedCA(self.plant.mixer.B)
        self.controller = lqr.LQRController(self.plant.Jinv,
                                            self.plant.m,
                                            self.plant.g)

        self.detection_time = [fault.time + self.fdi.delay for fault in self.actuator_faults]

    def step(self):
        *_, done = self.update()
        return done

    def control_allocation(self, t, forces, What):
        fault_index = self.fdi.get_index(t)

        if len(fault_index) == 0:
            rotors = np.linalg.pinv(self.plant.mixer.B.dot(What)).dot(forces)
        else:
            rotors = self.CCA.solve_opt(fault_index, forces,
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

    def set_dot(self, t):
        mult_states = self.plant.state
        W = self.fdi.get_true(t)
        What = self.fdi.get(t)
        ref = self.get_ref(t)

        # Set sensor faults
        for sen_fault in self.sensor_faults:
            mult_states = sen_fault(t, mult_states)

        # Controller
        forces = self.controller.get_FM(mult_states, ref)

        rotors_cmd = self.control_allocation(t, forces, What)

        # actuator saturation
        rotors = np.clip(rotors_cmd, 0, self.plant.rotor_max)

        # Set actuator faults
        for act_fault in self.actuator_faults:
            rotors = act_fault(t, rotors)

        self.plant.set_dot(t, rotors)

        return dict(t=t, x=self.plant.observe_dict(), What=What,
                    rotors=rotors, rotors_cmd=rotors_cmd, W=W, ref=ref)


def run():
    env = Env()
    env.logger = fym.Logger("data.h5")

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            env_info = {
                "detection_time": env.detection_time,
                "rotor_min": env.plant.rotor_min,
                "rotor_max": env.plant.rotor_max,
            }
            env.logger.set_info(**env_info)
            break

    env.close()


def exp1():
    run()


if __name__ == "__main__":
    exp1()
    loggerpath = "data.h5"
    exp_plot(loggerpath)
