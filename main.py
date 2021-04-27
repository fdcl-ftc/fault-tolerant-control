import numpy as np

from fym.core import BaseEnv, BaseSystem
from fym.logging import Logger

from ftc.models.multicopter import Multicopter
from ftc.faults.actuator import LoE, LiP, Float, HardOver


class Env(BaseEnv):
    def __init__(self):
        super().__init__(dt=0.01, max_t=20)
        self.plant = Multicopter()

        # Define faults
        self.sensor_faults = []
        self.actuator_faults = []
        # self.actuator_faults = [
        #     LoE(time=3, index=1, level=0.5),
        #     LoE(time=5, index=2, level=0.2),
        #     LiP(time=7, index=1),
        #     Float(time=10, index=0),
        #     HardOver(time=12, index=3,
        #              limit=self.plant.umax, rate=self.plant.udot_max),
        # ]

    def step(self):
        *_, done = self.update()
        return done

    def set_dot(self, t):
        x = self.plant.state

        # Set sensor faults
        for sen_fault in self.sensor_faults:
            x = sen_fault(t, x)

        f = self.get_forces(x)
        u = self.plant.mixer(f)  # Control surfaces

        # Set actuator faults
        for act_fault in self.actuator_faults:
            u = act_fault(t, u)

        self.plant.set_dot(t, u)

    def get_forces(self, x):
        return np.zeros((4, 1))


def run():
    env = Env()
    env.logger = Logger("data.h5")

    env.reset()

    while True:
        env.render()
        done = env.step()

        if done:
            break

    env.close()


if __name__ == "__main__":
    run()
