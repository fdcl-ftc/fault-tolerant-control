import fym
from fym.utils.rot import quat2angle
import numpy as np


class LQRController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        m, g, Jinv = env.plant.m, env.plant.g, env.plant.Jinv
        self.trim_forces = np.vstack([m * g, 0, 0, 0])

        A = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, -g, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, g, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        B = np.array([[0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [-1/m, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, 0, 0, 0],
                      [0, Jinv[0, 0], 0, 0],
                      [0, 0, Jinv[1, 1], 0],
                      [0, 0, 0, Jinv[2, 2]]])

        self.K, *_ = fym.agents.LQR.clqr(A, B, env.Q, env.R)

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])

        posd, posd_dot = env.get_ref(t, "posd", "posd_dot")

        x = np.vstack((pos, vel, ang, omega))
        x_ref = np.vstack((posd, posd_dot, np.zeros((6, 1))))
        forces = - self.K.dot(x - x_ref) + self.trim_forces

        controller_info = {
            "posd": posd,
            "ang": ang,
            "angd": np.zeros((3, 1)),
        }

        return forces, controller_info
