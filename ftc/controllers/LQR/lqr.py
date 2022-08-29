import fym
from fym.utils.rot import quat2angle
import numpy as np


class LQRController(fym.BaseEnv):
    def __init__(self):
        super().__init__()

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])

        posd, posd_dot = env.get_ref(t, "posd", "posd_dot")

        controller_info = {
            "posd": posd,
        }
        x = np.vstack((pos, vel, ang, omega))
        x_ref = np.vstack((posd, posd_dot, np.zeros((6, 1))))
        forces = - env.K.dot(x - x_ref) + env.trim_forces

        return forces, controller_info
