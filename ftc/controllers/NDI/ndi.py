import fym
from fym.utils.rot import quat2angle
import numpy as np
from numpy import cos, sin, tan


class NDIController(fym.BaseEnv):
    def __init__(self):
        super().__init__()

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])

        posd, posd_dot = env.get_ref(t, "posd", "posd_dot")

        controller_info = {
            "posd": posd,
        }

        # position error
        ep = pos - posd

        """ position control """

        """ pitch and roll angle control """

        """ angular velocity and vertical velocity control """

        controller_info = {
            "posd": posd,
            "ep": ep,
        }

        return FM0, controller_info
