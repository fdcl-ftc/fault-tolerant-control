"""This is LQR controller for LC62.
should be placed in fault-tolerant-control/ftc/controllers/
"""

import fym
import numpy as np
from fym.utils.rot import quat2angle


class LQR_LC62Controller(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        (
            pos_trim,
            vel_trim,
            quat_trim,
            omega_trim,
        ) = env.plant.x_trims  # plant defined at controller_lqr_LC62.py
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])
        pusher_trim, dels_trim = env.plant.u_trims_fixed
        rotor_trim = env.plant.u_trims_vtol

        self.x_trims = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims = np.vstack((rotor_trim, pusher_trim, dels_trim))

        ptrb = 1e-9

        A, B = env.plant.lin_model(self.x_trims, self.u_trims, ptrb)
        self.K, *_ = fym.agents.LQR.clqr(
            A, B, env.Q, env.R
        )  # Q, R defined at controller_lqr_LC62.py

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        x = np.vstack((pos, vel, ang, omega))
        posd = self.x_trims[0:3]
        veld = self.x_trims[3:6]
        angd = self.x_trims[6:9]
        omegad = self.x_trims[9:12]

        ctrls = -self.K.dot(x - self.x_trims) + self.u_trims

        controller_info = {"posd": posd, "veld": veld, "angd": angd, "omegad": omegad}

        return ctrls, controller_info


class LQR_LC62_FMController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        (
            pos_trim,
            vel_trim,
            quat_trim,
            omega_trim,
        ) = env.plant.x_trims  # plant defined at controller_lqr_LC62.py
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        pusher_trim, dels_trim = env.plant.u_trims_fixed
        rotor_trim = env.plant.u_trims_vtol

        self.x_trims = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims = np.vstack((rotor_trim, pusher_trim, dels_trim))


        self.FM_trim = env.plant.get_FM(pos_trim, vel_trim, quat_trim, omega_trim, self.u_trims)

        ptrb = 1e-9
        A, B = env.plant.lin_model_FM(self.x_trims, self.FM_trim, ptrb) 
        self.K, *_ = fym.agents.LQR.clqr(A,B, env.Q, env.R)

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        x = np.vstack((pos, vel, ang, omega))
        posd = self.x_trims[0:3]
        veld = self.x_trims[3:6]
        angd = self.x_trims[6:9]
        omegad = self.x_trims[9:12]

        FM_ctrl = -self.K.dot(x - self.x_trims) + self.FM_trim 
        
        controller_info = {"posd": posd, "veld": veld, "angd": angd, "omegad": omegad}

        return FM_ctrl, controller_info

