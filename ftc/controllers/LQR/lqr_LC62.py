import fym
import numpy as np
from fym.utils.rot import quat2angle


class LQRController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        (
            pos_trim,
            vel_trim,
            quat_trim,
            omega_trim,
        ) = env.plant.x_trims
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.plant.u_trims_vtol
        pusher_trim, dels_trim = env.plant.u_trims_fixed

        self.x_trims = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims = np.vstack((rotor_trim, pusher_trim, dels_trim))

        ptrb = 1e-9


        A, B = env.plant.lin_model(self.x_trims, self.u_trims, ptrb)
        self.K, *_ = fym.agents.LQR.clqr(A, B, env.Q, env.R)


    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])


        x = np.vstack((pos, vel, ang, omega))
        x_ref = self.x_trims


        ctrls = -self.K.dot(x - x_ref) + self.u_trims

        controller_info = {
            "posd": self.x_trims[0:3],
            "veld": self.x_trims[3:6],
            "angd": self.x_trims[6:9],
        }

        return ctrls, controller_info


class LQR_FMController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        (
            pos_trim,
            vel_trim,
            quat_trim,
            omega_trim,
        ) = env.plant.x_trims
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.plant.u_trims_vtol
        pusher_trim, dels_trim = env.plant.u_trims_fixed

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
        x_ref = self.x_trims


        FM_ctrl = -self.K.dot(x - x_ref) + self.FM_trim

        controller_info = {
            "posd": self.x_trims[0:3],
            "veld": self.x_trims[3:6],
            "angd": self.x_trims[6:9],
        }


        return FM_ctrl, controller_info

