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
        ) = env.x_trims_FW
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.u_trims_vtol_FW
        pusher_trim, dels_trim = env.u_trims_fixed_FW

        self.x_trims_FW = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims_FW = np.vstack((rotor_trim, pusher_trim, dels_trim))

        (
            pos_trim,
            vel_trim,
            quat_trim,
            omega_trim,
        ) = env.x_trims_HV
        ang_trim = np.vstack(quat2angle(quat_trim)[::-1])

        rotor_trim = env.u_trims_vtol_HV
        pusher_trim, dels_trim = env.u_trims_fixed_HV

        self.x_trims_HV = np.vstack((pos_trim, vel_trim, ang_trim, omega_trim))
        self.u_trims_HV = np.vstack((rotor_trim, pusher_trim, dels_trim))

        ptrb = 1e-9

        A_FW, B_FW = env.plant.lin_model(self.x_trims_FW, self.u_trims_FW, ptrb)
        A_HV, B_HV = env.plant.lin_model(self.x_trims_HV, self.u_trims_HV, ptrb)

        self.K_HV, *_ = fym.agents.LQR.clqr(A_HV, B_HV[:, :6], env.Q_HV, env.R_HV)
        self.K_FW, *_ = fym.agents.LQR.clqr(A_FW, B_FW[:, 6:], env.Q_FW, env.R_FW)

    def get_control(self, t, env):
        w_r = env.get_ref(t, "w_r")[0]
        K = np.concatenate((w_r * self.K_HV, (1 - w_r) * self.K_FW), axis=0)

        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        x = np.vstack((pos, vel, ang, omega))

        posd, veld = env.get_ref(t, "posd", "posd_dot")
        angd = np.zeros((3, 1))
        omegad = np.zeros((3, 1))
        x_ref = np.vstack((posd, veld, angd, omegad))

        if w_r == 0:
            ctrls = -K.dot(x - x_ref) + self.u_trims_FW
        elif w_r == 1:
            ctrls = -K.dot(x - x_ref) + self.u_trims_HV

        controller_info = {
            "posd": posd,
            "veld": veld,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
        }

        return ctrls, controller_info
