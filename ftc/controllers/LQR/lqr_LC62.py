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

        posd, veld = env.get_ref(t, "posd", "posd_dot")
        angd = self.x_trims[6:9]
        omegad = self.x_trims[9:]
        x_ref = np.vstack((posd, veld, angd, omegad))

        ctrls = -self.K.dot(x - x_ref) + self.u_trims

        controller_info = {
            "posd": posd,
            "veld": veld,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
        }

        return ctrls, controller_info

class LQR_modeController(fym.BaseEnv):
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
        self.FM_trim = env.plant.get_FM(
            pos_trim, vel_trim, quat_trim, omega_trim, self.u_trims
        )

        ptrb = 1e-9

        A, B = env.plant.lin_model_FM(self.x_trims, self.FM_trim, ptrb)
        self.K, *_ = fym.agents.LQR.clqr(A, B, env.Q, env.R)

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


class LQR_FMCAController(fym.BaseEnv):
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
        self.FM_trim = env.plant.get_FM(
            pos_trim, vel_trim, quat_trim, omega_trim, self.u_trims
        )
        self.B_Gravity = env.plant.B_Gravity(quat_trim)

        ptrb = 1e-9

        A, B = env.plant.lin_model_FM(self.x_trims, self.FM_trim, ptrb)
        self.K, *_ = fym.agents.LQR.clqr(A, B, env.Q, env.R)

    def CA_Bmatrix(self, env):
        # 6x6 B_VTOL matrix w.r.t. thrust of each rotors
        dx1, dx2, dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        dy1, dy2 = env.plant.dy1, env.plant.dy2
        self.Cth_r = np.array(([178, -22]))  # th_r = 178*rcmds-22
        Ctqth_r = 0.0338  # tq_r / th_r
        self.B_r2FM = np.array(
            (
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1, -1],
                [-dy2, dy1, dy1, -dy2, -dy2, dy1],
                [-dx2, -dx2, dx1, -dx3, dx1, -dx3],
                [-Ctqth_r, Ctqth_r, -Ctqth_r, Ctqth_r, Ctqth_r, -Ctqth_r],
            )
        )

        # 6x2 B_Pusher matrix w.r.t. thrust of each pushers
        # self.Cth_p = 108.2 # th_p = 108.2*rcmds**2
        self.Cth_p = np.array(([84, -12]))  # th_p = 84*rcmds-12
        Ctqth_p = 0.0835  # tq_p / th_p
        self.B_p2FM = np.array(
            ([1, 1], [0, 0], [0, 0], [Ctqth_p, -Ctqth_p], [0, 0], [0, 0])
        )

        # 6x3 B_Fuselage matrix w.r.t. dels
        pos = self.x_trims[0:3]
        vel = self.x_trims[3:6]
        rho = env.plant.get_rho(-pos[2])
        VT = np.linalg.norm(vel)
        qbar = 0.5 * rho * VT**2
        alp = np.arctan2(vel[2], vel[0])
        CL, CD, CM = env.plant.aero_coeff(alp)
        S, S2, St = env.plant.S, env.plant.S2, env.plant.St
        b, c, d = env.plant.b, env.plant.c, env.plant.d

        Cy_del_R, Cll_del_A, Cll_del_R = (
            env.plant.Cy_del_R,
            env.plant.Cll_del_A,
            env.plant.Cll_del_R,
        )
        Cm_del_A, Cm_del_E, Cn_del_A, Cn_del_R = (
            env.plant.Cm_del_A,
            env.plant.Cm_del_E,
            env.plant.Cn_del_A,
            env.plant.Cn_del_R,
        )

        self.B_dels2FM = qbar * np.array(
            (
                [0, 0, 0],
                [0, 0, S * Cy_del_R],
                [0, 0, 0],
                [S2 * b * Cll_del_A, 0, S * b * Cll_del_R],
                [S * c * Cm_del_A, S * c * Cm_del_E, 0],
                [St * d * Cn_del_A, 0, St * d * Cn_del_R],
            )
        )

        self.B_ctrl2FM = np.hstack((self.B_r2FM, self.B_p2FM, self.B_dels2FM))
        self.B_const = qbar * S * np.vstack((-CD, 0, -CL, 0, c * CM, 0))

        return self.B_ctrl2FM, self.B_const

    def get_control(self, t, env):
        # present state 
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        x = np.vstack((pos, vel, ang, omega))

        # reference state
        # x_ref = self.x_trims
        posd, veld = env.get_ref(t, "posd", "posd_dot")
        angd = self.x_trims[6:9]
        omegad = self.x_trims[9:]
        x_ref = np.vstack((posd, veld, angd, omegad))

        # get control
        # ctrls = -self.K.dot(x - x_ref) + self.u_trims

        FM_ctrl = -self.K.dot(x - x_ref)
        B_ctrl2FM, B_const = self.CA_Bmatrix(env)
        B_FM2th = np.linalg.pinv(B_ctrl2FM)

        th_ctrl = B_FM2th.dot(FM_ctrl)
        th_r = th_ctrl[0:6]
        th_p = th_ctrl[6:8]
        dels = th_ctrl[8:11] + self.u_trims[8:]

        self.cmd_trims = np.copy(self.u_trims)
        self.cmd_trims[:8] = env.plant.pwm2cmd(self.cmd_trims[:8])

        rcmds_r = (th_r / self.Cth_r[0]) + self.cmd_trims[:6]
        rcmds_p = (th_p / self.Cth_p[0]) + self.cmd_trims[6:8]

        pwms_rotor = env.plant.cmd2pwm(rcmds_r)
        pwms_pusher = env.plant.cmd2pwm(rcmds_p)

        ctrls = np.vstack((pwms_rotor, pwms_pusher, dels))
        controller_info = {
            "posd": posd,
            "veld": veld,
            "angd": angd,
            "omegad": omegad,
            "ang": ang,
        }

        return ctrls, controller_info
