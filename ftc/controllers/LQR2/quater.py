import fym
import control
import numpy as np
from fym.utils.rot import quat2angle


class QuaterController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        m, g, Jinv = env.plant.m, env.plant.g, env.plant.Jinv
        # self.trim_forces = np.vstack([m * g, 0, 0, 0])

        Aatt = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        Batt = np.vstack((np.zeros((3, 3)), np.identity(3)))
        Qatt = np.diag((1, 1, 1, 1, 1, 1))
        Ratt = np.diag((1, 1, 1))

        Apos = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        Bpos = np.vstack((np.zeros((3, 3)), np.identity(3)))
        Qpos = np.diag((1, 1, 1, 1, 1, 1))
        Rpos = np.diag((1, 1, 1))

        self.Katt, *_ = control.lqr(Aatt, Batt, Qatt, Ratt)
        self.Kpos, *_ = control.lqr(Apos, Bpos, Qpos, Rpos)

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])

        posd = np.deg2rad(np.vstack((1, 2, 0)))
        veld = np.zeros((3, 1))
        omegad = np.zeros((3, 1))
        angd = np.deg2rad(np.vstack((0, 0, 0)))

        q0 = quat[0]
        qbar = quat[1::]
        qnorm = np.linalg.norm(quat)
        qbarnorm = np.linalg.norm(qbar + 1e-10)
        if qbarnorm == 0:
            lnq = np.zeros((3, 1))
        else:
            lnq = qbar / qbarnorm * np.arccos(q0)

        theta = 2 * lnq

        xatt = np.vstack((theta, omega))
        xattd = np.vstack((angd, omegad))

        xpos = np.vstack((pos, vel))
        xposd = np.vstack((posd, veld))

        torques = -self.Katt @ (xatt - xattd) - self.Kpos @ (xpos - xposd)
        forces = np.vstack((env.plant.m * env.plant.g, torques))
        ctrl = np.linalg.pinv(env.plant.mixer.B) @ forces

        controller_info = {
            "angd": angd,
            "ang": ang,
            "posd": posd,
        }

        return ctrl, controller_info

        # x_pos = np.vstack((pos, vel))
        # x_pos_d = np.vstack((posd, posd_dot))
        # u_pos = -self.Kpos*(x_pos-x_pos_d)

        # theta_bar = 2*quat[1::] / np.norm(quat[1::]+1e-10) * np.acos(quat[0])
        # B=[0,0,1]
        # b= np.vstack.B
        # breakpoint()
        # # print(size(b))
        # quat_prime_d = np.vstack((np.dot(b,u_pos)+np.norm(u_pos), np.cross(b,u_pos)))
        # quat_d= quat_prime_d / np.norm(quat_prime_d)
        # omega_d=np.zeros((3, 1))

        # x_att = np.vstack((theta_bar, omega))
        # x_att_d = np.vstack((2*(quat_d[1::]/np.norm(quat_d[1::])*np.acos(quat_d[0])),omega_d))
        # u_att = -self.Katt*(x_att-x_att_d)

        # F=np.norm(u_pos) - np.vstack((0,0,9.81))
        # tau = u_att
        # ctrls = np.vstack((F, tau))  # virtual control (generalized forces)

        # controller_info = {
        #     "posd": posd,
        #     "ang": ang,
        #     "angd": np.zeros((3, 1)),
        # }

        # return ctrls, controller_info
