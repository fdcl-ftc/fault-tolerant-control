import numpy as np
from numpy import cos, sin, tan

import fym
from fym.utils.rot import quat2angle, quat2dcm


class NDIController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.dx1, self.dx2, self.dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        self.dy1, self.dy2 = env.plant.dy1, env.plant.dy2
        c, self.c_th = 0.0338, 128
        self.B_r2f = np.array((
            [-1, -1, -1, -1, -1, -1],
            [-self.dy2, self.dy1, self.dy1, -self.dy2, -self.dy2, self.dy1],
            [-self.dx2, -self.dx2, self.dx1, -self.dx3, self.dx1, -self.dx3],
            [-c, c, -c, c, c, -c]
        ))

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])

        posd, posd_dot = env.get_ref(t, "posd", "posd_dot")

        """ outer-loop control """
        xo = pos[0:2]
        xod = posd[0:2]
        xo_dot = vel[0:2]
        xod_dot = posd_dot[0:2]
        eo = xo - xod
        eo_dot = xo_dot - xod_dot
        Ko1 = 2*np.diag((1, 1))
        Ko2 = 5*np.diag((1, 1))
        nuo = (- Ko1 @ eo - Ko2 @ eo_dot) / env.plant.g
        angd = np.vstack((nuo[1], - nuo[0], 0))
        # angd = np.deg2rad(np.vstack((0, 0, 0)))

        """ inner-loop control """
        xi = np.vstack((pos[2], ang))
        xid = np.vstack((posd[2], angd))
        xi_dot = np.vstack((vel[2], omega))
        xid_dot = np.vstack((posd_dot[2], 0, 0, 0))
        ei = xi - xid
        ei_dot = xi_dot - xid_dot
        Ki1 = 10*np.diag((1, 10, 10, 10))
        Ki2 = 10*np.diag((1, 10, 10, 10))
        phi, theta, psi = ang.ravel()
        phi_dot, theta_dot, psi_dot = omega.ravel()
        f = np.vstack((env.plant.g, - np.cross(omega, env.plant.J @ omega, axis=0)))
        g = np.zeros((4, 4))
        g[0, 0] = quat2dcm(quat).T[2, 2]/env.plant.m
        g[1:4, 1:4] = env.plant.Jinv
        nui = np.linalg.inv(g) @ (- f - Ki1 @ ei - Ki2 @ ei_dot)

        th = np.linalg.pinv(self.B_r2f) @ nui
        pwms_rotor = (th / self.c_th) * 1000 + 1000
        ctrls = np.vstack((
            pwms_rotor,
            # env.plant.u_trims_vtol,
            np.vstack(env.plant.u_trims_fixed)
            # np.vstack((1000, 1000, 0, 0, 0))
        ))

        controller_info = {
            "posd": posd,
            "angd": angd,
            "ang": ang,
        }

        return ctrls, controller_info
