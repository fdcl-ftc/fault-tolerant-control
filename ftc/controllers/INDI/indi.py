import numpy as np
from numpy import cos, sin, tan

import fym
from fym.utils.rot import quat2angle, quat2dcm


class INDIController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.dx1, self.dx2, self.dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        self.dy1, self.dy2 = env.plant.dy1, env.plant.dy2
        c, self.c_th = 0.0338, 128  # tq / th, th / rcmds
        self.B_r2f = np.array((
            [-1, -1, -1, -1, -1, -1],
            [-self.dy2, self.dy1, self.dy1, -self.dy2, -self.dy2, self.dy1],
            [-self.dx2, -self.dx2, self.dx1, -self.dx3, self.dx1, -self.dx3],
            [-c, c, -c, c, c, -c]
        ))
        self.lpf_dxi = fym.BaseSystem(np.zeros((4, 1)))
        self.tau = 0.1

    def get_control(self, t, env, u0):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])

        posd, posd_dot = env.get_ref(t, "posd", "posd_dot")

        """ outer-loop control """
        xo, xod = pos[0:2], posd[0:2]
        xo_dot, xod_dot = vel[0:2], posd_dot[0:2]
        eo, eo_dot = xo - xod, xo_dot - xod_dot
        Ko1 = 0.5*np.diag((3, 1))
        Ko2 = 0.5*np.diag((3, 2))
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
        Ki1 = 5*np.diag((5, 10, 10, 10))
        Ki2 = 1*np.diag((5, 10, 10, 10))
        g = np.zeros((4, 4))
        g[0, 0] = quat2dcm(quat).T[2, 2] / env.plant.m
        g[1:4, 1:4] = env.plant.Jinv
        nui = - Ki1 @ ei - Ki2 @ ei_dot

        """ control increment """
        xi_dot_f = self.lpf_dxi.state
        ddxi = (xi_dot - xi_dot_f) / self.tau
        du = np.linalg.inv(g) @ (nui - ddxi)

        """ set derivatives """
        self.lpf_dxi.dot = - (xi_dot_f - xi_dot) / self.tau

        """ controls """
        nu0 = self.B_r2f @ ((u0[:6] - 1000) / 1000 * self.c_th)
        nu = nu0 + du

        th = np.linalg.pinv(self.B_r2f) @ nu
        pwms_rotor = (th / self.c_th) * 1000 + 1000

        ctrls = np.vstack((
            pwms_rotor,
            np.vstack(env.plant.u_trims_fixed)
        ))

        controller_info = {
            "posd": posd,
            "angd": angd,
            "ang": ang,
            "dxic": xi_dot,
            "dxi": xi_dot_f,
        }
        return ctrls, controller_info
