import numpy as np
from numpy import cos, sin, tan

import fym
from fym.utils.rot import quat2angle, quat2dcm


class GESOController(fym.BaseEnv):
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

        """ Extended State Observer """
        n = 3  # (n-1)-order derivative of disturbance
        l = 4  # observer output dimension
        self.obsv = fym.BaseSystem(np.zeros((l*(n + 1), 1)))
        self.B = np.zeros((l*(n+1), l))
        self.C = np.zeros((l*(n+1), l)).T
        self.B[0:l, 0:l] = np.eye(l)
        self.C[0:l, 0:l] = np.eye(l)
        self.A = np.eye(l*(n+1), l*(n+1), l)
        wb = 20
        if n == 3:
            clist = np.array([4, 6, 4, 1])
            llist = clist * wb ** np.array([1, 2, 3, 4])
        elif n == 2:
            clist = np.array([3, 3, 1])
            llist = clist * wb ** np.array([1, 2, 3])
        elif n == 1:
            clist = np.array([2, 1])
            llist = clist * wb ** np.array([1, 2])
        L = []
        for lval in llist:
            L.append(lval*np.eye(l))
        self.L = np.vstack(L)

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])

        posd, posd_dot = env.get_ref(t, "posd", "posd_dot")

        """ outer-loop control """
        # xo, xod = pos[0:2], posd[0:2]
        # xo_dot, xod_dot = vel[0:2], posd_dot[0:2]
        # eo, eo_dot = xo - xod, xo_dot - xod_dot
        # Ko1 = 0.5*np.diag((3, 1))
        # Ko2 = 0.5*np.diag((3, 2))
        # nuo = (- Ko1 @ eo - Ko2 @ eo_dot) / env.plant.g
        # angd = np.vstack((nuo[1], - nuo[0], 0))
        angd = np.deg2rad(np.vstack((0, 0, 0)))

        """ inner-loop control """
        xi = np.vstack((pos[2], ang))
        xid = np.vstack((posd[2], angd))
        xi_dot = np.vstack((vel[2], omega))
        xid_dot = np.vstack((posd_dot[2], 0, 0, 0))
        ei = xi - xid
        ei_dot = xi_dot - xid_dot
        Ki1 = 5*np.diag((5, 10, 10, 10))
        Ki2 = 1*np.diag((5, 10, 10, 10))
        f = np.vstack((env.plant.g,
                       - env.plant.Jinv @ np.cross(omega, env.plant.J @ omega, axis=0)))
        g = np.zeros((4, 4))
        g[0, 0] = quat2dcm(quat).T[2, 2] / env.plant.m
        g[1:4, 1:4] = env.plant.Jinv

        nui_N = np.linalg.inv(g) @ (- f - Ki1 @ ei - Ki2 @ ei_dot)
        dhat = self.obsv.state[4:8]
        nui_E = - dhat
        nui = nui_N + nui_E
        # nui = nui_N
        self.u_star = nui + np.linalg.inv(g) @ f

        th = np.linalg.pinv(self.B_r2f) @ nui
        pwms_rotor = (th / self.c_th) * 1000 + 1000
        ctrls = np.vstack((
            pwms_rotor,
            np.vstack(env.plant.u_trims_fixed)
        ))

        controller_info = {
            "posd": posd,
            "angd": angd,
            "ang": ang,
            "d": self.get_d(env.get_Lambda(t), ctrls),
            "dhat": dhat,
        }

        return ctrls, controller_info

    def set_dot(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        x_hat = self.obsv.state
        y_hat = self.C.dot(x_hat)
        y = np.vstack((env.plant.m * vel[2], env.plant.J.dot(omega)))
        self.obsv.dot = (
            self.A @ x_hat + self.B @ self.u_star + self.L @ (y - y_hat)
        )

    def get_d(self, Lambda, ctrls):
        dc = (Lambda - np.ones((ctrls.shape))) * ctrls
        dth = dc[:6] / 1000 * self.c_th
        return self.B_r2f @ dth