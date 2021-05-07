import numpy as np
from types import SimpleNamespace as SN

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2angle

cfg = SN()


def load_config():
    cfg.K1 = np.diag([10, 10, 10, 10]) * 60
    cfg.K2 = np.diag([10, 10, 10, 10]) * 60
    cfg.Kbar = np.diag([10, 10, 10, 10]) * 60


class DSCController(BaseEnv):
    """Description
    Parameters:
        l: distance between center of the mass and the actuators
        d: drag coefficient
        b: thrust coefficient
    """
    def __init__(self, n, J, m, g, Jr, l, d, b):
        super().__init__()
        self.xi2d = BaseSystem(shape=(4, 1))

        self.J, self.m, self.g, self.Jr = J, m, g, Jr
        self.Jinv = np.linalg.inv(J)
        self.u_factor = np.diag([l, l, d / b])

    def get_x(self, pos, vel, angle, omega):
        x = np.zeros((8, 1))
        x[:6:2, :] = angle
        x[1:7:2] = omega
        x[6] = -pos[2]
        x[7] = -vel[2]
        return x

    def get_h(self, *state):
        x = self.get_x(*state)
        h = x[:8:2]
        return h

    def get_forces(self, t, mult_states, xi2d, windvel, dl_hat, xi1d, xi1d_dot):
        pos, vel, quat, omega = mult_states
        angle = np.vstack(quat2angle(quat)[::-1])

        x = self.get_x(pos, vel, angle, omega)
        h = self.get_h(pos, vel, angle, omega)

        x2, x4, x6, x8 = x[(1, 3, 5, 7), 0]

        c1, c3 = np.cos(x[(0, 2), 0])
        s1, s3 = np.sin(x[(0, 2), 0])
        t3 = s3 / c3

        Jinv, J, m = self.Jinv, self.J, self.m

        f = np.zeros((8, 1))
        f[0:6:2] = np.array([
            [1, s1 * t3, c1 * t3],
            [0, c1, -s1],
            [0, s1/c3, c1/c3]
        ]).dot(omega)
        f[1:7:2] = - Jinv.dot(np.cross(omega, J.dot(omega), axis=0))
        f[6:] = np.vstack((x8, -self.g))

        G = np.zeros((8, 4))
        G[1:7:2, 1:] = Jinv.dot(self.u_factor)
        G[-1, 0] = 1/self.m * c1 * c3

        Jxinv, Jyinv, _ = np.diag(Jinv)
        P = np.zeros((8, 8))
        P[1:6:2, :3] = - Jinv.dot(x[1:6:2])
        P[(1, 3), 3] = -Jxinv * self.Jr * x4, Jyinv * self.Jr * x2
        P[1:6:2, 4:7] = Jinv
        P[-1, -1] = 1/m

        grad_h = np.zeros((8, 4))
        grad_h[:8:2, :] = np.eye(4)

        grad_Lfh = np.array([
            [x4*c1*t3 - x6*s1*t3, - x4*s1 - x6*c1, x4*c1/c3 - x6*s1/c3, 0],
            [1, 0, 0, 0],
            [x4*s1/c3**2 + x6*c1/c3**2, 0, x4*s1*t3/c3 + x6*c1*t3/c3, 0],
            [s1*t3, c1, s1/c3, 0],
            [0, 0, 0, 0],
            [c1*t3, -s1, c1/c3, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1]
        ])

        Lfh = grad_h.T.dot(f)
        LGLfh = grad_Lfh.T.dot(G)
        Lf2h = grad_Lfh.T.dot(f)
        LPLfh = grad_Lfh.T.dot(P)

        xi1 = h
        xi2 = Lfh

        S1 = xi1 - xi1d
        S2 = xi2 - xi2d

        xi2bar = xi1d_dot - cfg.K1.dot(S1)
        S2bar = xi2d - xi2bar

        S1dot = - cfg.K1.dot(S1) + S2 + S2bar

        xi2d_dot = - cfg.Kbar.dot(xi2d - xi2bar) - cfg.K1.dot(S1dot) - S1

        ud = np.linalg.inv(LGLfh).dot(
            - Lf2h - LPLfh.dot(dl_hat) + xi2d_dot - cfg.K2.dot(S2) - S1)

        return ud, xi2d_dot, (S1, S2, S2bar, xi2, xi2bar)

    def set_dot(self, xi2d_dot):
        self.xi2d.dot = xi2d_dot
