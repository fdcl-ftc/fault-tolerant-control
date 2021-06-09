import numpy as np
from scipy.linalg import solve_lyapunov
from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2dcm


def skew(x):
    x = x.ravel()  # squeeze
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def T_omega(T):
    return np.array([[0, -T, 0],
                     [T, 0, 0],
                     [0, 0, 0]])


def T_u_inv(T):
    return np.array([[0, 1/T, 0], [-1/T, 0, 0], [0, 0, -1]])


def T_u_inv_dot(T, T_dot):
    return np.array([[0, -T_dot/(T**2), 0],
                     [T_dot/(T**2), 0, 0],
                     [0, 0, 0]])


class BacksteppingController(BaseEnv):
    """ References:
    - Controller
    [1] G. P. Falconi and F. Holzapfel,
    “Adaptive Fault Tolerant Control Allocation for a Hexacopter System,”
    Proc. Am. Control Conf., vol. 2016-July, pp. 6760–6766, 2016.
    - Reference model (e.g., xd, vd, ad, ad_dot, ad_ddot)
    [2] S. J. Su, Y. Y. Zhu, H. R. Wang, and C. Yun, “A Method to Construct a Reference Model for Model Reference Adaptive Control,” Adv. Mech. Eng., vol. 11, no. 11, pp. 1–9, 2019.
    """
    def __init__(self, pos0, m, grav, **kwargs):
        super().__init__(**kwargs)
        self.xd = BaseSystem(pos0)
        self.vd = BaseSystem(np.zeros((3, 1)))
        self.ad = BaseSystem(np.zeros((3, 1)))
        self.ad_dot = BaseSystem(np.zeros((3, 1)))
        self.ad_ddot = BaseSystem(np.zeros((3, 1)))
        self.Td = BaseSystem(m*grav)
        # position
        self.Kx = m*1*np.eye(3)
        self.Kv = m*1*1.82*np.eye(3)
        self.Kp = np.hstack([self.Kx, self.Kv])
        self.Q = np.diag(1*np.ones(6))
        # thrust
        self.Kt = np.diag(4*np.ones(3))
        # angular
        self.Komega = np.diag(20*np.ones(3))
        # reference model
        self.Kxd = np.diag(1*np.ones(3))
        self.Kvd = np.diag(3.4*np.ones(3))
        self.Kad = np.diag(5.4*np.ones(3))
        self.Kad_dot = np.diag(4.9*np.ones(3))
        self.Kad_ddot = np.diag(2.7*np.ones(3))
        # others
        self.Ap = np.block([
            [np.zeros((3, 3)), np.eye(3)],
            [-(1/m)*self.Kx, -(1/m)*self.Kv],
        ])
        self.Bp = np.block([
            [np.zeros((3, 3))],
            [(1/m)*np.eye(3)],
        ])
        self.P = solve_lyapunov(self.Ap.T, -self.Q)

    def command(self):
        raise NotImplementedError("method `command` not defined")

    def step(self):
        t = self.clock.get()
        info = dict(t=t, **self.observe_dict())
        *_, done = self.update()
        next_obs = self.observe_list()
        return next_obs, np.zeros(1), info, done


class IndirectBacksteppingController(BacksteppingController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def dynamics(self, xd, vd, ad, ad_dot, ad_ddot, Td_dot, xc):
        d_xd = vd
        d_vd = ad
        d_ad = ad_dot
        d_ad_dot = ad_ddot
        d_ad_ddot = (-self.Kxd @ xd - self.Kvd @ vd - self.Kad @ ad
                     - self.Kad_dot @ ad_dot - self.Kad_ddot @ ad_ddot + self.Kxd @ xc)
        d_Td = Td_dot
        return d_xd, d_vd, d_ad, d_ad_dot, d_ad_ddot, d_Td

    def set_dot(self, Td_dot, xc):
        xd, vd, ad, ad_dot, ad_ddot, _ = self.observe_list()
        self.xd.dot, self.vd.dot, self.ad.dot, self.ad_dot.dot, self.ad_ddot.dot, self.Td.dot = self.dynamics(xd, vd, ad, ad_dot, ad_ddot, Td_dot, xc)

    def command(self, pos, vel, quat, omega,
                xd, vd, ad, ad_dot, ad_ddot, Td,
                m, J, g):
        rot = quat2dcm(quat)
        ex = xd - pos
        ev = vd - vel
        ep = np.vstack((ex, ev))
        Ap, Bp, P, Kp = self.Ap, self.Bp, self.P, self.Kp
        # u1
        u1 = m * (ad - g) + Kp @ ep
        zB = rot.T @ np.vstack((0, 0, 1))
        td = -Td * zB
        et = u1 - td
        ep_dot = Ap @ ep + Bp @ et
        u1_dot = m * ad_dot + Kp @ ep_dot
        T = Td  # TODO: no lag
        # u2
        u2 = T_u_inv(T[0]) @ rot @ (2*Bp.T @ P @ ep + u1_dot + self.Kt @ et)
        Td_dot = u2[-1]  # third element
        T_dot = Td_dot  # TODO: no lag
        zB_dot = -rot.T @ T_omega(1.0) @ omega
        et_dot = u1_dot + u2[-1] * zB + Td * zB_dot
        ep_ddot = Ap @ ep_dot + Bp @ et_dot
        u1_ddot = m * ad_ddot + Kp @ ep_ddot
        rot_dot = -skew(omega) @ rot
        u2_dot = (
            (T_u_inv_dot(T[0], T_dot[0]) @ rot
             + T_u_inv(T[0]) @ rot_dot) @ (2*Bp.T @ P @ ep + u1_dot + self.Kt@et)
            + (T_u_inv(T[0]) @ rot @ (2*Bp.T @ P @ ep_dot + u1_ddot + self.Kt @ et_dot))
        )
        omegad_dot = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 0, 0]]) @ u2_dot
        omegad = np.vstack((u2[:2], 0))
        eomega = omegad - omega
        Md = (
            np.cross(omega, J@omega, axis=0)
            + J @ (T_omega(T[0]).T @ rot @ et + omegad_dot + self.Komega @ eomega)
        )
        nud = np.vstack((Td, Md))
        return nud, Td_dot


class DirectBacksteppingController(BacksteppingController):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Theta_hat = BaseSystem(np.zeros((6, 4)))
        self.gamma = 1e-1
        # self.gamma = 1e-0  # for taeyoung_lee hexacopter model with hexa-x
        # self.gamma = 1e8  # for falconi hexacopter model

    def dynamics(self, xd, vd, ad, ad_dot, ad_ddot, Td_dot, Theta_hat_dot, xc):
        d_xd = vd
        d_vd = ad
        d_ad = ad_dot
        d_ad_dot = ad_ddot
        d_ad_ddot = (-self.Kxd @ xd - self.Kvd @ vd - self.Kad @ ad
                     - self.Kad_dot @ ad_dot - self.Kad_ddot @ ad_ddot + self.Kxd @ xc)
        d_Td = Td_dot
        d_Theta_hat = Theta_hat_dot
        return d_xd, d_vd, d_ad, d_ad_dot, d_ad_ddot, d_Td, d_Theta_hat

    def set_dot(self, Td_dot, Theta_hat_dot, xc):
        xd, vd, ad, ad_dot, ad_ddot, *_ = self.observe_list()
        self.xd.dot, self.vd.dot, self.ad.dot, self.ad_dot.dot, self.ad_ddot.dot, self.Td.dot, self.Theta_hat.dot = self.dynamics(xd, vd, ad, ad_dot, ad_ddot, Td_dot, Theta_hat_dot, xc)

    # def Proj(self, theta, y, epsilon=1e-2, theta_max=1e5):
    def Proj(self, theta, y, epsilon=1e-2, theta_max=1e5):
        f = ((1+epsilon) * np.linalg.norm(theta)**2 - theta_max**2) / (epsilon * theta_max**2)
        del_f = (2*(1+epsilon) / (epsilon * theta_max**2)) * theta  # nabla f
        if f > 0 and del_f.T @ y > 0:
            del_f_unit = del_f / np.linalg.norm(del_f)
            proj = y - np.dot(del_f_unit, y) * f * del_f_unit
        else:
            proj = y
        return proj

    def Proj_R(self, C, Y):
        proj_R = np.zeros(Y.shape)
        for i in range(Y.shape[0]):
            ci = C[i, :]
            yi = Y[i, :]
            proj_R[i, :] = self.Proj(ci, yi)
        return proj_R

    def command(self, pos, vel, quat, omega,
                xd, vd, ad, ad_dot, ad_ddot, Td, Theta_hat,
                m, J, g, B_A):
        rot = quat2dcm(quat)
        ex = xd - pos
        ev = vd - vel
        ep = np.vstack((ex, ev))
        Ap, Bp, P, Kp = self.Ap, self.Bp, self.P, self.Kp
        # u1
        u1 = m * (ad - g) + Kp @ ep
        zB = rot.T @ np.vstack((0, 0, 1))
        td = -Td * zB
        et = u1 - td
        n_ep_dot = Ap @ ep + Bp @ et
        n_u1_dot = m * ad_dot + Kp @ n_ep_dot
        T = Td  # TODO: no lag
        # u2
        u2 = T_u_inv(T[0]) @ rot @ (2*Bp.T @ P @ ep + n_u1_dot + self.Kt @ et)
        Td_dot = u2[-1]  # third element
        T_dot = Td_dot  # TODO: no lag
        zB_dot = -rot.T @ T_omega(1.0) @ omega
        n_et_dot = n_u1_dot + u2[-1] * zB + Td * zB_dot
        n_ep_ddot = Ap @ n_ep_dot + Bp @ n_et_dot
        n_u1_ddot = m * ad_ddot + Kp @ n_ep_ddot
        rot_dot = -skew(omega) @ rot
        n_u2_dot = (
            (T_u_inv_dot(T[0], T_dot[0]) @ rot + T_u_inv(T[0]) @ rot_dot)
            @ (2*Bp.T @ P @ ep + n_u1_dot + self.Kt@et)
            + (T_u_inv(T[0]) @ rot @ (2*Bp.T @ P @ n_ep_dot + n_u1_ddot + self.Kt @ n_et_dot))
        )
        n_omegad_dot = np.array([[1, 0, 0],
                                 [0, 1, 0],
                                 [0, 0, 0]]) @ n_u2_dot
        omegad = np.vstack((u2[:2], 0))
        eomega = omegad - omega
        # Md
        Md = (
            np.cross(omega, J@omega, axis=0)
            + J @ (T_omega(T[0]).T @ rot @ et + n_omegad_dot + self.Komega @ eomega)
        )
        # nud
        nud = np.vstack((Td, Md))
        # Theta_hat_dot
        e = np.vstack((ep, et, eomega))
        P_bar = np.block([
            [P, np.zeros((6, 6))],
            [np.zeros((6, 6)), 0.5*np.eye(6)]
        ])
        theta_ep_dot = Bp
        theta_u1_dot = Kp @ theta_ep_dot
        theta_et_dot = theta_u1_dot
        theta_ep_ddot = Ap @ theta_ep_dot + Bp @ theta_et_dot
        theta_u1_ddot = Kp @ theta_ep_ddot
        theta_u2_dot = T_u_inv(T[0]) @ rot @ (2*Bp.T @ P @ theta_ep_dot
                                              + theta_u1_ddot + self.Kt @ theta_et_dot)
        theta_omegad_dot = np.array([[1, 0, 0],
                                     [0, 1, 0],
                                     [0, 0, 0]]) @ theta_u2_dot
        B_bar = np.block([
            [-theta_ep_dot @ zB, np.zeros((6, 3))],
            [-theta_u1_dot @ zB, np.zeros((3, 3))],
            [-theta_omegad_dot @ zB, np.linalg.inv(J)]
        ])
        Theta_hat_dot = self.gamma * self.Proj_R(
            Theta_hat, (nud @ e.T @ P_bar @ B_bar @ B_A).T
        )
        return nud, Td_dot, Theta_hat_dot


if __name__ == "__main__":
    pass
