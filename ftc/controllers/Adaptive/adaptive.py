import fym
import numpy as np
from numpy import cos, sin, tan
from sklearn.gaussian_process.kernels import RBF

from ftc.utils import safeupdate


def cross(x, y):
    return np.cross(x, y, axis=0)


class Adaptive(fym.BaseEnv):
    CONTROLLER_CONFIG = {
        "lemniscate": {
            "outer_surface_factor": 7,
            "outer_proportional": 5,
            "outer_adaptive_gain": 0.3,
            "outer_adaptive_decay": 0.1,
            "inner_surface_factor": 20,
            "inner_proportional": 5,
            "inner_adaptive_gain": 0.3,
            "inner_adaptive_decay": 0.1,
            "aux_decay": 0.5,
            "use_Nussbaum": False,
        },
    }

    TUNE_CONFIG = {
        "outer_surface_factor": 3,
        "outer_proportional": 1.5,
        "outer_adaptive_gain": 1.35,
        "outer_adaptive_decay": 0.0012,
        "inner_surface_factor": 3,
        "inner_proportional": 3,
        "inner_adaptive_gain": 2,
        "inner_adaptive_decay": 0.01,
    }

    def __init__(self, controller_config={}, scenario="lemniscate"):
        super().__init__()

        self.controller_config = safeupdate(
            self.CONTROLLER_CONFIG[scenario], controller_config
        )

        """ Fym Systems """

        """ Aux """
        self.W1hat = fym.BaseSystem()
        self.W2hat = fym.BaseSystem()
        if self.controller_config["use_Nussbaum"]:
            self.mu = fym.BaseSystem(shape=(4, 1))

        """ Basis function """

        self.kernel = RBF(3.0, "fixed")
        self.centers = np.zeros((50, 1))

    def get_control(self, t, env):
        """Get control input"""

        cfg = self.controller_config

        """ Uncertain model """
        m = env.plant.m * 1.2
        g = env.plant.g
        J = env.plant.J * np.vstack((1.2, 1.1, 1.3))
        Jinv = 1 / J
        B = env.plant.B
        Binv = env.plant.Binv

        """ Outer-Loop Control """

        pos = env.plant.pos.state
        vel = env.plant.vel.state

        posd = env.scenario.posd(t)
        posd_dot = env.scenario.posd_dot(t)
        posd_ddot = env.scenario.posd_ddot(t)

        e1 = pos - posd
        e1_dot = vel - posd_dot
        z1 = e1_dot + cfg["outer_surface_factor"] * e1

        # adaptive
        Phi1 = self.get_Psi(
            pos,
            posd,
            vel,
            posd_dot,
            # posd_ddot,
        )
        varphi1 = Phi1.T @ Phi1

        W1hat = self.W1hat.state
        self.W1hat.dot = (
            cfg["outer_adaptive_gain"] * varphi1 * z1.T @ z1
            - cfg["outer_adaptive_decay"] * W1hat
        )

        # virtual control input
        u1 = m * (
            posd_ddot
            - cfg["outer_surface_factor"] * e1_dot
            - g * np.vstack((0, 0, 1))
            - cfg["outer_proportional"] * z1
            - cfg["outer_adaptive_gain"] * W1hat * varphi1 * z1
        )

        # transform
        uf = np.linalg.norm(u1)
        Qx, Qy, Qz = u1.ravel()
        psid = env.scenario.psid(t)

        phid = np.arcsin((-Qx * sin(psid) + Qy * cos(psid)) / uf)
        thetad = np.arctan((Qx * cos(psid) + Qy * sin(psid)) / Qz)

        """ Inner-Loop Control """
        angles = env.plant.get_angles()
        phi, theta, _ = angles
        xi = angles[:, None]
        omega = env.plant.omega.state
        H = np.array(
            [
                [1, sin(phi) * tan(theta), cos(phi) * tan(theta)],
                [0, cos(phi), -sin(phi)],
                [0, sin(phi) / cos(theta), cos(phi) / cos(theta)],
            ]
        )
        xi_dot = H @ omega

        phi_dot, theta_dot, _ = xi_dot.ravel()
        H_dot = np.array(
            [
                [
                    0,
                    cos(phi) * tan(theta) * phi_dot
                    + sin(phi) / cos(theta) ** 2 * theta_dot,
                    -sin(phi) * tan(theta) * phi_dot
                    + cos(phi) / cos(theta) ** 2 * theta_dot,
                ],
                [0, -sin(phi) * phi_dot, -cos(phi) * phi_dot],
                [
                    0,
                    cos(phi) / cos(theta) * phi_dot
                    + sin(phi) * sin(theta) / cos(theta) ** 2 * theta_dot,
                    -sin(phi) / cos(theta) * phi_dot
                    + cos(phi) * sin(theta) / cos(theta) ** 2 * theta_dot,
                ],
            ]
        )

        xid = np.vstack((phid, thetad, psid))

        # xi differentiator
        xid_dot = xid_ddot = np.zeros((3, 1))

        e2 = xi - xid
        e2_dot = xi_dot - xid_dot
        z2 = e2_dot + cfg["inner_surface_factor"] * e2

        Phi2 = self.get_Psi(
            xi,
            xid,
            xi_dot,
            xid_dot,
            # xid_ddot,
        )
        varphi2 = Phi2.T @ Phi2
        W2hat = self.W2hat.state
        self.W2hat.dot = (
            cfg["inner_adaptive_gain"] * varphi2 * z2.T @ z2
            - cfg["inner_adaptive_decay"] * W2hat
        )

        # control input
        v = J * (
            (1 / J) * cross(omega, J * omega)
            + np.linalg.inv(H)
            @ (
                -cfg["inner_adaptive_gain"] * W2hat * varphi2 * z2
                - H_dot @ omega
                + xid_ddot
                - cfg["inner_surface_factor"] * e2_dot
                - cfg["inner_proportional"] * z2
            )
        )

        # vmin = -np.vstack((0.8, 0.8, 0.05))
        # vmax = np.vstack((0.8, 0.8, 0.05))
        # hv = np.clip(v, vmin, vmax)
        # hv = v
        uc = Binv @ np.vstack((uf, v))

        G = np.block(
            [
                [-cos(phi) * cos(theta) / m, np.zeros((1, 3))],
                [np.zeros((3, 1)), H @ np.diag(Jinv.ravel())],
            ]
        )

        if cfg["use_Nussbaum"]:
            mu = self.mu.state
            N = self.Nussbaum(mu)

            b = 0.001
            self.mu.dot = -b * 1 * (B.T @ G.T @ np.vstack((z1[2:], z2))) * uc
            uc = N * uc

        """ set derivatives """

        # omegad = np.linalg.inv(H) @ xid_dot

        eta43l = np.nan * np.ones((3, 1))
        eta43u = np.nan * np.ones((3, 1))

        control_input = uc

        controller_info = {
            "pos": pos,
            "posd": posd,
            "epos_trans": np.nan * np.ones((3, 1)),
            "epos_upper_bound": np.nan * np.ones((3, 1)),
            "epos_lower_bound": np.nan * np.ones((3, 1)),
            "vel": vel,
            "veld": posd_dot,
            "angles": xi,
            "anglesd": xid,
            "omega": xi_dot + cfg["inner_surface_factor"] * xi,
            "omegad": xid_dot + cfg["inner_surface_factor"] * xid,
            "eomega_trans": np.nan * np.ones((3, 1)),
            "eomega_upper_bound": np.nan * np.ones((3, 1)),
            "eomega_lower_bound": np.nan * np.ones((3, 1)),
            "uc": uc,
            "eta43l": eta43l,
            "eta43u": eta43u,
            "v": v,
            "hv": v,
        }

        if env.brk and t >= env.clock.max_t:
            breakpoint()

        return control_input, controller_info

    def get_Psi(self, *args):
        x = np.hstack([np.ravel(a) for a in args])[None]
        if np.shape(x)[1] != self.centers.shape[1] and self.centers.shape[1] == 1:
            centers = np.tile(self.centers, x.shape[1])
        else:
            centers = self.centers
        Phi = self.kernel(x, centers).T
        return Phi

    def Nussbaum(self, mu):
        # return mu**2 * cos(mu)
        # return np.exp(mu**2) * cos(np.pi * mu / 2)
        return np.exp(mu**2 / 2) * (mu**2 + 2) * sin(mu) + 1

    def g(self, tau, low, high, with_grad=False):
        return np.clip(tau, low, high)
        # c = 0.5 * (high + low)
        # s = 0.5 * (high - low)
        # h = s * np.tanh((tau - c) / s) + c
        # if not with_grad:
        #     return h
        # else:
        #     h_grad = 1 - np.tanh((tau - c) / s) ** 2
        #     return h, h_grad
