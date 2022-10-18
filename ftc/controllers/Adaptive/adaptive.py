import fym
import numpy as np
from fym.utils.rot import quat2angle
from numpy import cos, sin, tan
from sklearn.gaussian_process.kernels import RBF


def cross(x, y):
    return np.cross(x, y, axis=0)


class Adaptive(fym.BaseEnv):
    CONTROLLER_CONFIG = {
        "outer_surface_factor": np.vstack((1.2, 1.2, 1.2)),
        "outer_proportional": np.vstack((0.2, 0.2, 5)),
        "outer_adaptive_gain": np.vstack((0.001, 0.001, 0.3)),
        "outer_adaptive_decay": np.vstack((0.1, 0.1, 0.1)),
        "inner_surface_factor": 20,
        "inner_proportional": 5,
        "inner_adaptive_gain": 0.001,
        "inner_adaptive_decay": 0.1,
        "use_Nussbaum": True,
    }

    def __init__(self, env):
        super().__init__()

        cfg = self.CONTROLLER_CONFIG

        """ Fym Systems """

        """ Aux """
        self.W1hat = fym.BaseSystem(shape=(3, 1))
        self.W2hat = fym.BaseSystem()
        if cfg["use_Nussbaum"]:
            self.mu = fym.BaseSystem(shape=(6, 1))

        """ Basis function """

        self.kernel = RBF(10.0, "fixed")
        self.centers = np.zeros((50, 1))

        """ LC62 """

        # c = 5.38
        c = 0.3156
        dy1, dy2 = env.plant.dy1, env.plant.dy2
        dx1, dx2, dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        self.B = np.array(
            (
                [1, 1, 1, 1, 1, 1],
                [-dy2, dy1, dy1, -dy2, -dy2, dy1],
                [-dx2, -dx2, dx1, -dx3, dx1, -dx3],
                [-c, c, -c, c, c, -c],
            )
        )
        self.Binv = np.linalg.pinv(self.B)

    def get_control(self, t, env):
        """Get control input"""

        cfg = self.CONTROLLER_CONFIG

        """ Model uncertainties """

        m = env.plant.m
        g = env.plant.g
        J = env.plant.J
        Jinv = env.plant.Jinv
        B = self.B
        Binv = self.Binv

        """ Outer-loop control """

        pos = env.plant.pos.state
        vel = env.plant.vel.state

        posd = env.scenario.posd(t)
        posd_dot = env.scenario.posd_dot(t)
        posd_ddot = env.scenario.posd_ddot(t)

        e1 = pos - posd
        e1_dot = vel - posd_dot
        z1 = e1_dot + cfg["outer_surface_factor"] * e1

        # adaptive
        Phi1 = self.get_Psi(pos, vel)
        varphi1 = (1 + np.linalg.norm(Phi1)) ** 2 * np.ones((3, 1))

        W1hat = self.W1hat.state
        self.W1hat.dot = (
            cfg["outer_adaptive_gain"] * varphi1 * z1**2
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

        """ Inner-loop control """

        angles = self.get_angles(env.plant.quat.state)
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

        Phi2 = self.get_Psi(xi, xi_dot)
        varphi2 = (1 + np.linalg.norm(Phi2)) ** 2
        W2hat = self.W2hat.state
        self.W2hat.dot = (
            cfg["inner_adaptive_gain"] * varphi2 * z2.T @ z2
            - cfg["inner_adaptive_decay"] * W2hat
        )

        # control input
        v = J @ (
            Jinv @ cross(omega, J @ omega)
            + np.linalg.inv(H)
            @ (
                -H_dot @ omega
                + xid_ddot
                - cfg["inner_surface_factor"] * e2_dot
                - cfg["inner_proportional"] * z2
            )
            - cfg["inner_adaptive_gain"] * W2hat * varphi2 * z2
        )

        uc = Binv @ np.vstack((uf, v))

        G = np.block(
            [
                [-cos(phi) * cos(theta) / m, np.zeros((1, 3))],
                [np.zeros((3, 1)), H @ Jinv],
            ]
        )

        if cfg["use_Nussbaum"]:
            mu = self.mu.state
            N = self.Nussbaum(mu)

            b = 1
            self.mu.dot = -b * 1 * (B.T @ G.T @ np.vstack((z1[2:], z2))) * uc
            uc = N * uc

        """ set derivatives """

        # Saturated PWM with LoE
        uc = np.clip(uc, 0, 160)
        Lambda = env.scenario.get_Lambda(t)
        pwm = Lambda * uc / 160 * 1000 + 1000

        control_input = np.vstack((pwm, *env.plant.u_trims_fixed))
        FM = env.plant.B_VTOL(control_input, env.plant.omega.state)
        f = -FM[2]
        tau = FM[3:]

        controller_info = {
            "pos": pos,
            "posd": posd,
            "vel": vel,
            "veld": posd_dot,
            "angles": xi,
            "anglesd": xid,
            "omega": xi_dot + cfg["inner_surface_factor"] * xi,
            "omegad": xid_dot + cfg["inner_surface_factor"] * xid,
            "uc": pwm,
            "u": control_input,
            "f": f,
            "tau": tau,
            "Lambda": Lambda,
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
        return np.exp(mu**2 / 2) * (mu**2 + 2) * sin(mu) + 1

    def get_angles(self, quat):
        return np.asarray(quat2angle(quat)[::-1])
