""" References
[1] F. L. Lewis, A. Das, and K. Subbarao, “Dynamic inversion with zero-dynamics stabilisation for quadrotor control,” IET Control Theory Appl., vol. 3, no. 3, pp. 303–314, Mar. 2009, doi: 10.1049/iet-cta:20080002.
"""

import fym
import numpy as np
from fym.utils.rot import quat2angle, quat2dcm
from numpy import cos, sin, tan


class NDIController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.dx1, self.dx2, self.dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        self.dy1, self.dy2 = env.plant.dy1, env.plant.dy2
        c, self.c_th = 0.0338, 128  # tq / th, th / rcmds
        self.B_r2f = np.array(
            (
                [-1, -1, -1, -1, -1, -1],
                [-self.dy2, self.dy1, self.dy1, -self.dy2, -self.dy2, self.dy1],
                [-self.dx2, -self.dx2, self.dx1, -self.dx3, self.dx1, -self.dx3],
                [-c, c, -c, c, c, -c],
            )
        )

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])

        posd, posd_dot = env.get_ref(t, "posd", "posd_dot")

        """ outer-loop control
        Objective: horizontal position (x, y) tracking control
        States:
            pos[0:2]: horizontal position
            posd[0:2]: desired horizontal position
        """
        xo, xod = pos[0:2], posd[0:2]
        xo_dot, xod_dot = vel[0:2], posd_dot[0:2]
        eo, eo_dot = xo - xod, xo_dot - xod_dot
        Ko1 = 0.5 * np.diag((3, 1))
        Ko2 = 0.5 * np.diag((3, 2))

        # outer-loop virtual control input
        nuo = (-Ko1 @ eo - Ko2 @ eo_dot) / env.plant.g
        angd = np.vstack((nuo[1], -nuo[0], 0))
        # angd = np.deg2rad(np.vstack((0, 0, 0)))  # to regulate Euler angle

        """ inner-loop control
        Objective: vertical position (z) and angle (phi, theta, psi) tracking control
        States:
            pos[2]: vertical position
            posd[2]: desired vertical position
            ang: Euler angle
            angd: desired Euler angle
        """
        xi = np.vstack((pos[2], ang))
        xid = np.vstack((posd[2], angd))
        xi_dot = np.vstack((vel[2], omega))
        xid_dot = np.vstack((posd_dot[2], 0, 0, 0))
        ei = xi - xid
        ei_dot = xi_dot - xid_dot
        Ki1 = 5 * np.diag((5, 10, 50, 10))
        Ki2 = 1 * np.diag((5, 10, 50, 10))
        f = np.vstack(
            (
                env.plant.g,
                -env.plant.Jinv @ np.cross(omega, env.plant.J @ omega, axis=0),
            )
        )
        g = np.zeros((4, 4))
        g[0, 0] = quat2dcm(quat).T[2, 2] / env.plant.m
        g[1:4, 1:4] = env.plant.Jinv

        # inner-loop virtual control input
        nui = np.linalg.inv(g) @ (-f - Ki1 @ ei - Ki2 @ ei_dot)

        # control input
        th = np.linalg.pinv(self.B_r2f) @ nui
        # transform to pwms
        pwms_rotor = env.plant.cmd2pwm(th / self.c_th)
        ctrls = np.vstack((pwms_rotor, np.vstack(env.plant.u_trims_fixed)))

        controller_info = {
            "posd": posd,
            "angd": angd,
            "ang": ang,
        }

        return ctrls, controller_info
