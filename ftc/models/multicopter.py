import numpy as np

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import dcm2quat, quat2dcm, angle2quat, quat2angle


class multicopter(BaseEnv):
    """
    Prof. Taeyoung Lee's model for quadrotor UAV is used.
    - (https://www.math.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf)
    """
    J = np.diag([0.0820, 0.0845, 0.1377])  # kg * m^2
    m = 4.34  # kg
    d = 0.0315  # m
    c = 8.004e-4  # m
    g = 9.81  # m/s^2
    state_lower_bound = np.array(-np.inf * np.ones(13))
    state_upper_bound = np.array(np.inf * np.ones(13))
    control_lower_bound = np.array(-np.inf * np.ones(4))
    control_upper_bound = np.array(np.inf * np.ones(4))

    def __init__(self, pos, vel, quat, omega, rtype, **kwargs):
        super().__init__(**kwargs)
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)
        self.quat = BaseSystem(quat)
        self.omega = BaseSystem(omega)
        self.rtype = rtype

    def deriv(self, pos, vel, quat, omega, control, fault, ctype="rotor"):
        mixer = self.mixing(self.rtype)
        if ctype == "force":
            # control force to rotor
            rotor = np.linalg.pinv(mixer).dot(control)
        elif ctype == "rotor":
            rotor = control

        # fault
        LoE = fault

        ctrl = LoE.dot(rotor)
        F, M1, M2, M3 = mixer.dot(ctrl)

        M = np.vstack((M1, M2, M3))

        m, g, J = self.m, self.g, self.J
        e3 = np.vstack((0, 0, 1))

        dpos = vel
        dcm = quat2dcm(quat)
        dvel = g*e3 - F*dcm.dot(e3)/m
        dquat = 0.5 * np.vstack((
            -omega.T.dot(quat[1:]),
            omega*quat[0] - np.cross(omega, quat[1:], axis=0)
        ))
        domeg = np.linalg.inv(J).dot(M - np.cross(omega, J.dot(omega), axis=0))

        return dpos, dvel, dquat, domeg

    def set_dot(self, t, control, fault, ctype="rotor"):
        pos, vel, quat, omega = self.observe_list()
        dots = self.deriv(pos, vel, quat, omega, control, fault, ctype)
        self.pos.dot, self.vel.dot, self.quat.dot, self.omega.dot = dots

    def hat(self, v):
        v1, v2, v3 = v.squeeze()
        return np.array([
            [0, -v3, v2],
            [v3, 0, -v1],
            [-v2, v1, 0]
        ])

    def mixing(self, rtype):
        d, c = self.d, self.c
        b = 1
        if rtype == "quad":
            mixer = np.array(
                [[1, 1, 1, 1],
                 [0, -d, 0, d],
                 [d, 0, -d, 0],
                 [-c, c, -c, c]]
            )
        elif rtype == "hexa-x":
            mixer = np.array(
                [[b, b, b, b, b, b],
                 [-b*d, b*d, b*d/2, -b*d/2, -b*d/2, b*d/2],
                 [0, 0, b*d*np.sqrt(3)/2, -b*d*np.sqrt(3)/2, b*d*np.sqrt(3)/2,
                  -b*d*np.sqrt(3)/2],
                 [c, -c, c, -c, -c, c]]
            )
        elif rtype == "hexa-+":
            mixer = np.array(
                [[b, b, b, b, b, b],
                 [0, 0, b*d*np.sqrt(3)/2, -b*d*np.sqrt(3)/2, b*d*np.sqrt(3)/2,
                  -b*d*np.sqrt(3)/2],
                 [-b*d, b*d, b*d/2, -b*d/2, -b*d/2, b*d/2],
                 [c, -c, c, -c, -c, c]]
            )
        else:
            mixer = np.eye(4)
        return mixer

if __name__ == "__main__":
    x = np.zeros((3, 1))
    v = np.zeros((3, 1))
    q = np.vstack((1, 0, 0, 0))
    omega = np.zeros((3, 1))
    rtype = "hexa-x"
    fault = np.eye(6)

    system = multicopter(x, v, q, omega, rtype)
    system.set_dot(0, np.zeros((6, 1)), fault)
