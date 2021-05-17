import numpy as np
from numpy import sin, cos
from math import sqrt

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2angle


def sgn(s):
    if s > 0:
        return 1
    elif s < 0:
        return -1
    else:
        return 0


def sat(s, BoundaryLayer):
    if s > BoundaryLayer:
        return 1
    elif s < -BoundaryLayer:
        return -1
    else:
        return s


def Herrera(s, delta):
    return s / (abs(s) + delta)


class SlidingModeController(BaseEnv):
    '''References:
    [1] Herrera, Marco, et al.
    "Sliding mode control: An approach to control a quadrotor."
    2015 Asia-Pacific Conference on Computer Aided System Engineering. IEEE, 2015.i
    [2] Runcharoon, Krittaya, and Varawan Srichatrapimuk.
    "Sliding mode control of quadrotor."
    2013 The International Conference on Technological Advances in Electrical, Electronics and Computer Engineering (TAEECE). IEEE, 2013.
    '''
    def __init__(self, env, pos, vel, quat, omega):
        super().__init__(dt=0.001, max_t=10)
        self.pos = BaseSystem(pos)
        self.vel = BaseSystem(vel)
        self.acc = BaseSystem(np.zeros((3, 1)))
        self.angle = BaseSystem(np.vstack(quat2angle(quat)[::-1]))
        self.omega = BaseSystem(omega)
        self.omega_dot = BaseSystem(np.zeros((3, 1)))

        self.env = env

        self.d, self.m, self.g, self.J = env.plant.d, env.plant.m, env.plant.g, env.plant.J
        self.n_rotors = env.plant.mixer.B.shape[1]
        trim_rotors = np.vstack([self.m * self.g / self.n_rotors] * self.n_rotors)
        self.u = env.plant.mixer.inverse(trim_rotors)

    def dynamics(self, pos, vel, acc, angle, omega, omega_dot):
        F, M1, M2, M3 = self.u
        phi, theta, psi = angle
        p, q, r = omega
        Ix = self.J[0, 0]
        Iy = self.J[1, 1]
        Iz = self.J[2, 2]
        d = self.d

        dpos = vel
        dvel = acc
        dacc = np.vstack(((cos(phi)*sin(theta)*cos(-psi) + sin(phi)*sin(-psi))/self.m*F,
                          (cos(phi)*sin(theta)*cos(-psi) - sin(phi)*sin(-psi))/self.m*F,
                          +self.g - (cos(phi)*cos(theta))/self.m*F))
        dangle = omega
        domega = omega_dot
        domega_dot = np.vstack((-p*r*(Iy-Iz)/Ix + d/Ix*M1,
                                -p*r*(Iz-Ix)/Iy + d/Iy*M2,
                                -p*q*(Ix-Iy)/Iz - 1/Iz*M3))
        return dpos, dvel, dacc, dangle, domega, domega_dot

    def set_dot(self):
        pos, vel, acc, angle, omega, omega_dot = self.observe_list()
        self.pos.dot, self.vel.dot, self.acc.dot, self.angle.dot, self.omega.dot, self.omega_dot.dot = self.dynamics(pos, vel, acc, angle. omega, omega_dot, self.u)

    def get_action(self, obs, gammaTune, kdTune, dtype="sat"):
        # Tuning parameters
        gt_F, gt_M1, gt_M2, gt_M3 = gammaTune
        kt_F, kt_M1, kt_M2, kt_M3 = kdTune
        # model
        m, g = self.m, self.g
        Ix = self.J[0, 0]
        Iy = self.J[1, 1]
        Iz = self.J[2, 2]
        d = self.d
        # reference
        zd, wd, dot_wd = self.pos.state[2], self.vel.state[2], self.acc.state[2]
        phid, thetad, psid = self.angle.state
        pd, qd, rd = self.omega.state
        pdd, qdd, rdd = self.omega_dot.state
        # observation
        obs = np.vstack((obs))
        obs = np.vstack((obs[0:6], np.vstack(quat2angle(obs[6:10])[::-1]), obs[10::]))
        z, w = obs[2], obs[5]
        phi, theta, psi = obs[6:9]
        p, q, r = obs[9:]

        # continuous part
        Feq = (g + gt_F*(wd-w) + dot_wd) * m / (cos(phi)*cos(psi))
        M1eq = (gt_M1*(pd-p) + pdd - q*r*(Iy-Iz)/Ix)*Ix/d
        M2eq = (gt_M2*(qd-q) + qdd - p*r*(Iz-Ix)/Iy)*Iy/d
        M3eq = (gt_M3*(rd-r) + rdd - p*q*(Ix-Iy)/Iz)*Iz

        # discrete part
        sF = wd - w + gt_F*(zd - z)
        sM1 = pd - p + gt_M1*(phid - phi)
        sM2 = qd - q + gt_M2*(thetad - theta)
        sM3 = rd - r + gt_M3*(psid - psi)

        if dtype == "sgn":
            Fd, M1d, M2d, M3d = sgn(sF), sgn(sM1), sgn(sM2), sgn(sM3)
        elif dtype == "sat":
            BoundaryLayer = 1
            Fd, M1d, M2d, M3d = sat(sF, BoundaryLayer), sat(sM1, BoundaryLayer), sat(sM2, BoundaryLayer), sat(sM3, BoundaryLayer)
        elif dtype == "Herrera":
            delta = 0.3
            Fd, M1d, M2d, M3d = Herrera(sF, delta), Herrera(sM1, delta), Herrera(sM2, delta), Herrera(sM3, delta)

        F = Feq + kt_F*Fd
        M1 = M1eq + kt_M1*M1d
        M2 = M2eq + kt_M2*M2d
        M3 = M3eq + kt_M3*M3d

        action = np.vstack((F, M1, M2, M3))
        # self.u = action

        return action


if __name__ == "__main__":
    pass
