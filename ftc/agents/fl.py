import numpy as np
from numpy import sin, cos

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2angle


class FLController(BaseEnv):
    def __init__(self, m, g, J):
        super().__init__()
        self.angle = BaseSystem(np.zeros((3, 1)))
        self.dangle = BaseSystem(np.zeros((3, 1)))
        self.u1 = BaseSystem(np.array((m*g)))
        self.du1 = BaseSystem(np.zeros((1)))
        self.m, self.g, self.J = m, g, J
        self.Jinv = np.linalg.inv(J)
        self.trim_forces = np.vstack([self.m * self.g, 0, 0, 0])

    def dynamics(self, angle, dangle, u1, du1, ctrl):
        Jinv = self.Jinv
        d2u1, u2, u3, u4 = ctrl

        angle_dot = dangle
        angle_2dot = Jinv.dot(np.vstack((u2, u3, u4)))

        u1_dot = du1
        u1_2dot = d2u1

        return angle_dot, angle_2dot, u1_dot, u1_2dot

    def get_virtual(self, t, mult_states, ref):
        m, g, J = self.m, self.g, self.J
        pos, vel, quat, omega = mult_states
        angle = np.vstack(quat2angle(quat)[::-1])

        posd = ref[0:3]
        angled = np.vstack(quat2angle(ref[6:10])[::-1])
        psid = angled[2]

        _, dangle, _u1, _du1 = self.observe_list()
        phi, theta, psi = angle.ravel()
        dphi, dtheta, dpsi = dangle.ravel()
        u1 = _u1[0]
        du1 = _du1[0]

        d2x = - u1*sin(theta) / m
        d2y = u1*sin(phi) / m
        d2z = - u1*cos(phi)*cos(theta) / m + g

        d3x = (- du1*sin(theta) - u1*dtheta*cos(theta)) / m
        d3y = (- du1*sin(phi) + u1*dphi*cos(phi)) / m
        d3z = ((- du1*cos(theta)*cos(phi) + u1*dtheta*sin(theta)*cos(phi)
               + u1*dphi*cos(theta)*cos(phi))) / m

        # no-failure
        kp1 = 1*np.diag([1, 2, 5])
        kp2 = 2.5*np.diag([1, 2, 5])
        kp3 = 2.5*np.diag([1, 2, 5])
        kp4 = 1*np.diag([1, 2, 5])
        v = (- kp1.dot(pos-posd)
             - kp2.dot(vel-np.vstack((0, 0, 0)))
             - kp3.dot(np.vstack((d2x, d2y, d2z))-np.vstack((0, 0, 0)))
             - kp4.dot(np.vstack((d3x, d3y, d3z))-np.vstack((0, 0, 0))))

        G = np.array([[-sin(theta)/m, -u1*cos(theta)/m, 0.0],
                      [sin(phi)/m, 0.0, -u1*cos(phi)/m],
                      [-cos(theta)*cos(phi)/m, u1*sin(theta)*cos(phi)/m,
                       u1*cos(theta)*sin(phi)/m]])
        F = np.vstack((2*du1*dtheta*cos(theta)/m - u1*dtheta**2*sin(theta)/m
                       + v[0],
                       - 2*du1*dphi*cos(phi)/m + u1*dphi**2*sin(phi)/m + v[1],
                       - 2*du1*dtheta*sin(theta)*cos(phi)/m
                       - 2*du1*dphi*cos(theta)*sin(phi)/m
                       + 2*u1*dtheta*dphi*sin(theta)*sin(phi)/m
                       - u1*(dtheta**2-dphi**2)*cos(theta)*cos(phi)/m + v[2]))
        f = np.linalg.inv(G).dot(F)
        d2u1, u2, u3 = f

        kh1 = 1
        kh2 = 1
        u4 = - kh1*(dpsi-0) - kh2*(psi-psid)
        return np.vstack((d2u1, u2/J[0, 0], u3/J[1, 1], u4/J[2, 2]))

    def get_FM(self, ctrl):
        _, _, u1, _ = self.observe_list()
        _, u2, u3, u4 = ctrl
        return np.vstack((u1, u2, u3, u4))

    def set_dot(self, ctrl):
        angle, dangle, u1, du1 = self.observe_list()
        self.angle.dot, self.dangle.dot, self.u1.dot, self.du1.dot = self.dynamics(angle, dangle, u1, du1, ctrl)


if __name__ == "__main__":
    pass
