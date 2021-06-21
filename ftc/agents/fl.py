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

        self.kp1 = 0.3*np.diag([2, 5, 8])
        self.kp2 = 0.7*np.diag([2, 5, 8])
        self.kp3 = 0.7*np.diag([2, 5, 8])
        self.kp4 = 0.3*np.diag([2, 5, 8])

    def get_virtual(self, t, plant, ref):
        m, g, J = self.m, self.g, self.J
        pos = plant.pos.state
        vel = plant.vel.state

        posd = ref[0:3]
        veld = np.zeros((3, 1))
        dveld = np.zeros((3, 1))
        ddveld = np.zeros((3, 1))
        psid = quat2angle(ref[6:10])[::-1][2]

        phi, theta, psi = quat2angle(plant.quat.state)[::-1]
        dphi, dtheta, dpsi = self.dangle.state.ravel()
        u1 = self.u1.state[0]
        du1 = self.du1.state[0]

        dvel = np.array([
            - u1*sin(theta) / m,
            u1*sin(phi) / m,
            - u1*cos(phi)*cos(theta) / m + g
        ])[:, None]

        ddvel = np.array([
            (- du1*sin(theta) - u1*dtheta*cos(theta)) / m,
            (- du1*sin(phi) + u1*dphi*cos(phi)) / m,
            ((- du1*cos(theta)*cos(phi) + u1*dtheta*sin(theta)*cos(phi)
              + u1*dphi*cos(theta)*cos(phi))) / m
        ])[:, None]

        v = (- self.kp1.dot(pos-posd)
             - self.kp2.dot(vel-veld)
             - self.kp3.dot(dvel-dveld)
             - self.kp4.dot(ddvel-ddveld))

        G = np.array([
            [-sin(theta)/m, -u1*cos(theta)/m, 0.0],
            [sin(phi)/m, 0.0, -u1*cos(phi)/m],
            [-cos(theta)*cos(phi)/m, u1*sin(theta)*cos(phi)/m,
             u1*cos(theta)*sin(phi)/m]])
        F = np.array([
            2*du1*dtheta*cos(theta)/m - u1*dtheta**2*sin(theta)/m,
            - 2*du1*dphi*cos(phi)/m + u1*dphi**2*sin(phi)/m,
            - 2*du1*dtheta*sin(theta)*cos(phi)/m
            - 2*du1*dphi*cos(theta)*sin(phi)/m
            + 2*u1*dtheta*dphi*sin(theta)*sin(phi)/m
            - u1*(dtheta**2-dphi**2)*cos(theta)*cos(phi)/m
        ])[:, None] + v
        f = np.linalg.inv(G).dot(F)
        d2u1, u2, u3 = f.ravel()

        kh1 = 1
        kh2 = 1
        u4 = - kh1*(dpsi-0) - kh2*(psi-psid)
        return d2u1, np.array([u2/J[0, 0], u3/J[1, 1], u4/J[2, 2]])[:, None]

    def get_FM(self, ctrl):
        return np.vstack((self.u1.state, ctrl[1]))

    def set_dot(self, ctrl):
        self.angle.dot = self.dangle.state
        self.dangle.dot = self.Jinv.dot(ctrl[1])

        self.u1.dot = self.du1.state
        self.du1.dot = ctrl[0]


if __name__ == "__main__":
    pass
