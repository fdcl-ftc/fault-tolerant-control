import numpy as np
from numpy import sin, cos

from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2angle


class FLController(BaseEnv):
    def __init__(self):
        super().__init__()

    def dynamics(self, angle, dangle, u1, du1, ctrl):
        d2u1, u2, u3, u4 = ctrl

        angle_dot = dangle
        angle_2dot = np.vstack((u2, u3, u4))

        u1_dot = du1
        u1_2dot = d2u1

        return angle_dot, angle_2dot, u1_dot, u1_2dot

    def get_virtual(self, t, mult_states, ref):
        pos, vel, quat, omega = mult_states
        # angle = np.vstack(quat2angle(quat)[::-1])

        posd = ref

        angle, dangle, u1, du1 = self.observe_list()
        phi, theta, psi = angle
        dphi, dtheta, dpsi = dangle

        kp1 = 10*np.diag([1, 1, 1])
        kp2 = 50*np.diag([1, 1, 1])
        v = - kp1*(pos-posd) - kp2*(vel-np.vstack((0, 0, 0)))

        G = np.array(([-sin(theta), -u1*cos(theta), 0],
                      [sin(phi), 0, -u1*cos(phi)],
                      [-cos(theta)*cos(phi), u1*sin(theta)*cos(phi),
                       u1*cos(theta)*sin(phi)]
                      ))
        G_inv = np.linalg.inv(G)
        F = np.vstack((2*du1*dtheta*cos(theta) - u1*dtheta**2*sin(theta) + v[0],
                       - 2*du1*dphi*cos(phi) + u1*dphi**2*sin(phi) + v[1],
                       - 2*du1*dtheta*sin(theta)*cos(phi)
                       - 2*du1*dphi*cos(theta)*sin(phi)
                       + 2*u1*dtheta*dphi*sin(theta)*sin(phi)
                       - u1*(dtheta**2-dphi**2)*cos(theta)*cos(phi) + v[2]))

        f = G_inv*F
        d2u1, u2, u3 = f

        kh1 = 1
        kh2 = 1
        u4 = - kh1*(dpsi-0) - kh2*(psi-0)
        return np.vstack((d2u1, u2, u3, u4))

    # def get_FM(self, ctrl_state, ctrl):
        # _, _, u1, _ = ctrl_state
    def get_FM(self, ctrl):
        _, _, u1, _ = self.observe_list()
        _, u2, u3, u4 = ctrl
        return np.vstack((u1, u2, u3, u4))

    def set_dot(self, ctrl):
        angle, dangle, u1, du1 = self.observe_list()
        self.angle.dot, self.dangle.dot, self.u1.dot, self.du1.dot = self.dynamics(angle, dangle, u1, du1, ctrl)


if __name__ == "__main__":
    pass
