from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2angle

import numpy as np


def func_g(x, theta):
    delta = 1
    if abs(x) < delta:
        return x / delta**(1-theta)
    else:
        return np.sign(x) * abs(x)**theta


class BLFController(BaseEnv):
    def __init__(self):
        # controllers
        self.Cx = outerLoop()
        self.Cy = outerLoop()
        self.Cz = outerLoop()
        self.Cphi = innerLoop()
        self.Ctheta = innerLoop()
        self.Cpsi = innerLoop()

    def get_control(self, t, env):
        ''' quad state '''
        pos, vel, quat, omega = env.plant.observe_list()
        euler = quat2angle(quat)[::-1]

        ''' plant parameters '''
        # m = env.plant.m
        # g = env.plant.g
        J = env.plant.J
        b = np.array([1/J[0], 1/J[1], 1/J[2]])

        ''' external signals '''
        posd = env.get_ref(t, "posd")

        ''' outer loop control '''
        q = np.zeros((3, 1))
        q[0] = self.Cx.get_virtual(t)
        q[1] = self.Cy.get_virtual(t)
        q[2] = self.Cz.get_virtual(t)
        # Inverse solution
        u1 = self.plant.m * (q[0]**2 + q[1]**2 + (q[2]-self.plant.g)**2)**(1/2)
        phid = np.clip(np.arcsin(q[1] * self.plant.m / u1),
                       - np.deg2rad(40), np.deg2rad(40))
        thetad = np.clip(np.arctan(q[0] / (q[2] - self.plant.g)),
                         - np.deg2rad(40), np.deg2rad(40))
        psid = 0
        eulerd = np.vstack([phid, thetad, psid])
        # caculate f
        J = np.diag(J)
        obs_p = self.Cphi.get_obsdot()
        obs_q = self.Ctheta.get_obsdot()
        obs_r = self.Cpsi.get_obsdot()
        f = np.array([(J[1]-J[2]) / J[0] * obs_q * obs_r,
                      (J[2]-J[0]) / J[1] * obs_p * obs_r,
                      (J[0]-J[1]) / J[2] * obs_p * obs_q])

        ''' inner loop control '''
        u2 = self.Cphi.get_u(t, phid, f[0])
        u3 = self.Ctheta.get_u(t, thetad, f[1])
        u4 = self.Cpsi.get_u(t, psid, f[2])
        # rotors
        forces = np.vstack([u1, u2, u3, u4])
        rotors = np.linalg.pinv(env.B.dot(env.get_Lambda(t-env.fault_delay))).dot(forces)

        ''' set derivatives '''
        x, y, z = self.plant.pos.state.ravel()
        self.Cx.set_dot(t, x, posd[0])
        self.Cy.set_dot(t, y, posd[1])
        self.Cz.set_dot(t, z, posd[2])
        self.Cphi.set_dot(t, euler[0], phid, b[0], f[0])
        self.Ctheta.set_dot(t, euler[1], thetad, b[1], f[1])
        self.Cpsi.set_dot(t, euler[2], psid, b[2], f[2])

        # Disturbance
        dist = np.zeros((6, 1))
        dist[0] = self.Cx.get_dist()
        dist[1] = self.Cy.get_dist()
        dist[2] = self.Cz.get_dist()
        dist[3] = self.Cphi.get_dist()
        dist[4] = self.Ctheta.get_dist()
        dist[5] = self.Cpsi.get_dist()

        # Observation
        obs_pos = np.zeros((3, 1))
        obs_pos[0] = self.Cx.get_err()
        obs_pos[1] = self.Cy.get_err()
        obs_pos[2] = self.Cz.get_err()
        obs_ang = np.zeros((3, 1))
        obs_ang[0] = self.Cphi.get_obs()
        obs_ang[1] = self.Ctheta.get_obs()
        obs_ang[2] = self.Cpsi.get_obs()

        # Prescribed bound
        bound_err = self.Cx.get_rho(t)
        bound_ang = self.Cphi.get_rho()

        controller_info = {
            "obs_pos": obs_pos,
            "obs_ang": obs_ang,
            "dist": dist,
            "forces": forces,
            "q": q,
            "eulerd": eulerd,
            "bound_err": bound_err,
            "bound_ang": bound_ang,
        }
        return rotors, controller_info


class outerLoop(BaseEnv):
    PARAM = {
        "alp": np.array([3, 3, 1]),
        "eps": np.array([5, 5, 5]),
        "rho": np.array([1, 0.5]),
        "rho_k": 0.5,
        "K": np.array([4, 15, 0]),
        "theta": 0.7,
    }

    def __init__(self):
        super().__init__()
        self.integ_e = BaseSystem(np.zeros((1,)))

        self.k = self.PARAM.k
        self.rho_0, self.rho_inf = self.PARAM.rho.ravel()
        theta = self.PARAM.theta
        self.theta = np.array([theta, 2*theta-1, 3*theta-2])

    def deriv(self, e, integ_e, y, ref, t):
        alp, eps, theta = self.PARAM.alp, self.PARAM.eps, self.theta
        e_real = y - ref

        if t == 0:
            self.e = BaseSystem(np.vstack([e_real, 0, 0]))

        q = self.get_virtual(t)
        edot = np.zeros((3, 1))
        edot[0, :] = e[1] + (alp[0]/eps) * func_g(eps**2 * (e_real - e[0]), theta[0])
        edot[1, :] = e[2] + q + alp[1] * func_g(eps**2 * (e_real - e[0]), theta[1])
        edot[2, :] = alp[2] * eps * func_g(eps**2 * (e_real - e[0]), theta[2])
        integ_edot = y - ref
        return edot, integ_edot

    def get_virtual(self, t):
        rho_0, rho_inf, k, K = self.rho_0, self.rho_inf, self.k, self.PARAM.K
        e = self.e.state
        integ_e = self.integ_e.state
        rho = (rho_0-rho_inf) * np.exp(-k*t) + rho_inf
        drho = - k * (rho_0-rho_inf) * np.exp(-k*t)
        ddrho = k**2 * (rho_0-rho_inf) * np.exp(-k*t)

        z1 = e[0] / rho
        dz1 = e[1]/rho - e[0]*drho/rho**2
        alpha = - rho*K[0]*z1 + drho*z1 - K[2]*(1-z1**2)*rho**2*integ_e
        z2 = e[1] - alpha
        dalpha = ddrho*z1 + drho*dz1 - drho*K[0]*z1 - rho*K[0]*dz1 \
            - K[2]*(1-z1**2)*(rho**2*e[0]+2*rho*drho*integ_e) \
            + K[2]*2*z1*dz1*rho**2*integ_e
        q = - e[2] + dalpha - K[1]*z2 - z1/(1-z1**2)/rho
        return q

    def set_dot(self, t, y, ref):
        states = self.observe_list()
        self.e.dot, self.integ_e.dot = self.deriv(*states, y, ref, t)

    def get_err(self):
        return self.e.state[0]

    def get_dist(self):
        return self.e.state[2]

    def get_rho(self, t):
        rho_0, rho_inf, k = self.rho_0, self.rho_inf, self.k
        rho = (rho_0-rho_inf) * np.exp(-k*t) + rho_inf
        return rho


class innerLoop(BaseEnv):
    '''
    xi: lower and upper bound of u (moments for my case), [lower, upper]
    rho: bound of state x, dx
    virtual input nu = f + b*u
    '''
    PARAM = {
        "alp": np.array([3, 3, 1]),
        "eps": np.array([10, 10, 10]),
        "xi": np.array([-1, 1]) * 0.15,
        "rho": np.array([30, 90]),
        "c": np.array([20, 20]),
        "K": np.array([20, 25, 0]),
        "theta": 0.7,
    }

    def __init__(self, b):
        super().__init__()
        self.x = BaseSystem(np.zeros((3, 1)))
        self.lamb = BaseSystem(np.zeros((2, 1)))
        self.integ_e = BaseSystem(np.zeros((1,)))

        theta = self.PARAM.theta
        self.theta = np.array([theta, 2*theta-1, 3*theta-2])

    def deriv(self, x, lamb, integ_e, t, y, ref, b, f):
        alp, eps, theta = self.PARAM.alp, self.PARAM.eps, self.theta
        nu = self.get_virtual(t, ref)
        bound = f + b*self.PARAM.xi
        nu_sat = np.clip(nu, bound[0], bound[1])

        xdot = np.zeros((3, 1))
        xdot[0, :] = x[1] + (alp[0]/eps) * func_g(eps**2 * (y - x[0]), theta[0])
        xdot[1, :] = x[2] + nu_sat + alp[1] * func_g(eps**2 * (y - x[0]), theta[1])
        xdot[2, :] = alp[2] * eps * func_g(eps**2 * (y - x[0]), theta[2])
        lambdot = np.zeros((2, 1))
        lambdot[0] = - self.PARAM.c[0]*lamb[0] + lamb[1]
        lambdot[1] = - self.PARAM.c[1]*lamb[1] + (nu_sat - nu)
        integ_edot = y - ref
        return xdot, lambdot, integ_edot

    def get_virtual(self, t, ref):
        K, c, rho = self.PARAM.K, self.PARAM.c, self.PARAM.rho
        x = self.x.state
        lamb = self.lamb.state
        integ_e = self.integ_e.state
        dref, ddref = 0, 0

        z1 = x[0] - ref - lamb[0]
        dz1 = x[1] - dref + c[0]*lamb[0] - lamb[1]
        alpha = - K[0]*z1 - c[0]*lamb[0] - K[2]*integ_e*(rho[0]**2-z1**2)
        z2 = x[1] - dref - alpha - lamb[1]
        dalpha = - K[0]*(x[1] - dref + c[0]*lamb[0] - lamb[1]) \
            - c[0]*(-c[0]*lamb[0] + lamb[1]) - K[2]*(x[0]-ref)*(rho[0]**2-z1**2) \
            + K[2]*2*z1*dz1*integ_e
        nu = - c[1]*lamb[1] + dalpha + ddref - K[1]*z2 \
            - (rho[1]**2 - z2**2)/(rho[0]**2 - z1**2)*z1 - x[2]
        return nu

    def get_u(self, t, ref, b, f):
        nu = self.get_virtual(t, ref)
        bound = f + b*self.PARAM.xi
        nu_sat = np.clip(nu, bound[0], bound[1])
        u = (nu_sat - f) / b
        return u

    def set_dot(self, t, y, ref, b, f):
        states = self.observe_list()
        dots = self.deriv(*states, t, y, ref, b, f)
        self.x.dot, self.lamb.dot, self.integ_e.dot = dots

    def get_obs(self):
        return self.x.state[0]

    def get_obsdot(self):
        return self.x.state[1]

    def get_dist(self):
        return self.x.state[2]

    def get_rho(self):
        return self.rho


if __name__ == "__main__":
    pass
