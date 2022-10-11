from fym.core import BaseEnv, BaseSystem
from fym.utils.rot import quat2angle

import numpy as np


def func_g(x, theta):
    delta = 1
    if abs(x) < delta:
        return x / delta**(1-theta)
    else:
        return np.sign(x) * abs(x)**theta


def q(x):
    if x > 0:
        return 1
    else:
        return 0


class BLFController(BaseEnv):
    def __init__(self, env):
        super().__init__()
        # controller gain
        alp = np.array([3, 3, 1])
        rho = np.array([1, 0.5])
        rho_k = 0.5
        theta = 0.7
        # controllers
        env_config = env.env_config
        self.Cx = outerLoop(alp, env_config["eps11"], rho, rho_k, theta,
                            np.array([env_config["k11"], env_config["k12"],
                                      env_config["k13"]]))
        self.Cy = outerLoop(alp, env_config["eps12"], rho, rho_k, theta,
                            np.array([env_config["k11"], env_config["k12"],
                                      env_config["k13"]]))
        self.Cz = outerLoop(alp, env_config["eps13"], rho, rho_k, theta,
                            np.array([env_config["k11"], env_config["k12"],
                                      env_config["k13"]]))
        rho = np.deg2rad([45, 130])
        xi = np.array([-1, 1]) * 10
        c = np.zeros((2,))
        self.Cphi = innerLoop(alp, env_config["eps21"], xi, rho, c, theta,
                              np.array([env_config["k21"], env_config["k22"],
                                        env_config["k23"]]))
        self.Ctheta = innerLoop(alp, env_config["eps22"], xi, rho, c, theta,
                                np.array([env_config["k21"], env_config["k22"],
                                          env_config["k23"]]))
        self.Cpsi = innerLoop(alp, env_config["eps23"], xi, rho, c, theta,
                              np.array([env_config["k21"], env_config["k22"],
                                        env_config["k23"]]))

        self.dx1, self.dx2, self.dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        self.dy1, self.dy2 = env.plant.dy1, env.plant.dy2
        c, self.c_th = 0.0338, 128  # tq / th, th / rcmds
        self.B_r2f = np.array((
            [-1, -1, -1, -1, -1, -1],
            [-self.dy2, self.dy1, self.dy1, -self.dy2, -self.dy2, self.dy1],
            [-self.dx2, -self.dx2, self.dx1, -self.dx3, self.dx1, -self.dx3],
            [-c, c, -c, c, c, -c]
        ))

    def get_control(self, t, env):
        ''' quad state '''
        pos, vel, quat, omega = env.plant.observe_list()
        euler = np.vstack(quat2angle(quat)[::-1])

        ''' plant parameters '''
        m = env.plant.m
        g = env.plant.g
        J = np.diag(env.plant.J)
        b = np.array([1/J[0], 1/J[1], 1/J[2]])

        ''' external signals '''
        posd, posd_dot = env.get_ref(t, "posd", "posd_dot")

        ''' outer loop control '''
        q = np.zeros((3, 1))
        q[0] = self.Cx.get_virtual(t)
        q[1] = self.Cy.get_virtual(t)
        q[2] = self.Cz.get_virtual(t)

        # Inverse solution
        u1 = m * (q[0]**2 + q[1]**2 + (q[2]-g)**2)**(1/2)
        phid = np.clip(np.arcsin(- q[1] * m / u1),
                       - np.deg2rad(45), np.deg2rad(45))
        thetad = np.clip(np.arctan(q[0] / (q[2] - g)),
                         - np.deg2rad(45), np.deg2rad(45))
        psid = 0
        eulerd = np.vstack([phid, thetad, psid])

        ''' inner loop control '''
        y_phi = np.vstack([euler[0], omega[0]])
        y_theta = np.vstack([euler[1], omega[1]])
        y_psi = np.vstack([euler[2], omega[2]])
        u2 = self.Cphi.get_u(t, y_phi, phid, b[0])
        u3 = self.Ctheta.get_u(t, y_theta, thetad, b[1])
        u4 = self.Cpsi.get_u(t, y_psi, psid, b[2])
        ctr_forces = np.vstack([q, u2, u3, u4])

        # rotors
        ctrls1 = np.vstack([u1, u2, u3, u4])
        th = np.linalg.pinv(self.B_r2f) @ ctrls1
        pwms_rotor = (th / self.c_th) * 1000 + 1000
        forces = np.vstack((
            pwms_rotor,
            np.vstack(env.plant.u_trims_fixed)
        ))

        ''' set derivatives '''
        x, y, z = env.plant.pos.state.ravel()
        self.Cx.set_dot(t, x, posd[0])
        self.Cy.set_dot(t, y, posd[1])
        self.Cz.set_dot(t, z, posd[2])
        self.Cphi.set_dot(t, euler[0], phid, b[0])
        self.Ctheta.set_dot(t, euler[1], thetad, b[1])
        self.Cpsi.set_dot(t, euler[2], psid, b[2])

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
            "bound_err": bound_err,
            "bound_ang": bound_ang,
            "posd": posd,
            "angd": eulerd,
            "ang": euler,
            "ctr_forces": ctr_forces,
        }
        return ctr_forces, forces, controller_info


class outerLoop(BaseEnv):
    def __init__(self, alp, eps, rho, rho_k, theta, K):
        super().__init__()
        self.e = BaseSystem(np.zeros((3, 1)))
        self.integ_e = BaseSystem(np.zeros((1,)))

        self.alp, self.eps, self.k = alp, eps, rho_k
        self.rho_0, self.rho_inf = rho.ravel()
        self.theta = np.array([theta, 2*theta-1, 3*theta-2])
        self.K = np.array([2, 10, 0])

    def deriv(self, e, integ_e, y, ref, t):
        alp, eps, theta = self.alp, self.eps, self.theta
        e_real = y - ref

        if t == 0:
            e = self.e.state = np.vstack([e_real, 0, 0])

        q = self.get_virtual(t)
        edot = np.zeros((3, 1))
        edot[0, :] = e[1] + (alp[0]/eps) * func_g(eps**2 * (e_real - e[0]), theta[0])
        edot[1, :] = e[2] + q + alp[1] * func_g(eps**2 * (e_real - e[0]), theta[1])
        edot[2, :] = alp[2] * eps * func_g(eps**2 * (e_real - e[0]), theta[2])
        integ_edot = y - ref
        return edot, integ_edot

    def get_virtual(self, t):
        rho_0, rho_inf, k, K = self.rho_0, self.rho_inf, self.k, self.K
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
    def __init__(self, alp, eps, xi, rho, c, theta, K):
        super().__init__()
        self.x = BaseSystem(np.zeros((3, 1)))
        self.lamb = BaseSystem(np.zeros((2, 1)))
        self.integ_e = BaseSystem(np.zeros((1,)))

        self.alp, self.eps, self.rho, self.xi, self.c = alp, eps, rho, xi, c
        self.K = K
        self.theta = np.array([theta, 2*theta-1, 3*theta-2])

    def deriv(self, x, lamb, integ_e, t, y, ref, b):
        alp, eps, theta = self.alp, self.eps, self.theta
        nu = self.get_virtual(t, y, ref)
        bound = b*self.xi
        nu_sat = np.clip(nu, bound[0], bound[1])

        xdot = np.zeros((3, 1))
        xdot[0, :] = x[1] + (alp[0]/eps) * func_g(eps**2 * (y - x[0]), theta[0])
        xdot[1, :] = x[2] + nu_sat + alp[1] * func_g(eps**2 * (y - x[0]), theta[1])
        xdot[2, :] = alp[2] * eps * func_g(eps**2 * (y - x[0]), theta[2])
        lambdot = np.zeros((2, 1))
        lambdot[0] = - self.c[0]*lamb[0] + lamb[1]
        lambdot[1] = - self.c[1]*lamb[1] + (nu_sat - nu)
        integ_edot = y - ref
        return xdot, lambdot, integ_edot

    def get_virtual(self, t, y, ref):
        K, c, rho = self.K, self.c, self.rho
        x = self.x.state
        lamb = self.lamb.state
        if t == 0:
            dlamb = np.zeros((2, 1))
        else:
            dlamb = self.lamb.dot
        integ_e = self.integ_e.state
        dref = 0

        rho1a = ref + lamb[0] + rho[0]
        rho1b = rho[0] - ref - lamb[0]
        drho1a = dlamb[0]
        drho1b = - dlamb[0]
        ddrho1a = - c[0]*dlamb[0] + dlamb[1]
        ddrho1b = c[0]*dlamb[0] - dlamb[1]

        z1 = x[0] - ref - lamb[0]
        dz1 = x[1] - dref - dlamb[0]

        xi_1a = z1 / rho1a
        dxi_1a = (dz1*rho1a-z1*drho1a) / (rho1a**2)
        xi_1b = z1 / rho1b
        dxi_1b = (dz1*rho1b-z1*drho1b) / (rho1b**2)
        xi1 = q(z1)*xi_1b + (1-q(z1))*xi_1a
        dxi1 = q(z1)*dxi_1b + (1-q(z1))*dxi_1a

        bar_k1 = ((drho1a/rho1a)**2 + (drho1b/rho1b)**2 + 0.1) ** (1/2)
        alpha = - (K[0] + bar_k1)*z1 - c[0]*lamb[0] - K[2]*integ_e*(1-xi1**2)

        dbar_k1 = 1 / 2 / bar_k1 * (
            2*drho1a*(ddrho1a*rho1a-drho1a**2)/(rho1a**3)
            + 2*drho1b*(ddrho1b*rho1b-drho1b**2)/(rho1b**3)
        )
        dalpha = (- dbar_k1*z1 - bar_k1*dz1 - c[0]*dlamb[0]
                  - K[2]*(x[0]-ref)*(1-xi1**2)
                  - K[2]*integ_e*(-2*xi1*dxi1))

        rho2a = alpha + lamb[1] + rho[1]
        rho2b = rho[1] - alpha - lamb[1]
        drho2a = dalpha + dlamb[1]
        drho2b = - dalpha - dlamb[1]

        z2 = x[1] - alpha - lamb[1]

        mu1 = q(z1)/(rho1b**2-z1**2) + (1-q(z1))/(rho1a**2-z1**2)
        bar_k2 = ((drho2a/rho2a)**2 + (drho2b/rho2b)**2 + 0.1) ** (1/2)
        nu = - (K[1] + bar_k2)*z2 + dalpha - x[2] - c[1]*lamb[1] - mu1*z1

        return nu

    def get_u(self, t, y, ref, b):
        nu = self.get_virtual(t, y, ref)
        # bound = b*self.xi
        # nu_sat = np.clip(nu, bound[0], bound[1])
        # u = nu_sat / b
        u = nu / b
        return u

    def set_dot(self, t, y, ref, b):
        states = self.observe_list()
        dots = self.deriv(*states, t, y, ref, b)
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
