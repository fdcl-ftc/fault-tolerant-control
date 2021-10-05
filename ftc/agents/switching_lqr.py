import numpy as np
import scipy.optimize
import itertools
import sys
from loguru import logger

from fym.agents.LQR import clqr
from fym.utils.linearization import jacob_analytic
import fym.utils.rot as rot

import ftc.config

logger.remove()
logger.add(sys.stdout, level="INFO")

cfg = ftc.config.load(__name__)


def angle2quat(angle):
    """angle: phi, theta, psi in radian"""
    return rot.angle2quat(*np.ravel(angle)[::-1])


def quat2angle(quat):
    """angle: phi, theta, psi in radian"""
    return np.vstack(rot.quat2angle(quat)[::-1])


def omega2dangle(omega, phi, theta):
    dangle = np.array([
        [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
        [0, np.cos(phi), -np.sin(phi)],
        [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
    ]) @ omega
    return dangle


def wrap(func, indices):
    def deriv(x, u):
        pos, vel, angle, omega = x[:3], x[3:6], x[6:9], x[9:12]
        quat = angle2quat(angle)
        u = u.copy()
        u[indices, :] = 0
        dpos, dvel, _, domega = func(pos, vel, quat, omega, u)
        phi, theta, _ = angle.ravel()
        dangle = omega2dangle(omega, phi, theta)
        xdot = np.vstack((dpos, dvel, dangle, domega))
        return xdot
    return deriv


class LQR:
    def __init__(self, A, B, Q, R, xtrim, utrim):
        self.A, self.B = A, B
        self.Q, self.R = Q, R
        self.xtrim, self.utrim = xtrim, utrim

        try:
            K, _ = clqr(A, B, Q, R)
        except np.linalg.LinAlgError:
            K = None

        self.K = K

    def get(self, x):
        return self.utrim - self.K @ (x - self.xtrim)


class LQRLibrary:
    def __init__(self, plant):
        self.plant = plant
        m = self.plant.mixer.B.shape[1]

        # LQR table
        keys = [()]
        keys += [(k,) for k in range(m)]
        keys += itertools.combinations(range(m), 2)
        self.lqr_table = dict.fromkeys(keys)

        logger.debug("LQR table keys: {}", keys)

        for indices in list(self.lqr_table.keys()):
            logger.debug("indices: {} (len: {})", indices, len(indices))

            deriv = wrap(plant.deriv, indices)
            xtrim, utrim = self.get_trims(deriv, indices)

            A = jacob_analytic(deriv, 0)(xtrim, utrim)[:, :, 0]
            B = jacob_analytic(deriv, 1)(xtrim, utrim)[:, :, 0]

            Q = cfg.LQRGainList[len(indices)]["Q"]
            R = cfg.LQRGainList[len(indices)]["R"]
            self.lqr_table[indices] = LQR(A, B, Q, R, xtrim, utrim)

            if (K := self.lqr_table[indices].K) is None:
                logger.info("LQR Table ({}): {}", indices, K)

            if len(indices) > 1:
                self.lqr_table[tuple(reversed(indices))] = self.lqr_table[indices]

        logger.info("LQR Table has been succesfully created")

    def get_trims(self, deriv, indices):
        def cost(u, x):
            """u is a (n, ) array of rotor forces without faulty rotors"""
            dx = deriv(x[:, None], u[:, None])
            return np.ravel(dx.T @ dx)

        xtrim = np.zeros(12)

        weight = self.plant.m * self.plant.g
        nrotors = self.plant.mixer.B.shape[1]
        u0 = np.ones(nrotors) * weight / (nrotors - len(indices))
        u0[indices, ] = 0

        logger.debug("u0: {}", u0)

        bounds = ((0, self.plant.rotor_max), ) * nrotors

        result = scipy.optimize.minimize(
            cost, u0, args=(xtrim, ), method="SLSQP",
            bounds=bounds,
        )
        utrim = result.x
        return xtrim[:, None], utrim[:, None]

    def transform(self, y):
        return np.vstack((y[0:6], quat2angle(y[6:10]), y[10:]))

    def get_rotors(self, obs, ref, fault_index):
        x = self.transform(obs)
        x_ref = self.transform(ref)
        indices = tuple(fault_index)
        rotors = self.lqr_table[indices].get(x - x_ref)
        return rotors
