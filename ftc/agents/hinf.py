import numpy as np
from scipy.linalg import sqrtm, block_diag
import cvxpy as cp
from scipy.optimize import minimize_scalar
from control import ctrb

import fym
from fym.utils.linearization import jacob_analytic
import fym.utils.rot as rot
import fym.config

from ftc.agents.switching_lqr import LQR, LQRLibrary, omega2dangle


fym.config.register(
    {
        "Q": {
            "int_err.pos": {
                "x": 4,
                "y": 4,
                "z": 4,
            },
            "int_err.psi": 0.001,
            "pos": {
                "x": 0.1,
                "y": 0.1,
                "z": 0.1,
            },
            "vel": {
                "x": 0.001,
                "y": 0.001,
                "z": 0.001,
            },
            "angles": {
                "phi": 0.1,
                "theta": 0.1,
                "psi": 0,
            },
            "omega": {
                "p": 0.1,
                "q": 0.1,
                "r": 0.1,
            },
        },
        "R": {
            "thrust": 10,
            "moment": {
                "L": 100000,
                "M": 100000,
                "N": 100000,
            },
        },
    },
    base=__name__,
)
cfg = fym.config.load(__name__)


class HinfSolver:
    """H-infinity synthesis for linear systems [1]_

    References
    ----------
    .. [1] Z.Liu et al., 2015, doi.org/10.1007/s10846-015-0293-0
    """
    def __init__(self, A, B, G, Q, R, xtrim, utrim,
                 use_preset=False, preset_name="hinf_preset.h5"):
        self.A, self.B, self.G = A, B, G
        self.Q, self.R = Q, R
        self.xtrim, self.utrim = xtrim, utrim

        self.X = cp.Variable(A.shape, PSD=True)
        self.Y = cp.Variable(B.T.shape)
        self.Z = cp.Variable(A.shape, PSD=True)
        self.gamma = cp.Variable(nonneg=True)  # gamma**2 in the paper

        self.obj = cp.Minimize(cp.trace(self.Z))

        if use_preset:
            data = fym.load(preset_name)

            self.X.value = data["X"]
            self.Y.value = data["Y"]
            self.Z.value = data["Z"]
            self.gamma.value = data["gamma"][0]

            self.K = data["K"]
        else:
            self.prob = cp.Problem(self.obj, self.make_consts())
            self.prob.solve(solver=cp.MOSEK, verbose=True)

            self.K = self.Y.value @ np.linalg.inv(self.X.value)

            fym.save(
                "hinf_preset.h5",
                {
                    "X": self.X.value,
                    "Y": self.Y.value,
                    "Z": self.Z.value,
                    "K": self.K,
                    "gamma": [self.gamma.value],
                    "obj": [self.obj.value],
                },
                info={
                    "status": self.prob.status,
                }
            )

    def make_consts(self):
        def He(X):
            """Hermitian operator"""
            return X + X.T

        def symm(*arr, size=None):
            """Make a block symmetric matrix from lower triangular components"""
            size = int(size or - 0.5 + np.sqrt(0.5**2 + 2 * len(arr)))
            assert len(arr) == size * (size + 1) / 2
            barr = np.zeros((size, size), dtype="O")
            indl = np.tril_indices(size)
            barr[indl] = arr
            barr.T[indl] = list(map(cp.transpose, barr[indl]))
            return cp.bmat(barr.tolist())

        A, B, G = self.A, self.B, self.G
        Q, R = self.Q, self.R
        X, Y, Z, gamma = self.X, self.Y, self.Z, self.gamma

        nx, nu = B.shape
        nw = G.shape[1]

        consts = [
            symm(
                He(A @ X + B @ Y),
                G.T, - gamma * np.eye(nw),
                sqrtm(R) @ Y, np.zeros((nu, nw)), -np.eye(nu),
                sqrtm(Q) @ X, np.zeros((nx, nw)), np.zeros((nx, nu)), -np.eye(nx)
            ) << 0,
            cp.bmat([[Z, np.eye(nx)], [np.eye(nx), X]]) >> 0,
        ]

        return consts

    def get(self, x):
        return self.utrim + self.K @ (x - self.xtrim)


class Hinf(fym.BaseEnv):
    """H-infinity control for switched systems."""
    def __init__(self, plant, config=dict(use_preset=False)):
        """Initialize Hinf

        Parameters
        ----------
        plant : class
            The plant instance, e.g., `ftc.models.multicopter.Multicopter`.
        config : dict, optional
            A configuration for the Hinf class.

            use_preset : bool
            Set to True to use local H-infinity gain file. Default: ``False``.

            preset_name: str
                The name of preset file. Default: ``hinf_preset.h5``.

        """
        super().__init__()

        # Linearize the model
        def deriv(x, u):
            x = self.from_lin(x)

            forces = u
            rotors = plant.mixer(forces)

            plant.state = x
            plant.set_dot(0, rotors)

            dx = plant.dot.copy()
            angles = x[6:9]
            omega = plant.omega.state
            dangles = omega2dangle(omega, *angles[:2].ravel())
            dx = np.insert(np.delete(dx, slice(6, 10), 0), 6, dangles, 0)
            return dx

        xtrim = np.zeros((12, 1))  # pos, vel, angles, omega
        utrim = np.vstack((plant.m * plant.g, 0, 0, 0))  # force and moments
        A = jacob_analytic(deriv, 0)(xtrim, utrim)[:, :, 0]
        B = jacob_analytic(deriv, 1)(xtrim, utrim)[:, :, 0]
        G = np.vstack((
            np.zeros((9, 3)),
            np.eye(3),
        ))

        # Augmented model
        C = np.block([
            [np.eye(3), np.zeros((3, 9))],  # x, y, z
            [np.zeros((1, 8)), 1, np.zeros((1, 3))],  # psi
        ])
        self.err_int = fym.BaseSystem(shape=(4, 1))

        nu = B.shape[1]
        ny, nx = C.shape
        Aa = np.block([[np.zeros((ny, ny)), -C], [np.zeros((nx, ny)), A]])
        Ba = np.vstack([np.zeros((ny, nu)), B])
        Ga = block_diag(np.eye(ny), G)

        Q = np.diag(np.hstack(list(fym.parser.wind(cfg.Q).values())))
        R = np.diag(np.hstack(list(fym.parser.wind(cfg.R).values())))

        xtrim = np.vstack([np.zeros((C.shape[0], 1)), xtrim])

        self.cntr = HinfSolver(Aa, Ba, Ga, Q, R, xtrim, utrim, **config)
        self.C = C

    def from_lin(self, x):
        angles = x[6:9]
        quat = self.angles2quat(angles)
        return np.insert(np.delete(x, slice(6, 9), 0), 6, quat, 0)

    def to_lin(self, x):
        quat = x[6:10]
        angles = self.quat2angles(quat)
        return np.insert(np.delete(x, slice(6, 10), 0), 6, angles, 0)

    def angles2quat(self, angles):
        return rot.angle2quat(*angles[::-1])

    def quat2angles(self, quat):
        return np.vstack(rot.quat2angle(quat))[::-1]

    def get_rotors(self, plant, ref, fault_index):
        """Get the rotor outputs and information.

        Parameter
        ---------
        plant : class
            The plant instance, e.g., `ftc.models.multicopter.Multicopter`.
        ref : (13, 1) array_like
            The reference signal. The elements are as follows.

            ref[0:3] : The reference position.
            ref[3:6] : The reference velocity.
            ref[6:10] : The reference quaternion.
            ref[10:13] : The reference angluar velocity.
        fault_index : (N, ) array_like
            The array of rotor indices where the faults occur.

        """
        x = self.to_lin(plant.state)
        xref = self.to_lin(ref)
        xa = np.vstack((self.err_int.state, x - xref))
        forces = self.cntr.get(xa)
        rotors = plant.mixer(forces)

        # update dot
        y = self.C @ x
        yref = self.C @ xref
        self.err_int.dot = yref - y

        info = dict()
        return rotors, info


class SwitchingHinf(Hinf):
    """H-infinity control with switching LQR."""
    def __init__(self, plant, config):
        """Initialize SwitchingHinf

        Parameters
        ----------
        plant : class
            The plant instance, e.g., `ftc.models.multicopter.Multicopter`.
        config : dict, optional
            A configuration for the Hinf class.

            use_preset : bool
            Set to True to use local H-infinity gain file. Default: ``False``.

            preset_name: str
                The name of preset file. Default: ``hinf_preset.h5``.

        """
        super().__init__(plant, config)
        self.lqrlib = LQRLibrary(plant)

    def get_rotors(self, plant, ref, fault_index):
        if len(fault_index) == 0:
            rotors, info = super().get_rotors(plant, ref, fault_index)
        else:
            rotors, info = self.lqrlib.get_rotors(plant, ref, fault_index)

        return rotors, info
