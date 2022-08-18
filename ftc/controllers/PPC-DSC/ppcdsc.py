import fym
import numpy as np
from numpy import cos, sin, tan
from scipy.spatial.transform import Rotation


def cross(x, y):
    return np.cross(x, y, axis=0)


def get_angles(R):
    return Rotation.from_matrix(R).as_euler("ZYX")[::-1]


class DSCController(fym.BaseEnv):
    def __init__(self):
        super().__init__()
        # filters
        self.filter_vd = fym.BaseSystem(0.000 * np.ones((3, 1)))
        self.filter_Xid0 = fym.BaseSystem(0.000 * np.ones((2, 1)))
        self.filter_omegaXid = fym.BaseSystem(0.000 * np.ones((2, 1)))
        # auxs
        self.varphi = fym.BaseSystem(0.01 * np.ones((2, 1)))
        self.eta = fym.BaseSystem(shape=(4, 1))
        # adaptive
        self.wxihat = fym.BaseSystem()
        self.wchihat = fym.BaseSystem()

    def rbf(self, points, grids, ngrid):
        assert len(points) == len(grids)
        mgrid = np.meshgrid(*(np.linspace(grid[0], grid[1], ngrid) for grid in grids))
        sigmas = np.array([(grid[1] - grid[0]) / ((ngrid - 1) * 2) for grid in grids])
        centers = np.vstack(list(map(np.ravel, mgrid))).T
        rbf = np.exp(-np.sum((points - centers) ** 2 / sigmas**2, axis=-1))
        return rbf[:, None]

    def get_control(self, t, env):
        """quad state"""
        pos, vel, R, omega = env.plant.observe_list()
        angles = get_angles(R)
        phi, theta, psi = angles
        omegaXi, r = omega[:2], omega[2:]

        """ filter state """
        veld = self.filter_vd.state
        Xid0 = self.filter_Xid0.state
        omegaXid = self.filter_omegaXid.state

        """ aux state """
        varphi = self.varphi.state
        eta = self.eta.state

        """ adaptive """
        wxihat = self.wxihat.state
        wchihat = self.wchihat.state

        """ external signals """
        posd, posd_dot = env.get_ref(t, "posd", "posd_dot")

        """ DSC parameters """
        # PPC
        rhop0 = np.vstack((0.8, 0.8, 0.8))
        rhopinf = np.vstack((0.2, 0.2, 0.2))
        Tp = 20
        nrho = 2
        # error transform
        beta_min = np.vstack((1, 1, 1))
        beta_max = np.vstack((1, 1, 1))
        # position control
        kp1 = 2
        varepsp = 0.5
        # velocity control
        tauvel = 0.02
        kvxi1 = 2
        kvxi2 = 1
        kvxi3 = 2
        kvxi4 = 1
        varepsvxi = 2
        varphimin = 0.001
        rbfxi_grids = (
            [-4, 4],
            [-4, 4],
            [-4, 4],
            np.deg2rad([-20, 20]),
            np.deg2rad([-20, 20]),
            np.deg2rad([-180, 180]),
        )
        rbfxi_ngrid = 3
        gammaxi1 = 0.3
        gammaxi2 = 0.01
        # attitude control
        tauXi = 0.02
        kXi1 = 2
        Ximax = np.vstack((0.5, 0.5))
        Ximin = np.vstack((-0.5, -0.5))
        varepsXi = 0.3
        # angular velocity and altitude control
        tauomegaXi = 0.02
        Kchi1 = np.diag([1.5, 0.5, 0.5, 0.02])
        rbfchi_grids = (
            np.deg2rad([-20, 20]),
            np.deg2rad([-20, 20]),
            np.deg2rad([-180, 180]),
            np.deg2rad([-80, 80]),
            np.deg2rad([-80, 80]),
            np.deg2rad([-80, 80]),
        )
        rbfchi_ngrid = 3
        gammachi1 = 0.3
        gammachi2 = 0.01
        # rd = 1 * np.tanh(0.1 * t)
        # rd_dot = 1 * (1 - np.tanh(0.1 * t) ** 2) * 0.1
        rd = 0
        rd_dot = 0

        """ DSC known parameters """
        m = env.plant.m * 1.1
        g = env.plant.g * 1.01
        J = env.plant.J * 1.1
        Jinv = np.linalg.inv(J)
        fTmax = env.plant.rfmax * 4
        B = env.plant.B

        """ error transformation """
        # PPC
        rhop = (
            (rhop0 - 1 * rhopinf) * (1 - t / Tp) ** (nrho + 1) + rhopinf
            if t < Tp
            else rhopinf
        )
        rhop_dot = (
            -1 / Tp * (rhop0 - 1 * rhopinf) * (nrho + 1) * (1 - t / Tp) ** nrho
            if t < Tp
            else 0
        )
        # position error
        ep = pos - posd
        # transformed error
        epsp = np.log(beta_min + ep / rhop) - np.log(beta_max - ep / rhop)
        epspd = np.log(beta_min / beta_max)

        """ position control """
        Gp = np.diag((1 / (beta_min * rhop + ep) + 1 / (beta_max * rhop - ep)).ravel())
        zp = epsp - epspd
        velc = self.get_vc(
            z=zp,
            f=-Gp @ (rhop_dot / rhop * ep + posd_dot),
            G=Gp,
            k1=kp1,
            vareps=varepsp and None,
        )

        if t == 0:
            veld = self.filter_vd.state = velc

        """ horizontal velocity control """
        # velocity
        veld_dot = (velc - veld) / tauvel
        zv = vel - veld

        # horizontal velocity
        vxid_dot, vzd_dot = veld_dot[:2], veld_dot[2:]
        zvxi, zvz = zv[:2], zv[2:]
        R2 = np.array([[cos(psi), -sin(psi)], [sin(psi), cos(psi)]])
        Gvxi = fTmax * R2 / m

        # adaptive term
        Phixi = self.rbf(
            points=np.hstack((vel.ravel(), angles)),
            grids=rbfxi_grids,
            ngrid=rbfxi_ngrid,
        )
        varthetaxi = Phixi.T @ Phixi
        adaptxi = -wxihat * varthetaxi / 2 / gammaxi1**2 * zvxi

        Xic = self.get_vc(
            z=zvxi,
            f=0,
            G=Gvxi,
            k1=kvxi1 + varepsp,
            vareps=varepsvxi and None,
            des_dot=vxid_dot,
            aux=-kvxi2 * varphi + adaptxi,
        )

        if t == 0:
            Xid0 = self.filter_Xid0.state = Xic

        """ pitch and roll angle control """
        Xi = np.vstack([-sin(theta) * cos(phi), sin(phi)])
        # saturation (element-wize)
        Xid = 0.99 * Ximax * np.tanh(Xid0)
        # saturation derivative (element-wise)
        Xid0_dot = (Xic - Xid0) / tauXi
        Xid_dot = 0.99 * Ximax * (1 - np.tanh(Xid0) ** 2) * Xid0_dot
        zXi = Xi - Xid

        H = np.array([[sin(theta) * sin(phi), -cos(theta) * cos(phi)], [cos(phi), 0]])

        kappaover = Ximax - Xid
        kappaover_dot = -Xid_dot
        kappaunder = Xid - Ximin
        kappaunder_dot = Xid_dot

        indXi = zXi > 0

        K1 = (1 - indXi) / (kappaunder**2 - zXi**2) + indXi / (
            kappaover**2 - zXi**2
        )  # This is a vector not a matrix
        K2bar = (1 - indXi) * np.abs(kappaunder_dot / kappaunder) + indXi * np.abs(
            kappaover_dot / kappaover
        )  # This is a vector not a matrix

        GXi = H @ np.array([[1, sin(phi) * tan(theta)], [0, cos(phi)]])
        omegaXic = self.get_vc(
            z=zXi,
            f=H @ np.vstack([cos(phi) * tan(theta), -sin(phi)]) * r,
            G=GXi,
            k1=(kXi1 + varepsvxi / K1 + K2bar),
            aux=-1 * GXi.T @ (K1 * zXi) / 2 / varepsXi,
            vareps=None,
            des_dot=Xid_dot,
        )

        if t == 0:
            omegaXid = self.filter_omegaXid.state = omegaXic

        """ angular velocity and vertical velocity control """

        fchi = np.vstack((g, -Jinv @ cross(omega, J @ omega)))
        Gchi = (
            np.block(
                [
                    [-cos(theta) * cos(phi) / m, np.zeros((1, 3))],
                    [np.zeros((3, 1)), Jinv],
                ]
            )
            @ B
        )
        zchi = np.vstack([zvz, omegaXi - omegaXid, 0 - rd])

        omegaXid_dot = (omegaXic - omegaXid) / tauomegaXi
        chid_dot = np.vstack([vzd_dot, omegaXid_dot, rd_dot])
        Kchi = Kchi1 + np.diag([varepsp, varepsXi, varepsXi, 0])

        # adaptive term
        Phichi = self.rbf(
            points=np.hstack((angles, omega.ravel())),
            grids=rbfchi_grids,
            ngrid=rbfchi_ngrid,
        )
        varthetachi = Phichi.T @ Phichi
        adaptchi = -wchihat * varthetachi / 2 / gammachi1**2 * zchi

        uf = np.linalg.inv(Gchi) @ (-fchi + chid_dot - Kchi @ zchi + adaptchi)
        N = self.Nussbaum(eta)
        rfs0 = N * uf

        """ set derivatives """
        # filters
        self.filter_vd.dot = veld_dot
        self.filter_Xid0.dot = Xid0_dot
        self.filter_omegaXid.dot = omegaXid_dot

        # auxs
        DeltaXi = Xid - Xid0
        varphi_dot = (
            -kvxi3 * varphi
            - (np.abs(zvxi.T @ Gvxi @ DeltaXi) + kvxi4**2 * DeltaXi.T @ DeltaXi / 2)
            / (varphi.T @ varphi)
            * varphi
            + kvxi4 * DeltaXi
        )
        self.varphi.dot = (
            varphi_dot if np.linalg.norm(varphi) > varphimin else np.zeros((2, 1))
        )
        self.eta.dot = -1 * (Gchi.T @ zchi) * uf

        # adaptive laws
        self.wxihat.dot = (
            varthetaxi * zvxi.T @ zvxi / 2 / gammaxi1**2 - gammaxi2 * wxihat
        )
        self.wchihat.dot = (
            varthetachi * zchi.T @ zchi / 2 / gammachi1**2 - gammachi2 * wchihat
        )

        controller_info = {
            "ppc_min": -rhop * beta_min + posd,
            "ppc_max": rhop * beta_max + posd,
            "posd": posd,
            "ep": ep,
            "zp": zp,
            "zv": zv,
            "zXi": zXi,
            "zchi": zchi,
        }

        return rfs0, controller_info

    def get_vc(self, z, f, G, k1, vareps=None, des_dot=0, aux=0):
        """Get virtual control"""
        vc = np.linalg.pinv(G) @ (-f - k1 * z + des_dot + aux)

        if vareps is not None:
            vc -= -1 / (2 * vareps) * G.T @ z
        return vc

    def Nussbaum(self, eta):
        return np.exp(eta**2 / 2) * (eta**2 + 2) * sin(eta) + 1
        # return eta**2 * cos(eta)
