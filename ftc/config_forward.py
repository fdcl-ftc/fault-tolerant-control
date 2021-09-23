"""References
[1] https://www.math.ucsd.edu/~mleok/pdf/LeLeMc2010_quadrotor.pdf
[2] V. S. Akkinapalli, G. P. Falconí, and F. Holzapfel, “Attitude control of a multicopter using L1 augmented quaternion based backstepping,” Proceeding - ICARES 2014 2014 IEEE Int. Conf. Aerosp. Electron. Remote Sens. Technol., no. November, pp. 170–178, 2014.
[3] M. C. Achtelik, K. M. Doth, D. Gurdan, and J. Stumpf, “Design of a multi rotor MAV with regard to efficiency, dynamics and redundancy,” AIAA Guid. Navig. Control Conf. 2012, no. August, pp. 1–17, 2012.
[4] https://kr.mathworks.com/help/aeroblks/6dofquaternion.html#mw_f692de78-a895-4edc-a4a7-118228165a58
[5] M. C. Achtelik, K. M. Doth, D. Gurdan, and J. Stumpf, “Design of a multi rotor MAV with regard to efficiency, dynamics and redundancy,” AIAA Guid. Navig. Control Conf. 2014, no. August, pp. 1–17, 2012, doi: 10.2514/6.2012-4779.
"""
import numpy as np
from functools import reduce

import fym


default_settings = fym.parser.parse({
    # :::::: FTC Modules :::::: #

    # ====== ftc.faults ====== #

    # ------ ftc.faults.manager ------ #

    "faults.manager": {
        "delay": 0.,
        "threshold": 0.,
    },

    # ====== ftc.agents ====== #

    # ------ ftc.agents.switcing ------ #

    "agents.switching": {

        "LQRGainList": [

            # No failure
            {
                "Q": np.diag(np.hstack((
                    [10, 10, 10],
                    [1, 1, 1],
                    [100, 100, 100],
                    [1, 1, 1],
                ))),
                "R": np.diag([1, 1, 1, 1, 1, 1]),
            },

            # One failure
            {
                "Q": np.diag(np.hstack((
                    [10, 10, 10],
                    [1, 1, 1],
                    [100, 100, 100],
                    [1, 1, 1],
                ))),
                "R": np.diag([1, 1, 1, 1, 1, 1]),
            },

            # Two failures
            {
                "Q": np.diag(np.hstack((
                    [1000, 1000, 1000],
                    [100, 100, 100],
                    [0, 0, 0],
                    [1, 1, 1],
                ))),
                "R": np.diag([1, 1, 1, 1, 1, 1]),
            },
        ],
    },

    # ====== ftc.plants ====== #

    # ------ ftc.plants.multicopter ------ #

    "models.multicopter": {
        # Initial states
        "init": {
            "pos": np.zeros((3, 1)),
            "vel": np.zeros((3, 1)),
            "quat": np.vstack((1, 0, 0, 0)),
            "omega": np.zeros((3, 1)),
        },

        # Mixer
        "mixer.rtype": "hexa-x",

        # Physical properties
        "physProp": {
            # General physical constants
            "g": 9.81,
            "rho": 1.225,

            # Parameters from Baldini et al., 2020
            "kr": 1e-3 * np.eye(3),  # Rotational friction coefficient [N*s*m/rad]
            "Jr": 6e-5,  # Rotor inertia [N*m]
            "CdA": 0.08,  # Flat plate area [m^2]
            "R": 0.15,  # Rotor radius [m]
            "ch": 0.04,  # Propeller chord [m]
            "a0": 6,  # Slope of the lift curve per radian [-]

            # Parameters from P. Pounds et al., 2010
            "sigma": 0.054,  # Solidity ratio [-]
            "thetat": np.deg2rad(4.4),  # Blade tip angle [rad]
            "CT": 0.0047,  # Thrust coefficient [-]
        },

        # Physical properties by several authors
        "modelFrom": "Taeyoung_Lee",

        "physPropBy": {
            # Prof. Taeyoung Lee's model for quadrotor UAV [1]
            "Taeyoung_Lee": {
                "J": np.diag([0.0820, 0.0845, 0.1377]),
                "m": 4.34,
                "d": 0.315,  # distance to each rotor from the center of mass
                "c": 8.004e-4,  # z-dir moment coefficient caused by rotor force
                "b": 1,
                "rotor_min": 0,
            },

            # G. P. Falconi's multicopter model [2-4]
            "GP_Falconi": {
                "J": np.diag([0.010007, 0.0102335, 0.0081]),
                "m": 0.64,
                "d": 0.215,  # distance to each rotor from the center of mass
                "c": 1.2864e-7,  # z-dir moment coefficient caused by rotor force
                "b": 6.546e-6,
                "rotor_min": 0,
            },
        },
    },

    # :::::: FTC EVALUATION ENV :::::: #

    # ====== env ====== #

    "parallel.max_workers": None,
    "episode.N": 100,
    "episode.range": {
        "pos": (-1, 1),
        "vel": (-1, 1),
        "vel_x": (3, 7),
        "omega": np.deg2rad((-2, 2)),
        "angle": np.deg2rad((-2, 2)),
        "angle_theta": np.deg2rad((-10, -5)),
    },
    "evaluation.cuttime": 5,
    "evaluation.threshold": 0.5,
    "env.kwargs": {
        "dt": 0.01,
        "max_t": 10,
    },
    "ref.pos": np.vstack((10, 0, 0)),
})

settings = fym.parser.parse(default_settings)


def _maximum_thrust(m, g):
    return m * g * 0.6371  # maximum thrust for each rotor [5]


def _set_maximum_rotor(settings):
    modelauthor = settings.models.multicopter.modelFrom
    g = settings.models.multicopter.physProp.g
    if modelauthor == "Taeyoung_Lee":
        m = settings.models.multicopter.physPropBy.Taeyoung_Lee.m
        rotor_max = {"models.multicopter.physPropBy.Taeyoung_Lee":
                     {"rotor_max": _maximum_thrust(m, g)}
                     }
    elif modelauthor == "GP_falconi":
        rotor_max = {"models.multicopter.physPropBy.GP_falconi":
                     {"rotor_max": 3e5}  # about 2 * m * g / b / 6
                     }
    fym.parser.update(settings, rotor_max)


_set_maximum_rotor(settings)


def load(key=None):
    if isinstance(key, str):
        chunks = key.split(".")
        if key.startswith("ftc"):
            chunks.pop(0)
        return reduce(lambda v, k: v.__dict__[k], chunks, settings)
    return settings


def set(d):
    fym.parser.update(settings, d)


def reset():
    fym.parser.update(settings, default_settings, prune=True)
