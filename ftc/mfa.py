import numpy as np

import ftc
from ftc.mission_determiners.polytope_determiner import PolytopeDeterminer


class MFA:
    def __init__(self, env):
        pwm_min, pwm_max = env.plant.control_limits["pwm"]
        self.determiner = PolytopeDeterminer(
            pwm_min * np.ones(6), pwm_max * np.ones(6), self.allocator
        )
        self.controller = ftc.make("Flat", env)

        dx1, dx2, dx3 = env.plant.dx1, env.plant.dx2, env.plant.dx3
        dy1, dy2 = env.plant.dy1, env.plant.dy2
        c, self.c_th = 0.0338, 128  # tq / th, th / rcmds
        self.B_r2f = np.array(
            (
                [-1, -1, -1, -1, -1, -1],
                [-dy2, dy1, dy1, -dy2, -dy2, dy1],
                [-dx2, -dx2, dx1, -dx3, dx1, -dx3],
                [-c, c, -c, c, c, -c],
            )
        )

    def allocator(self, nu, lmbd=np.ones(6)):
        nu_f = np.vstack((-nu[0], nu[1:]))
        th = np.linalg.pinv(lmbd * self.B_r2f) @ nu_f
        pwms_rotor = (th / self.c_th) * 1000 + 1000
        return pwms_rotor

    def predict(self, tspan, lmbd):
        for t in tspan:
            FM_traj = self.controller.get_control(t)
            nu = FM_traj[2:]
            is_in = self.determiner.determine_is_in(nu, lmbd)
            if not is_in:
                return False
        return True
