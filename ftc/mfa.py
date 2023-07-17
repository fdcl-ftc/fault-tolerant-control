import ftc


class MFA:
    def __init__(self, env):
        self.controller = ftc.make("Flat", env)
        self.determiner = env.determiner

    def predict(self, tspan, lmbd):
        for t in tspan:
            FM_traj = self.controller.get_control(t)
            nu = FM_traj[2:]
            is_in = self.determiner.determine_is_in(nu, lmbd)
            if not is_in:
                return False
        return True

    def get_nus(self, tspan):
        return [self.controller.get_control(t)[2:] for t in tspan]
