import fym
import numpy as np

from fym.utils.rot import quat2angle

class PIDController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.ang_int = fym.BaseSystem(np.zeros((3, 1)))
        
    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        
        posd, _ = env.get_ref(t, "posd", "posd_dot")
        
        angd = np.deg2rad(np.vstack((0, 0, 0)))
        omegad = np.zeros((3,1))
        Kp = np.diag([1, 1, 1])
        Kd = np.diag([1, 1, 1])
        Ki = np.diag([0.1, 0.1, 0.1])
        ang_int = self.ang_int.state
        
        torques = -Kp @ (ang - angd) - Kd @ (omega - omegad) - Ki @ (ang_int)
        forces = np.vstack((env.plant.m * env.plant.g, torques))
        
        ctrls = np.linalg.pinv(env.plant.mixer.B) @ forces
        
        self.ang_int.dot = ang
        
        controller_info = {
            "angd": angd,
            "ang": ang,
            "posd": posd,
        }
        
        return ctrls, controller_info