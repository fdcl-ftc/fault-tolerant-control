import fym
import numpy as np 
from fym.utils.rot import quat2angle


class PIDcontroller(fym.BaseEnv):
    def  __init__(self, env):
        super().__init__()
        
    def get_control(self, t, env)
    pos, vel, quat, omega = env.plant.observe_list()
    ang = np.vstack(quat2angle(quat)[::-1]) 
    angd = np.deg2rad(np.vstack((0,0,0)))
    omegad = np.zeors((3,1))
    Kp = np.diag([1,1,1])
    Kd = np.diag([1,1,1])
    torqus = -Kp @ (ang-angd) - Kd @ (omega - omegad)
    # forces = np.vstack((env.plant.mixer)) @ torqus
    # ctrls = np.linalg.inv(env.plant.mixer) @ forces
    np.env.plant.mixer
    
    
    
    controller_info = {
        "angd" : angd,
        "ang" : ang,
    }    
    return ctrl, controller_info 