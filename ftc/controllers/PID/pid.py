import fym
import numpy as np

from fym.utils.rot import quat2angle


class PIDController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        self.ang_int = fym.BaseSystem(np.zeros((3, 1)))
        self.pos_int = fym.BaseSystem(np.zeros((3, 1)))

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        
        posd, posd_dot = env.get_ref(t, "posd", "posd_dot")


        angd = np.deg2rad(np.vstack((0, 0, 0)))
        omegad = np.zeros((3, 1))
        
        Kp_pos = np.diag([1.3, 1.3, 2])
        Kd_pos = np.diag([2.2, 2.2, 1])
        Ki_pos = np.diag([0.01, 0.01, 1])
        
        Kp_att = np.diag([8, 8, 5])
        Kd_att = np.diag([5, 5, 1])
        Ki_att = np.diag([0.1, 0.1, 0.1])
        
        pos_int = self.ang_int.state
        ang_int = self.ang_int.state
        
        pos_error = pos-posd
        vel_error = vel-posd_dot

         # positioning control (acceleration control)
        accd= -Kp_pos @ pos_error - Kd_pos @ vel_error - Ki_pos @ pos_int
        accd[2]= (env.plant.g + accd[2]) / (np.cos(ang[0])) * np.cos(ang[1])
        thrust = -env.plant.m * accd[2]

        accd_mag= np.linalg.norm(accd)
        if accd_mag ==0:
            accd_mag ==1
            
        angd = np.vstack(
            (
                np.arcsin(accd[1]/accd_mag),
                np.arcsin(-accd[0] / accd_mag / np.cos(ang[0])),
                0,
            )
        )
        
        ang_error = ang - angd
        omega_error = omega - omegad
        
        #attitude control
        torques = -Kp_att @ ang_error - Kd_att @ omega_error - Ki_att @ ang_int

        forces = np.vstack((thrust, torques))

        ctrls = np.linalg.pinv(env.plant.mixer.B) @ forces

        self.pos_int.dot = pos_error
        self.ang_int.dot = ang_error

        controller_info = {
            "angd": angd,
            "ang": ang,
            "posd": posd,
        }

        return ctrls, controller_info


# """References
# [1] J. Li and Y. Li, "Dynamic analysis and PID control for a quadrotor," 2011 IEEE International Conference on Mechatronics and Automation, Beijing, China, 2011, pp. 573-578, doi: 10.1109/ICMA.2011.5985724.
# """
# import fym
# import numpy as np
# from fym.utils.rot import quat2angle

# class PIDController(fym.BaseEnv):
#     def __init__(self, env):
#         super().__init__()
#         self.ang_int = fym.BaseSystem(np.zeros((3, 1)))
#         self.pos_int = fym.BaseSystem(np.zeros((3, 1)))
    
#     def get_control(self, t, env):
#         pos, vel, quat, omega = env.plant.observe_list()
#         ang = np.vstack(quat2angle(quat)[::-1])
        
#         posd, posd_dot = env.get_ref(t, "posd", "posd_dot")
        
#         angd = np.deg2rad(np.vstack((0, 0, 0)))
#         omegad = np.zeros((3, 1))
        
#         Kp_pos = np.diag([1.3, 1.3, 2])
#         Kd_pos = np.diag([2.2, 2.2, 1])
#         Ki_pos = np.diag([0.01, 0.01, 1])
        
#         Kp_att = np.diag([8, 8, 5])
#         Kd_att = np.diag([5, 5, 1])
#         Ki_att = np.diag([0.1, 0.1, 0.01])
        
#         pos_int = self.pos_int.state
#         ang_int = self.ang_int.state
        
#         pos_e = pos - posd
#         vel_e = vel - posd_dot
        
#         accd = -Kp_pos @ pos_e - Kd_pos @ vel_e - Ki_pos @ pos_int
#         accd[2] = (env.plant.g * accd[2]) / (np.cos(ang[0]) * np.cos(ang[1]))
#         thrust = -env.plant.m * accd[2]
        
#         accd_mag = np.linalg.norm(accd)
#         if accd_mag == 0:
#             accd_mag = 1
            
#         angd = np.vstack(
#             (
#                np.arcsin(accd[1] / accd_mag),
#                np.arcsin(-accd[0] / accd_mag / np.cos(ang[0])),
#                0,
#             )
#         ) 
        
#         ang_e = ang - angd
#         omega_e = omega - omegad
        
#         # attitude control
#         torques = -Kp_att @ ang_e - Kd_att @ omega_e - Ki_att @ ang_int
        
#         forces = np.vstack((thrust, torques))
        
#         ctrls = np.linalg.pinv(env.plant.mixer.B) @ forces
        
#         self.pos_int.dot = pos_e
#         self.ang_int.dot = ang_e
        
#         controller_info = {
#             "angd": angd,
#             "ang": ang,
#             "posd": posd,
#         }
        
#         return ctrls, controller_info