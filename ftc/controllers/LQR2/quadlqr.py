import fym
import numpy as np
import control 
from fym.utils.rot import quat2angle

class QUADLQRController(fym.BaseEnv):
    def __init__(self, env):
        super().__init__()
        m, g, Jinv = env.plant.m, env.plant.g, env.plant.Jinv
        # self.trim_forces = np.vstack([m * g, 0, 0, 0])
        
        Aatt = np.array(
            [
                [0, 0, 0, 1, 0, 0],                
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        Batt = np.vstack((np.zeros((3,3)), np.identity(3)))
        Qatt = 10*np.diag((1, 1, 1, 1, 1, 1))
        Ratt = np.diag((1, 1, 1))
        
        Apos = np.array(
            [
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0],
            ]
        )
        Bpos = np.vstack((np.zeros((3,3)), np.identity(3)))
        Qpos = 10*np.diag((1, 1, 1, 1, 1, 1))
        Rpos = 100*np.diag((1, 1, 1))
        
        self.Katt, *_ = control.lqr(Aatt, Batt, Qatt, Ratt)
        self.Kpos, *_ = control.lqr(Apos, Bpos, Qpos, Rpos)         
        

    def get_control(self, t, env):
        pos, vel, quat, omega = env.plant.observe_list()
        ang = np.vstack(quat2angle(quat)[::-1])
        
        posd = np.deg2rad(np.vstack((1, 2, 0)))
        veld = np.zeros((3,1))
        omegad = np.zeros((3,1))
        angd = np.deg2rad(np.vstack((0, 0, 0)))
        
        q0 = quat[0]
        qbar = quat[1::]    
        theta = 2 * qbar / np.linalg.norm(qbar+1e-10) * np.arccos(q0)
            
        xpos = np.vstack((pos, vel))
        xposd = np.vstack((posd, veld))
        upos=-self.Kpos @ (xpos - xposd)      
        up=upos-np.vstack((0,0,9.81))
        
        xatt = np.vstack((theta, omega))
        b = np.vstack((0, 0, -1))
        
        qprimed = np.vstack((b.T @ up + np.linalg.norm(up), np.cross(b, up, axis=0)))
        qd = qprimed / np.linalg.norm(qprimed)
        
        xattd = np.vstack((2*(qd[1::]/np.linalg.norm(qd[1::]))*np.arccos(qd[0]), omegad))
        uatt=-self.Katt @ (xatt-xattd)    
       

        torques = uatt
        forces = np.vstack((np.linalg.norm(up)*env.plant.m, torques))
        # forces = np.vstack((env.plant.m*9.81, torques))

        ctrl = np.linalg.pinv(env.plant.mixer.B) @ forces
        
        controller_info = {
            "angd": angd,
            "ang": ang,
            "posd": posd,
        }
        
        return ctrl, controller_info
        
              


if __name__ == "__main__":
    controller = QUADLQRController()
    
    
    
    
    
    
    
#  def get_control(self, t, env):
#         pos, vel, quat, omega = env.plant.observe_list()
#         ang = np.vstack(quat2angle(quat)[::-1])
        
#         posd = np.deg2rad(np.vstack((1, 2, 0)))
#         veld = np.zeros((3,1))
#         omegad = np.zeros((3,1))
#         angd = np.deg2rad(np.vstack((0, 0, 0)))
        
#         q0 = quat[0]
#         qbar = quat[1::]    
#         theta = 2 * qbar / np.linalg.norm(qbar+1e-10) * np.arccos(q0)
            
#         xpos = np.vstack((pos, vel))
#         xposd = np.vstack((posd, veld))
#         upos=-self.Kpos @ (xpos - xposd)      
#         up=upos-np.vstack((0,0,9.81))
        
#         xatt = np.vstack((theta, omega))
#         b = np.vstack((0, 0, -1))
        
#         qprimed = np.vstack((b.T @ up + np.linalg.norm(up), np.cross(b, up, axis=0)))
#         qd = qprimed / np.linalg.norm(qprimed)
        
#         xattd = np.vstack((2*(qd[1::]/np.linalg.norm(qd[1::]))*np.arccos(qd[0]), omegad))
#         uatt=-self.Katt @ (xatt-xattd)    
       

#         torques = uatt
#         forces = np.vstack((np.linalg.norm(up)*env.plant.m, torques))
#         # forces = np.vstack((env.plant.m*9.81, torques))

#         ctrl = np.linalg.pinv(env.plant.mixer.B) @ forces
        
#         controller_info = {
#             "angd": angd,
#             "ang": ang,
#             "posd": posd,
#         }
        
#         return ctrl, controller_info
        
              

# if __name__ == "__main__":
#     controller = QUADLQRController()
        
    
    
    
    
 