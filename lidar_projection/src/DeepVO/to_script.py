import torch
import torch.nn as nn
from model import DeepVO

if __name__=='__main__':
    model = DeepVO(batchNorm=True)
    model.load_state_dict(torch.load('/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/Epoch:100, Loss: 0.04158163006449568, err: 0.177788572193219750.008426715025742507_DeepVO.pth'))
    model.eval()

    scripted_model = torch.jit.script(model)
    scripted_model.save('DeepVO_Scripted.pt')