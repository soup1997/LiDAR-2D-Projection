import torch
import torch.nn as nn
from dataloader import *
from PointFlow.models.FlowFlowNet import *
from dataloader import *
from main import *


if __name__ == '__main__':
    seq = 10
    root_dir = '/home/smeet/catkin_ws/src/PointFlow-Odometry/dataset/custom_sequence/'
    dataset = DataLoader(KittiDataset(root_dir=root_dir, sequence=seq, valid_time=kitti_time[seq]), shuffle=False)

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = PoseFlowNet
    model.load_state_dict(torch.load('/home/smeet/catkin_ws/src/PointFlow-Odometry/trained_model/SqueezeFlowNet.pth'))
    model.eval()

    model.to(device)
    criterion = Criterion().to(device)

    output_file = "/home/smeet/catkin_ws/src/PointFlow-Odometry/dataset/output/output{0:02d}.txt".format(seq)

    with torch.no_grad(), open(output_file, "w") as f:
        for img, gt in dataset:
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            pose = output.cpu().numpy().reshape(-1)
            x, y, z = pose[:3]
            roll, pitch, yaw = pose[3:]
            #print(f"Translation: (x:{x:.6e} y:{y:.6e} z:{z:.6e}) | Orientation: (qw:{qw:.6e} qx:{qx:.6e} qy:{qy:.6e} qz:{qz:.6e})")
            f.write(f"{x:.6e} {y:.6e} {z:.6e} {roll:.6e} {pitch:.6e} {yaw:.6e}\n")
