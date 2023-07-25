import torch
import torch.nn as nn
from dataloader import *
from models.PointflowNet import *
from dataloader import *
from main import *


if __name__ == '__main__':
    seq = 0
    root_dir = '/home/smeet/catkin_ws/src/PointFlow-Odometry/dataset/custom_sequence/'
    dataset = DataLoader(KittiDataset(root_dir=root_dir, sequence=seq, valid_time=kitti_time[seq]), shuffle=False)

    device = "cuda" if torch.cuda.is_available() else 'cpu'
    model = PointflowNet()
    model.load_state_dict(torch.load('/home/smeet/catkin_ws/src/PointFlow-Odometry/trained_model/PointFlow2_model_final.pth'))
    model.to(device)
    model.eval()

    criterion = Criterion().to(device)
    valid_loss = 0.0
    translation_acc = 0.0
    orientation_acc = 0.0

    output_file = "/home/smeet/catkin_ws/src/PointFlow-Odometry/dataset/output/output{0:02d}.txt".format(seq)

    with torch.no_grad(), open(output_file, "w") as f:
        for batch_idx, (img, gt) in enumerate(dataset):
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            loss = criterion(output, gt, model.st, model.sq)
            valid_t_acc, valid_q_acc = calculate_rmse(output, gt)

            valid_loss += loss.item()
            translation_acc += valid_t_acc.item()
            orientation_acc += valid_q_acc.item()

             # Write the output to the file
            for i in range(output.shape[0]):  # Assuming batch size is in the first dimension
                pose = output[i].cpu().numpy()
                x, y, z = pose[:3]
                qw, qx, qy, qz = pose[3:]
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {qw:.6f} {qx:.6f} {qy:.6f} {qz:.6f}\n")
