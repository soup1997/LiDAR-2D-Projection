import torch
import torch.nn as nn
from model import *
from utils.dataloader import *

def calculate_rmse(pred, gt):
    r_mse = nn.MSELoss()(pred[:, :, :3], gt[:, :, :3])
    p_mse = nn.MSELoss()(pred[:, :, 3:], gt[:, :, 3:])
    r_rmse, p_rmse = torch.sqrt(r_mse), torch.sqrt(p_mse)

    return r_rmse, p_rmse


def test_one_epoch(test_loader):
    model.eval()
    test_loss = 0.0
    position_error = 0.0
    rotation_error = 0.0
    
    with torch.no_grad():
        for batch_idx, (img, gt) in enumerate(test_loader):
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            loss = criterion(output, gt)
            test_r_error, test_p_error = calculate_rmse(output, gt)

            test_loss += loss.item()
            rotation_error += test_r_error.item()
            position_error += test_p_error.item()

    test_loss /= len(test_loader)
    position_error /= len(test_loader)
    rotation_error /= len(test_loader)

    return test_loss, position_error, rotation_error

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = Criterion(100)
    model = DeepVO(batchNorm=True).to(device)
    model.load_state_dict(torch.load('/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO/trained/DeepVO.pth'))
    model.eval()

    dataset_dir = '/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO/dataset/'
    dataset = DataLoader(ImageSequenceDataset(dataset_dir=dataset_dir, kitti_sequence='00', seq_len=7), batch_size=1, shuffle=False)

    test_loss = 0.0
    position_error = 0.0
    rotation_error = 0.0

    with torch.no_grad():
        _, _, test_loader = load_dataset(dataset_dir, batch_size=8, shuffle=True)
        for img, gt in test_loader:
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            loss = criterion(output, gt)
            test_r_error, test_p_error = calculate_rmse(output, gt)

            test_loss += loss.item()
            rotation_error += test_r_error.item()
            position_error += test_p_error.item()


        test_loss /= len(test_loader)
        position_error /= len(test_loader)
        rotation_error /= len(test_loader)
        print(test_loss, position_error, rotation_error)