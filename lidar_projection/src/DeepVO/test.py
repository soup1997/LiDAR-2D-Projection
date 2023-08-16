import torch
import torch.nn as nn
from model import *
from utils.dataloader import *

def calculate_rmse(pred, gt):
    r_mse = nn.MSELoss()(pred[:, :, :3], gt[:, :, :3])
    p_mse = nn.MSELoss()(pred[:, :, 3:], gt[:, :, 3:])
    r_rmse, p_rmse = torch.sqrt(r_mse), torch.sqrt(p_mse)

    return r_rmse, p_rmse


def valid_one_epoch(valid_loader):
    model.eval()
    valid_loss = 0.0
    position_error = 0.0
    rotation_error = 0.0
    
    with torch.no_grad():
        for batch_idx, (img, gt) in enumerate(valid_loader):
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            loss = criterion(output, gt)
            valid_r_error, valid_p_error = calculate_rmse(output, gt)

            valid_loss += loss.item()
            rotation_error += valid_r_error.item()
            position_error += valid_p_error.item()

    valid_loss /= len(valid_loader)
    position_error /= len(valid_loader)
    rotation_error /= len(valid_loader)

    return valid_loss, position_error, rotation_error

if __name__=='__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = Criterion(100)
    model = DeepVO(batchNorm=True).to(device)
    model.load_state_dict(torch.load('/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO/trained/DeepVO.pth'))
    model.eval()

    dataset_dir = '/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO/dataset/'
    dataset = DataLoader(ImageSequenceDataset(dataset_dir=dataset_dir, kitti_sequence='00', seq_len=7), batch_size=1, shuffle=False)

    poses = []
    valid_loss = 0.0
    position_error = 0.0
    rotation_error = 0.0

    with torch.no_grad():
        train_loader, valid_loader, test_loader = load_dataset(dataset_dir, batch_size=8, shuffle=True)
        for img, gt in train_loader:
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            loss = criterion(output, gt)
            valid_r_error, valid_p_error = calculate_rmse(output, gt)

            valid_loss += loss.item()
            rotation_error += valid_r_error.item()
            position_error += valid_p_error.item()


        valid_loss /= len(train_loader)
        position_error /= len(train_loader)
        rotation_error /= len(train_loader)
        print(valid_loss, position_error, rotation_error)