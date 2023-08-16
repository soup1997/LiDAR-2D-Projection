import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR
from model import DeepVO, Criterion
from utils.dataloader import *

# define model hyperparmeters
hyperparams = {'epochs': 200,
               'lr': 1e-3,
               'batch_size': 8,
               'weight_decay': 5e-6,
               'lr_step': 50,
               'lr_decay_factor':0.1}

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

    progress_bar = tqdm(valid_loader, total=len(valid_loader), desc=f'Epoch {epoch}/{num_epochs}, Valid Loss: 0.0000')
    
    with torch.no_grad():
        for batch_idx, (img, gt) in enumerate(progress_bar):
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            loss = criterion(output, gt)
            valid_r_error, valid_p_error = calculate_rmse(output, gt)

            valid_loss += loss.item()
            rotation_error += valid_r_error.item()
            position_error += valid_p_error.item()

            progress_bar.set_description(f'Epoch {epoch}/{num_epochs}, Valid Loss: {valid_loss / (batch_idx + 1):.5f}, Valid position error: {position_error / (batch_idx + 1):.5f}, Valid rotation error: {rotation_error / (batch_idx + 1):.5f}')

    valid_loss /= len(valid_loader)
    position_error /= len(valid_loader)
    rotation_error /= len(valid_loader)
    progress_bar.close()

    return valid_loss, position_error, rotation_error

def train_one_epoch(epoch, train_loader):
    train_loss = 0.0
    position_error = 0.0
    rotation_error = 0.0

    model.train()
    progress_bar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs}, Train Loss: 0.0000')

    for batch_idx, (img, gt) in enumerate(progress_bar):
        img, gt = img.to(device), gt.to(device)
        output = model(img)  # output is (roll, pitch, yaw, x, y, z)
        loss = criterion(output, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_r_error, train_p_error = calculate_rmse(output, gt)

        train_loss += loss.item()
        rotation_error += train_r_error.item()
        position_error += train_p_error.item()

        progress_bar.set_description(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss / (batch_idx + 1):.5f}, Train position error: {position_error / (batch_idx + 1):.5f}, Train rotation error: {rotation_error / (batch_idx + 1):.5f}')

    train_loss /= len(train_loader)
    position_error /= len(train_loader)
    rotation_error /= len(train_loader)
    progress_bar.close()

    return train_loss, position_error, rotation_error


if __name__=='__main__':
    dataset_dir = '/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO/dataset/'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Create the model
    model = DeepVO(batchNorm=True)
    model.load_Flownet()
    model = model.to(device)

    criterion = Criterion(k=100.0).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'], betas=(0.9, 0.999), weight_decay=hyperparams['weight_decay'])
    scheduler = StepLR(optimizer, gamma=hyperparams['lr_decay_factor'], step_size=hyperparams['lr_step'])
    num_epochs = hyperparams['epochs']
    
    writer = SummaryWriter()
    torch.set_printoptions(sci_mode=False, precision=10)
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(1, num_epochs+1):
        train_loader, valid_loader, _ = load_dataset(dataset_dir, batch_size=hyperparams['batch_size'], shuffle=True)
        train_loss, train_t_err, train_q_err = train_one_epoch(epoch, train_loader)

        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Translation Error/Train", train_t_err, epoch)
        writer.add_scalar("Orientation Error/Train", train_q_err, epoch)

        valid_loss, valid_t_err, valid_q_err = valid_one_epoch(valid_loader)
        writer.add_scalar("Loss/Valid", valid_loss, epoch)
        writer.add_scalar("Translation Error/Valid", valid_t_err, epoch)
        writer.add_scalar("Orientation Error/Valid", valid_q_err, epoch)

        if epoch % hyperparams['lr_step'] == 0:
            scheduler.step()
            model.eval()
            torch.save(model.state_dict(), f"Error:{valid_t_err},{valid_q_err}_DeepVO.pth")

    writer.close()