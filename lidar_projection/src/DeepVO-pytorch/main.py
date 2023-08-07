#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from dataloader import *
from model import *
from torchinfo import summary
from tqdm import tqdm

# define model hyperparmeters
hyperparams = {'Epoch': 250,
               'lr': 0.0005,
               'batch_size': 8}


def load_pretrained_flownet():
    global model

    print('Load pretrained FlowNetS weight')
    
    pretrained_flownet = '/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO-pytorch/trained_model/flownets_EPE1.951.pth.tar'
    pretrained_weight = torch.load(pretrained_flownet)

    model_dict = model.state_dict()
    update_dict = {k: v for k, v in pretrained_weight['state_dict'].items() if k in model_dict}
    model_dict.update(update_dict)
    model.load_state_dict(model_dict)


def criterion(pred, gt):
    p_hat, p = pred[:, :3], gt[:, :3]  # translation
    q_hat, q = pred[:, 3:], gt[:, 3:]  # orientation(euler)

    p_error = nn.MSELoss()(p, p_hat)
    q_error = nn.MSELoss()(q, q_hat)

    loss = (p_error) + (100.0 * q_error)

    return loss

def calculate_rmse(predictions, targets):
    t_mse = nn.MSELoss()(predictions[:, :3], targets[:, :3])
    q_mse = nn.MSELoss()(predictions[:, 3:], targets[:, 3:])
    t_rmse, q_rmse = torch.sqrt(t_mse), torch.sqrt(q_mse)
    return t_rmse, q_rmse


def train_one_epoch(epoch, train_loader):
    train_loss = 0.0
    translation_error = 0.0
    orientation_error = 0.0

    model.train()
    progress_bar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs}, Train Loss: 0.0000')

    for batch_idx, (img, gt) in enumerate(progress_bar):
        img, gt = img.to(device), gt.to(device)
        output = model(img)  # output is (x, y, z, roll, pitch, yaw)
        loss = criterion(output, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_t_error, train_q_error = calculate_rmse(output, gt)

        train_loss += loss.item()
        translation_error += train_t_error.item()
        orientation_error += train_q_error.item()

        progress_bar.set_description(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss / (batch_idx + 1):.4f}, Train translation error: {translation_error / (batch_idx + 1):.4f}, Train orientation error: {orientation_error / (batch_idx + 1):.4f}')

    train_loss /= len(train_loader)
    translation_error /= len(train_loader)
    orientation_error /= len(train_loader)
    progress_bar.close()

    return train_loss, translation_error, orientation_error


def test_epoch(test_loader):
    model.eval()
    test_loss = 0.0
    translation_error = 0.0
    orientation_error = 0.0

    progress_bar = tqdm(test_loader, total=len(test_loader), desc=f'Epoch {epoch}/{num_epochs}, Test Loss: 0.0000')

    with torch.no_grad():
        for batch_idx, (img, gt) in enumerate(progress_bar):
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            loss = criterion(output, gt)
            test_t_error, test_q_error = calculate_rmse(output, gt)

            test_loss += loss.item()
            translation_error += test_t_error.item()
            orientation_error += test_q_error.item()

            progress_bar.set_description(f'Epoch {epoch}/{num_epochs}, Test Loss: {test_loss / (batch_idx + 1):.4f}, Test translation error: {translation_error / (batch_idx + 1):.4f}, Test orientation error: {orientation_error / (batch_idx + 1):.4f}')

    test_loss /= len(test_loader)
    translation_error /= len(test_loader)
    orientation_error /= len(test_loader)
    progress_bar.close()
    
    return test_loss, translation_error, orientation_error


if __name__ == '__main__':
    root_dir = '/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/Dataset/custom_sequence/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepVO(batchNorm=True).to(device)
    # load_pretrained_flownet()

    optimizer = optim.Adagrad(model.parameters(), lr=hyperparams['lr'])
    num_epochs = hyperparams['Epoch']

    writer = SummaryWriter()
    torch.set_printoptions(sci_mode=False, precision=10)
    torch.autograd.set_detect_anomaly(True)

    # summary(model, input_size=(hyperparams['batch_size'], 6, 128, 1800))
    for epoch in range(1, num_epochs + 1):
        train_loader, test_loader = load_dataset(root_dir=root_dir, batch_size=hyperparams['batch_size'], shuffle=True)
        train_loss, train_t_error, train_q_error = train_one_epoch(epoch, train_loader)
        test_loss, test_t_error, test_q_error = test_epoch(test_loader)
        
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Translation error/Train", train_t_error, epoch)
        writer.add_scalar("Orientation error/Train", train_q_error, epoch)

        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Translation error/Test", test_t_error, epoch)
        writer.add_scalar("Orientation error/Test", test_q_error, epoch)

    # After training, save the model
    model.to('cpu')
    model.eval()

    torch.save(model.state_dict(), "/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO-pytorch/trained_model/DeepVO.pth")

    # Convert the model to torch.jit.script to load in cpp
    model_scripted = torch.jit.script(model)
    model_scripted.save("/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO-pytorch/trained_model/DeepVO_Scripted.pth")
    writer.close()