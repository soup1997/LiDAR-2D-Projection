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
hyperparams = {'Epoch': 100,
               'lr': 1e-5,
               'batch_size': 8,
               'step_size':20,
               'gamma':0.8}


def load_pretrained_flownet():
    global model

    print('Load pretrained FlowNetS weight')
    
    pretrained_flownet = '/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO-pytorch/TrainedModel/flownets_EPE1.951.pth.tar'
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
    translation_acc = 0.0
    orientation_acc = 0.0

    model.train()
    progress_bar = tqdm(train_loader, total=len(train_loader), desc=f'Epoch {epoch}/{num_epochs}, Train Loss: 0.0000')

    for batch_idx, (img, gt) in enumerate(progress_bar):
        img, gt = img.to(device), gt.to(device)
        output = model(img)  # output is (x, y, z, roll, pitch, yaw)
        loss = criterion(output, gt)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #print(f"output: {output[0]}\ngt: {gt[0]}")

        train_t_acc, train_q_acc = calculate_rmse(output, gt)

        train_loss += loss.item()
        translation_acc += train_t_acc.item()
        orientation_acc += train_q_acc.item()

        progress_bar.set_description(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss / (batch_idx + 1):.4f}, Train translation acc: {translation_acc / (batch_idx + 1):.4f}, Train orientation acc: {orientation_acc / (batch_idx + 1):.4f}')
    
    #print(f"Epoch: {epoch}, Train loss: {train_loss / len(train_loader):.4f}, Train translation acc: {train_t_acc:.4f}, Train orientation acc: {train_q_acc:.4f}")

    train_loss /= len(train_loader)
    translation_acc /= len(train_loader)
    orientation_acc /= len(train_loader)
    progress_bar.close()

    return train_loss, translation_acc, orientation_acc


def test_epoch(test_loader):
    model.eval()
    test_loss = 0.0
    translation_acc = 0.0
    orientation_acc = 0.0

    progress_bar = tqdm(test_loader, total=len(test_loader), desc=f'Epoch {epoch}/{num_epochs}, Test Loss: 0.0000')

    with torch.no_grad():
        for batch_idx, (img, gt) in enumerate(progress_bar):
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            loss = criterion(output, gt)
            test_t_acc, test_q_acc = calculate_rmse(output, gt)

            test_loss += loss.item()
            translation_acc += test_t_acc.item()
            orientation_acc += test_q_acc.item()

            progress_bar.set_description(f'Epoch {epoch}/{num_epochs}, Test Loss: {test_loss / (batch_idx + 1):.4f}, Test translation acc: {translation_acc / (batch_idx + 1):.4f}, Test orientation acc: {orientation_acc / (batch_idx + 1):.4f}')

    test_loss /= len(test_loader)
    translation_acc /= len(test_loader)
    orientation_acc /= len(test_loader)
    progress_bar.close()
    
    return test_loss, translation_acc, orientation_acc


if __name__ == '__main__':
    root_dir = '/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/Dataset/custom_sequence/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeepVO(batchNorm=True).to(device)
    load_pretrained_flownet()

    optimizer = optim.Adagrad(model.parameters(), lr=hyperparams['lr'])
    num_epochs = hyperparams['Epoch']
    lr_scheduler = StepLR(optimizer, step_size=hyperparams['step_size'], gamma=hyperparams['gamma'])

    writer = SummaryWriter()
    summary(model, input_size=(hyperparams['batch_size'], 6, 64, 1800))
    torch.set_printoptions(sci_mode=False, precision=10)
    torch.autograd.set_detect_anomaly(True)

    for epoch in range(1, num_epochs + 1):
        train_loader, test_loader = load_dataset(root_dir=root_dir, batch_size=hyperparams['batch_size'], shuffle=True)
        train_loss, train_t_acc, train_q_acc = train_one_epoch(epoch, train_loader)
        test_loss, test_t_acc, test_q_acc = test_epoch(test_loader)
        
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Translation Acc/Train", train_t_acc, epoch)
        writer.add_scalar("Orientation Acc/Train", train_q_acc, epoch)

        writer.add_scalar("Loss/Test", test_loss, epoch)
        writer.add_scalar("Translation Acc/Test", test_t_acc, epoch)
        writer.add_scalar("Orientation Acc/Test", test_q_acc, epoch)

        
        if epoch % hyperparams['step_size'] == 0:
            lr_scheduler.step()

    # After training, save the model
    model.to('cpu')
    model.eval()

    torch.save(model.state_dict(), "/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO-pytorch/TrainedModel/DeepVO.pth")

    # Convert the model to torch.jit.script to load in cpp
    model_scripted = torch.jit.script(model)
    model_scripted.save("/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO-pytorch/TrainedModel/DeepVO_Scripted.pth")
    writer.close()