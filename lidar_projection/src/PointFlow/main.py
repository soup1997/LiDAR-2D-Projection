#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from dataloader import *
from models.PointflowNet import *

import torchsummary
from tqdm import tqdm

# define model hyperparmeters
hyperparams = {'Epoch': 50,
               'lr': 1e-5,
               'betas': [0.9, 0.999],
               'wd': 0.3,
               'batch_size': 128}


def calculate_rmse(predictions, targets):
    t_mse = nn.MSELoss()(predictions[:3], targets[:3])

    q_pred = predictions[3:]
    q_magnitude = torch.norm(q_pred)
    q = torch.div(q_pred, q_magnitude)

    q_mse = nn.MSELoss()(q, targets[3:])
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
        output = model(img)  # output is (x, y, z, qw, qx, qy, qz)
        loss = criterion(output, gt, model.st, model.sq)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_t_acc, train_q_acc = calculate_rmse(output, gt)

        train_loss += loss.item()
        translation_acc += train_t_acc.item()
        orientation_acc += train_q_acc.item()

        progress_bar.set_description(f'Epoch {epoch}/{num_epochs}, Train Loss: {train_loss / (batch_idx + 1):.4f}, Train translation acc: {train_t_acc:.4f}, Train orientation acc: {train_q_acc:.4f}')

    print(f"Epoch: {epoch}, Train loss: {train_loss / len(train_loader):.4f}, Train translation acc: {train_t_acc:.4f}, Train orientation acc: {train_q_acc:.4f}")

    train_loss /= len(train_loader)
    translation_acc /= len(train_loader)
    orientation_acc /= len(train_loader)

    progress_bar.close()
    return train_loss, translation_acc, orientation_acc


def valid_epoch(valid_loader):
    model.eval()
    valid_loss = 0.0
    translation_acc = 0.0
    orientation_acc = 0.0

    with torch.no_grad():
        for batch_idx, (img, gt) in enumerate(valid_loader):
            img, gt = img.to(device), gt.to(device)
            output = model(img)
            loss = criterion(output, gt, model.st, model.sq)
            valid_t_acc, valid_q_acc = calculate_rmse(output, gt)

            valid_loss += loss.item()
            translation_acc += valid_t_acc.item()
            orientation_acc += valid_q_acc.item()

    print(f"Epoch: {epoch}, Valid Loss: {valid_loss/ len(valid_loader):.4f}, Valid translation acc: {valid_t_acc:.4f}, Valid orientaion acc: {valid_q_acc:.4f}")

    valid_loss /= len(valid_loader)
    translation_acc /= len(valid_loader)
    orientation_acc /= len(valid_loader)

    return valid_loss, translation_acc, orientation_acc


if __name__ == '__main__':
    root_dir = '/home/smeet/catkin_ws/src/PointFlow-Odometry/dataset/custom_sequence/'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PointflowNet(init_st=-1.250, init_sq=0.500).to(device)
    criterion = Criterion().to(device)
    optimizer = optim.Adam(model.parameters(),
                           betas=hyperparams['betas'],
                           lr=hyperparams['lr'],
                           weight_decay=hyperparams['wd'])


    num_epochs = hyperparams['Epoch']
    lr_scheduler = StepLR(optimizer, step_size=10, gamma=0.75)
    train_loader, valid_loader, valid_loader = load_dataset(root_dir=root_dir, batch_size=hyperparams['batch_size'])

    writer = SummaryWriter()
    torchsummary.summary(model, input_size=(6, 64, 1024))

    for epoch in range(1, num_epochs + 1):
        train_loss, train_t_acc, train_q_acc = train_one_epoch(epoch, train_loader)
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Translation Acc/Train", train_t_acc, epoch)
        writer.add_scalar("Orientation Acc/Train", train_q_acc, epoch)

        if epoch % 10 == 0:
            lr_scheduler.step()  # apply StepLR every 10 epochs
            valid_loss, valid_t_acc, valid_q_acc = valid_epoch(valid_loader)
            writer.add_scalar("Loss/Valid", valid_loss, epoch)
            writer.add_scalar("Translation Acc/Valid", valid_t_acc, epoch)
            writer.add_scalar("Orientation Acc/Valid", valid_q_acc, epoch)

    # After training, save the model
    model.to('cpu')
    model.eval()

    torch.save(model.state_dict(), "/home/smeet/catkin_ws/src/PointFlow-Odometry/trained_model/PointFlow2_model_final.pth")

    # Convert the model to torch.jit.script to load in cpp
    model_scripted = torch.jit.script(model)
    model_scripted.save("/home/smeet/catkin_ws/src/PointFlow-Odometry/trained_model/PointFlow2_model_scripted.pt")
    writer.close()