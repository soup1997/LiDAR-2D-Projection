#!/usr/bin/env python3

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, ConcatDataset

import os
from PIL import Image
import numpy as np


kitti_time = {0: [0, 4540],
              1: [0, 1100],
              2: [0, 4660],
              4: [0, 270],
              5: [0, 2760],
              6: [0, 1100],
              7: [0, 1100],
              8: [1100, 5170],
              9: [0, 1590],
              10: [0, 1200]}


class KittiDataset(Dataset):
    def __init__(self, root_dir, sequence, valid_time):
        self.root_dir = root_dir
        self.sequence = sequence
        self.idx = valid_time

        # Set (img, gt) paths
        self.image_dir = os.path.join(root_dir, f'seq{sequence:02d}', 'img')
        self.pose_dir = os.path.join(
            root_dir, f'seq{sequence:02d}', f'pose_seq{sequence:02d}.txt')

        # Load (img, gt) files
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])[valid_time[0]: valid_time[1]+1]
        self.pose_file = self._load_poses()

        # According to sequence, apply different transformations
        self.transforms = {0: transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.58692193, 0.3301893, 0.0962865),
                                                                       (0.35340258, 0.21320646, 0.08324931)),
                                                  transforms.Resize((64, 1024))]),

                           1: transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.50520706, 0.2867759, 0.10206651),
                                                                       (0.36996016, 0.22401515, 0.10400321)),
                                                  transforms.Resize((64, 1024))]),

                           2: transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.6087684, 0.3437133, 0.09761499),
                                                                       (0.34740967, 0.21028088, 0.08274511)),
                                                  transforms.Resize((64, 1024))]),

                           4: transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.6039709, 0.34532478, 0.10994011),
                                                                       (0.34533167, 0.21076982, 0.09049769)),
                                                  transforms.Resize((64, 1024))]),

                           5: transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5969699, 0.3360664, 0.10014473),
                                                                       (0.34749728, 0.21047164, 0.08389255)),
                                                  transforms.Resize((64, 1024))]),

                           6: transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.58958834, 0.33824897, 0.113432385),
                                                                       (0.35417318, 0.21972811, 0.102402635)),
                                                  transforms.Resize((64, 1024))]),

                           7: transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5878722, 0.3301315, 0.09515252),
                                                                       (0.3536716, 0.21258134, 0.08227222)),
                                                  transforms.Resize((64, 1024))]),

                           8: transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5868147, 0.32986256, 0.098948255),
                                                                       (0.35147962, 0.2127681, 0.08660019)),
                                                  transforms.Resize((64, 1024))]),

                           9: transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize((0.5988516, 0.33750767, 0.10193553),
                                                                       (0.3488066, 0.21126577, 0.08636672)),
                                                  transforms.Resize((64, 1024))]),

                           10: transforms.Compose([transforms.ToTensor(),
                                                   transforms.Normalize((0.6061787, 0.3401464, 0.09553494),
                                                                        (0.34480524, 0.20893832,0.08096962)),
                                                   transforms.Resize((64, 1024))])}

    def _load_poses(self):
        pose_data = np.loadtxt(self.pose_dir)
        pose_data = torch.tensor(pose_data, dtype=torch.float32)
        return pose_data

    def _stack_image(self, img1, img2):
        stacked_img = torch.cat((img1, img2), dim=0)
        return stacked_img

    def __len__(self):  # Specifies the the size of dataset
        return len(self.image_files) - 1

    def __getitem__(self, idx):

        img1_name = self.image_files[idx]
        img2_name = self.image_files[idx+1]

        img1_path = os.path.join(self.image_dir, img1_name)
        img2_path = os.path.join(self.image_dir, img2_name)

        img1_tensor = self.transforms[self.sequence](np.array(Image.open(img1_path)))
        img2_tensor = self.transforms[self.sequence](np.array(Image.open(img2_path)))

        stacked_img = self._stack_image(img1_tensor, img2_tensor)
        ground_truth = self.pose_file[idx]

        return stacked_img, ground_truth


def calcultate_norm(dataset):
    mean_ = np.array([np.mean(x.numpy(), axis=(1, 2)) for x, _ in dataset])

    mean_r = mean_[:, 0].mean()
    mean_g = mean_[:, 1].mean()
    mean_b = mean_[:, 2].mean()

    std_ = np.array([np.std(x.numpy(), axis=(1, 2)) for x, _ in dataset])

    std_r = std_[:, 0].mean()
    std_g = std_[:, 1].mean()
    std_b = std_[:, 2].mean()

    return (mean_r, mean_g, mean_b), (std_r, std_g, std_b)


def load_dataset(root_dir, batch_size=64):
    train_datasets = []
    valid_datasets = []
    test_datasets = []

    for seq, valid_time in kitti_time.items():
        dataset = KittiDataset(root_dir=root_dir, sequence=seq, valid_time=valid_time)
        
        '''
        mean_, std_ = calcultate_norm(dataset)
        print(mean_, std_)
        '''

        if seq == 7 or seq == 8:
            valid_datasets.append(dataset)

        elif seq == 9 or seq == 10:
            test_datasets.append(dataset)

        else:
            train_datasets.append(dataset)

    train_loader = DataLoader(dataset=ConcatDataset(train_datasets),
                              batch_size=batch_size,
                              shuffle=True)

    valid_loader = DataLoader(dataset=ConcatDataset(valid_datasets),
                              batch_size=batch_size,
                              shuffle=True)

    test_loader = DataLoader(dataset=ConcatDataset(test_datasets),
                             batch_size=batch_size,
                             shuffle=True)

    return train_loader, valid_loader, test_loader


if __name__ == '__main__':
    # img.size, gt.size = torch.Size([batch_size, 6, 64, 1024]) torch.Size([batch_size, 7])

    root_dir = '/home/smeet/catkin_ws/src/PointFlow-Odometry/lidar_projection/src/dataset/custom_sequence/'
    train_loader, valid_loader, test_loader = load_dataset(root_dir=root_dir, batch_size=64)

    print("Train Loader Length:", len(train_loader))
    print("Valid Loader Length:", len(valid_loader))
    print("Test Loader Length:", len(test_loader))
