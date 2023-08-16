import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchvision.transforms as transforms

import os
import numpy as np
from PIL import Image
from natsort import natsorted

kitti_dataset = {'00': [0, 4540],
                 '01': [0, 1100],
                 '02': [0, 4660],
                 '04': [0, 270],
                 '05': [0, 2760],
                 '06': [0, 1100],
                 '07': [0, 1100],
                 '08': [1100, 5170],
                 '09': [0, 1590],
                 '10': [0, 1200]}

# Generate the data frames per path
def get_data_info(dataset_dir, kitti_sequence, seq_len=7, overlap=1, drop_last=True):
    pose_dir = os.path.join(dataset_dir, 'poses')
    image_dir = os.path.join(dataset_dir, 'images')

    img_sequence, pose_sequence = [], []

    # Load & sort the raw data
    poses = np.load(os.path.join(pose_dir, f'{kitti_sequence}.npy'))  # (n_images, 6)
    images = natsorted([f for f in os.listdir(os.path.join(image_dir, kitti_sequence)) if f.endswith('.jpg')])

    n_frames = len(images)
    start = 0
    
    while (start + seq_len) < n_frames:
        x_seg = images[start:start+seq_len]
        img_sequence.append([os.path.join(image_dir, kitti_sequence, img) for img in x_seg])
        pose_sequence.append(poses[start:start+seq_len-1])
        start += seq_len - overlap

    if not drop_last:
        img_sequence.append([os.path.join(image_dir, kitti_sequence, img) for img in images[start:]])
        pose_sequence.append(poses[start:])

    # Store in a dictionary
    data = {'image_path': img_sequence, 'pose': pose_sequence}
    return data

def load_dataset(dataset_dir, batch_size=64, shuffle=True):
    train_datasets = []
    valid_datasets = []
    test_datasets = []

    for seq, valid_time in kitti_dataset.items():
        dataset = ImageSequenceDataset(dataset_dir=dataset_dir, kitti_sequence=seq, seq_len=7)

        if seq== '04' or seq == '06':
            valid_datasets.append(dataset)

        elif seq == '09' or seq == '10':
            test_datasets.append(dataset)

        else:
            train_datasets.append(dataset)

    train_loader = DataLoader(dataset=ConcatDataset(train_datasets),
                              batch_size=batch_size,
                              shuffle=shuffle)

    valid_loader = DataLoader(dataset=ConcatDataset(valid_datasets),
                             batch_size=batch_size,
                             shuffle=shuffle)
    
    test_loader = DataLoader(dataset=ConcatDataset(test_datasets),
                             batch_size=batch_size,
                             shuffle=shuffle)

    return train_loader, valid_loader, test_loader

class SubtractConstant:
    def __init__(self, constant=0.5):
        self.constant = constant

    def __call__(self, img):
        return (img - self.constant)


class ImageSequenceDataset(Dataset):
    def __init__(self, dataset_dir, kitti_sequence, seq_len=7, overlap=1, drop_last=True):
        self.data_info = get_data_info(dataset_dir, kitti_sequence, seq_len, overlap, drop_last)
        self.image = self.data_info['image_path']
        self.groundtruth = self.data_info['pose']
        self.transforms = transforms.Compose([transforms.ToTensor(),
                                              transforms.Resize((64, 1200)),
                                              SubtractConstant(constant=0.5)])

    def __len__(self):
        return len(self.data_info['image_path'])

    def __getitem__(self, index):
        image_sequence = []
        image_path_sequence = self.image[index]

        for img_path in image_path_sequence:
            img_as_img = Image.open(img_path)
            img_as_tensor = self.transforms(img_as_img).unsqueeze(0) # torch.Size([1, 3, 64, 1200]), which is (1, channel, height, width)
            image_sequence.append(img_as_tensor)

        image_sequence = torch.cat(image_sequence, 0) # torch.Size([7, 3, 64, 1200]), which is (sequence length, channel, height, width)

        # Prepare the ground truth pose
        gt_sequence = self.groundtruth[index]
        gt_sequence = torch.tensor(gt_sequence, dtype=torch.float32) # torch.Size([6, 6]), which is (sequence length, 6dof)

        return image_sequence, gt_sequence

if __name__=='__main__':
    dataset_dir = '/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO/dataset/'
    train_loader, valid_loader, test_loader = load_dataset(dataset_dir, batch_size=8, shuffle=True)

    for img, gt in train_loader:
        print(img.size(), gt.size()) # torch.Size([8, 7, 3, 64, 1200]) torch.Size([8, 6, 6])