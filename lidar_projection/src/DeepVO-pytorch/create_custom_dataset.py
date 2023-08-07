import numpy as np
import math
from PIL import Image
import torch
from torchvision import transforms
from natsort import natsorted
import os

def isRotationMatrix(R):
    R_inv = R.T
    shouldbeIdentity = np.dot(R_inv, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldbeIdentity)

    return n < 1e-6


def normalize_angle(angle):
    if angle > np.pi:
        angle = angle - 2*np.pi

    elif angle < -np.pi:
        angle = 2*np.pi + angle

    return angle


def R_to_angle(R):
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if not singular:
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    
    else:
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0
 
    return np.array([x, y, z])


def create_poses(gt_dir):
    for kitti_seq in kitti_time.keys():
        GT = []
        fn = f'{kitti_seq:02d}.txt'
        with open(gt_dir + fn) as f:
            for line in f:
                line = line.split(' ')
                line = [float(pose) for pose in line]
                poses = np.append(np.array([line[3],line[7],line[11]]), R_to_angle(np.array([line[0:3],line[4:7],line[8:11]])))
                GT.append(poses)
            np.savetxt(gt_dir + f'euler_gt{kitti_seq:02d}.txt', GT)

def local2global(gt_dir):
    for kitti_seq in kitti_time.keys():
        local_gt = []
        poses_array = np.loadtxt(gt_dir + f'euler_gt{kitti_seq:02d}.txt')
        
        for line in range(poses_array.shape[0]-1):

            if line  == 0:
                local_gt.append(poses_array[line])

            else:
                local_pose = poses_array[line + 1] - poses_array[line]
                local_gt.append(local_pose)
        
        local_gt = np.array(local_gt)
        np.savetxt(gt_dir + f'local_gt{kitti_seq:02d}.txt', local_gt)        

def calculate_rgb_mean_std(image_dir):
    total_cnt_pixels = 0
    to_tensor = transforms.ToTensor()

    num_sequences = len(kitti_time.keys())

    for kitti_seq in kitti_time.keys():
        seq_dir = os.path.join(image_dir, f'seq{kitti_seq:02d}', 'img')
        image_files = natsorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])[kitti_time[kitti_seq][0]: kitti_time[kitti_seq][1] + 1]
        n_images = len(image_files)
        print(f'Numbers of frames in training dataset for seq{kitti_seq:02d}: {n_images}')

        mean_np = [0, 0, 0]
        mean_tensor = [0, 0, 0]

        for idx, img in enumerate(image_files):
            print(f'{idx + 1} / {n_images}', end='\r')
            img_as_img = Image.open(os.path.join(seq_dir, img))
            img_as_tensor = to_tensor(img_as_img)

            img_as_np = np.array(img_as_img)
            img_as_np = np.rollaxis(img_as_np, axis=2, start=0)

            cnt_pixels = img_as_np.shape[1] * img_as_np.shape[2]
            total_cnt_pixels += cnt_pixels

            for c in range(3):
                mean_tensor[c] += float(torch.sum(img_as_tensor[c]))
                mean_np[c] += float(np.sum(img_as_np[c]))

        mean_tensor = [v / total_cnt_pixels for v in mean_tensor]
        mean_np = [v / total_cnt_pixels for v in mean_np]
        print('mean_tensor = ', mean_tensor)
        print('mean_np = ', mean_np)

        std_tensor = [0, 0, 0]
        std_np = [0, 0, 0]

        for idx, img in enumerate(image_files):
            print(f'{idx + 1} / {n_images}', end='\r')
            img_as_img = Image.open(os.path.join(seq_dir, img))
            img_as_tensor = to_tensor(img_as_img)
            img_as_np = np.array(img_as_img)
            img_as_np = np.rollaxis(img_as_np, 2, 0)
            for c in range(3):
                tmp = (img_as_tensor[c] - mean_tensor[c])**2
                std_tensor[c] += float(torch.sum(tmp))
                tmp = (img_as_np[c] - mean_np[c])**2
                std_np[c] += float(np.sum(tmp))

    std_tensor = [math.sqrt(v / total_cnt_pixels) for v in std_tensor]
    std_np = [math.sqrt(v / total_cnt_pixels) for v in std_np]

    print('std_tensor = ', std_tensor)
    print('std_np = ', std_np)

if __name__ == '__main__':
    gt_dir = '/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/Dataset/ground_truth/'
    image_dir = '/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/Dataset/custom_sequence/'
    
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
    
    create_poses(gt_dir=gt_dir)
    local2global(gt_dir=gt_dir)
    # calculate_rgb_mean_std(image_dir=image_dir)