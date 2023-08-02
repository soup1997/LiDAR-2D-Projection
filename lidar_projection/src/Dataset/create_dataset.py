import os
import glob
import numpy as np
from torchvision import transforms
from PIL import Image
from natsort import natsort
import math

# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

KITTI_SEQUENCE = {0: [0, 4540],
                  1: [0, 1100],
                  2: [0, 4660],
                  4: [0, 270],
                  5: [0, 2760],
                  6: [0, 1100],
                  7: [0, 1100],
                  8: [1100, 5170],
                  9: [0, 1590],
                  10: [0, 1200]}


def isRotationMatrix(R):
    R_inv = R.T
    I = np.identity(3, dtype=R.dtype)

    check_identity = np.dot(R_inv, R)
    norm = np.linalg.norm(I - check_identity)

    return norm < 1e-6


def euler_from_matrix(R):
    i = 2
    j = 0
    k = 1
    repetition = 0
    frame = 1
    parity = 0

    M = np.array(R, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] * M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    
    if frame:
        ax, az = az, ax

    return ax, ay, az

def normalize_angle(angle):
    if(angle > np.pi):
        angle = angle - 2 * np.pi

    elif(angle < -np.pi):
        angle = 2 * np.pi + angle

    return angle

def get_local_pose(global_pose):
    local_poses = []
    for i in range(1, global_pose.shape[0]):
        current = global_pose[i]
        previous = global_pose[i-1]
        
        relative_motion = (current - previous).tolist()
        local_poses.append(relative_motion)

    local_poses = np.array(local_poses)
    
    return local_poses


def R2euler(H):
    H = np.array(H).reshape(3, 4)
    t = H[:, -1]
    R = H[:, :3]

    assert (isRotationMatrix(R))

    x, y, z = t[0], t[1], t[2]
    roll, pitch, yaw = euler_from_matrix(R)
    roll, pitch, yaw = normalize_angle(roll), normalize_angle(pitch), normalize_angle(yaw)

    pose = [x, y, z, roll, pitch, yaw]

    return pose


# transform poseGT [R|t] to [x, y, z, roll, pitch, yaw]
def create_pose_data(gt_path='/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/Dataset/Groundtruth/', local=True):
    gt_files = natsort.natsorted(os.listdir(gt_path))

    for gt in gt_files:
        if gt.endswith('.txt'):
            with open(gt_path + gt, 'r') as f:
                lines = [line.split('\n')[0] for line in f.readlines()]
                global_pose = np.array([R2euler([float(value) for value in line.split(' ')]) for line in lines])

                if local:
                    local_poses = get_local_pose(global_pose)
                    np.savetxt(gt_path + 'relative' + gt, local_poses)

                else:
                    np.savetxt(gt_path+'euler'+gt, global_pose)

if __name__ == '__main__':
    create_pose_data(local=False)
