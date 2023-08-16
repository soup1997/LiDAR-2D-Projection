import numpy as np
import matplotlib.pyplot as plt
from utils.util import eular_to_SO3

gt_data = np.load('/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO/dataset/poses/10.npy')
output_data = np.loadtxt('/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/DeepVO/10_ouput.txt')

H_new = np.identity(4)
H_new2 = np.identity(4)

gt_list = []
output_list = []

for i in range(gt_data.shape[0]):
    H_rel = np.identity(4)
    poses = gt_data[i, :]
    rotation = poses[:3]
    rot_mat = eular_to_SO3(rotation)
    translation = poses[3:]
    H_rel[:3, :3] = rot_mat
    H_rel[:3, 3] = translation

    H_new = np.dot(H_new, H_rel)

    position = H_new[:3, 3]
    gt_list.append(position)

gt_list = np.array(gt_list)

print(output_data.shape[0])
for i in range(output_data.shape[0]):
    H_rel = np.identity(4)
    poses = output_data[i, :]
    rotation = poses[:3]
    rot_mat = eular_to_SO3(rotation)
    translation = poses[3:]
    H_rel[:3, :3] = rot_mat
    H_rel[:3, 3] = translation

    H_new2 = np.dot(H_new2, H_rel)

    position = H_new2[:3, 3]
    output_list.append(position)

output_list = np.array(output_list)

plt.figure(0)
plt.plot(gt_list[:, 0], gt_list[:, 2], 'r')
plt.plot(output_list[:, 0], output_list[:, 2], 'b')

plt.show()
