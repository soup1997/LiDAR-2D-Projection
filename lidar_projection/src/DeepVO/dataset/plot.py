import matplotlib.pyplot as plt
import numpy as np
from create_dataset import *


gt_path = '/home/smeet/catkin_ws/src/LiDAR-Inertial-Odometry/lidar_projection/src/Dataset/Groundtruth/'
gt_sequence = {'00': (gt_path + 'euler00.txt'),
               '01': (gt_path + 'euler01.txt'),
               '02': (gt_path + 'euler02.txt'),
               '04': (gt_path + 'euler04.txt'),
               '05': (gt_path + 'euler05.txt'),
               '06': (gt_path + 'euler06.txt'),
               '07': (gt_path + 'euler07.txt'),
               '08': (gt_path + 'euler08.txt'),
               '09': (gt_path + 'euler09.txt'),
               '10': (gt_path + 'euler10.txt')}

rel_sequence = {'00': (gt_path + 'relative00.txt'),
                '01': (gt_path + 'relative01.txt'),
                '02': (gt_path + 'relative02.txt'),
                '04': (gt_path + 'relative04.txt'),
                '05': (gt_path + 'relative05.txt'),
                '06': (gt_path + 'relative06.txt'),
                '07': (gt_path + 'relative07.txt'),
                '08': (gt_path + 'relative08.txt'),
                '09': (gt_path + 'relative09.txt'),
                '10': (gt_path + 'relative10.txt')}

sequence = '00'
compare = [gt_sequence[sequence], rel_sequence[sequence]]

if __name__ == '__main__':
    gt_data = np.loadtxt(compare[0])
    custom_data = np.loadtxt(compare[1])

    fig1 = plt.figure(0)

    x = gt_data[:, 0]
    y = gt_data[:, 1]
    z = gt_data[:, 2]
    roll = gt_data[:, 3]
    pitch = gt_data[:, 4]
    yaw = gt_data[:, 5]

    xc = np.cumsum(custom_data[:, 0])
    yc = np.cumsum(custom_data[:, 1])
    zc = np.cumsum(custom_data[:, 2])
    rollc = np.cumsum(custom_data[:, 3])
    pitchc = np.cumsum(custom_data[:, 4])
    yawc = np.cumsum(custom_data[:, 5])

    ax = fig1.add_subplot(111)
    ax.set_title('sequence' + sequence)
    ax.scatter(x, y, c='k', marker='o', s=10, label='Ground Truth')
    ax.scatter(xc, yc, c='g', marker='v', s=3, label='Custom Ground Truth')
    ax.scatter(x[0], y[0], c='r', marker='o', s=50, label='Start')
    ax.scatter(x[-1], y[-1], c='b', marker='o', s=50, label='End')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()

    fig2 = plt.figure(1)
    ax2 = fig2.add_subplot(311)
    ax2.plot(roll, 'r', linewidth = 1, label='Roll')
    ax2.plot(rollc, 'b', linewidth = 0.5, label='Roll')
    ax2.legend()

    ax3 = fig2.add_subplot(312)
    ax3.plot(pitch, 'r', linewidth = 1, label='Pitch')
    ax3.plot(pitchc, 'b', linewidth = 0.5,label='Pitch')
    ax3.legend()

    ax4 = fig2.add_subplot(313)
    ax4.plot(yaw, 'r', linewidth = 1, label='Yaw')
    ax4.plot(yawc, 'b', linewidth = 0.5, label='Yaw')
    ax4.legend()
    

    plt.show()

