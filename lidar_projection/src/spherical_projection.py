#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

import cv2
import numpy as np
from tools import *

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

class lidar_projection:
    def __init__(self, lidar_model):
        # Start the node, load parameters
        rospy.init_node('projector', anonymous=True)
        self.lidar_model = lidar_model
        self.get_params(self.lidar_model)
        self.sub = rospy.Subscriber(self.point_cloud_topic, PointCloud2, self.lidar_callback)
        self.rate = rospy.Rate(10)

        # callback variables
        self.pcd = None

    def lidar_callback(self, msg):
        self.pcd = msg
        
    def get_params(self, lidar_model):
        self.point_cloud_topic = rospy.get_param('point_cloud_topic')
        self.imu_topic = rospy.get_param('imu_topic')

        if lidar_model == "VLP16":
            vlp16 = rospy.get_param(lidar_model)
            self.h_res = vlp16['HRES']
            self.v_res = vlp16['VRES']
            self.h_res_rad = self.h_res * (np.pi / 180.0)
            self.v_res_rad = self.v_res * (np.pi / 180.0)
            self.v_fov = vlp16['VFOV']
            self.v_fov_total = vlp16['VFOV_TOTAL']
            self.lidar_range = vlp16['RANGE']
            self.y_fudge = vlp16['YFUDGE']
            self.z_fudge = vlp16['ZFUDGE']

        elif lidar_model == "HDL64E":
            hdl64e = rospy.get_param(lidar_model)
            self.h_res = hdl64e['HRES']
            self.v_res = hdl64e['VRES']
            self.h_res_rad = self.h_res * (np.pi / 180.0)
            self.v_res_rad = self.v_res * (np.pi / 180.0)
            self.v_fov = hdl64e['VFOV']
            self.v_fov_total = hdl64e['VFOV_TOTAL']
            self.lidar_range = hdl64e['RANGE']
            self.y_fudge = hdl64e['YFUDGE']

    @ measure_execution_time
    def pointCloud2ParnomaicView(self, msg):
        # Read points (x, y, z), if not transformed need to transform coordinate check VLP16 coordinate system
        # if VLP16, image size is (15, 900)
        pcd = np.array(list(pc2.read_points(msg, field_names=(
            'x', 'y', 'z'), skip_nans=True)))

        x_points = pcd[:, 0]
        y_points = pcd[:, 1]
        z_points = pcd[:, 2]
        r_points = np.sqrt(x_points ** 2 + y_points ** 2)

        # MAPPING TO CYLINDER
        x_img = np.arctan2(y_points, x_points) / self.h_res_rad
        y_img = -np.arctan2(z_points, r_points) / self.v_res_rad

        # THEORETICAL MAX HEIGHT FOR IMAGE
        d_plane = (self.v_fov_total / self.v_res) / \
            (self.v_fov_total * (np.pi / 180))
        h_below = d_plane * np.tan(-self.v_fov[0] * (np.pi / 180))
        h_above = d_plane * np.tan(self.v_fov[1] * (np.pi / 180))

        # SHIFT COORDINATES TO MAKE 0,0 THE MINIMUM
        x_min = -(360.0 / self.h_res) / 2
        x_max = int(360.0 / self.h_res)
        x_img = np.trunc(x_img - x_min).astype(np.int32)

        y_min = -((self.v_fov[1] / self.v_res) + self.y_fudge)
        y_max = int(np.ceil(h_below + h_above + self.y_fudge))
        y_img = np.trunc(y_img - y_min).astype(np.int32)

        # CLIP DISTANCES
        r_points = np.clip(r_points, a_min=0, a_max=self.lidar_range)

        # CONVERT TO IMAGE ARRAY
        img = np.zeros((y_max + 1, x_max + 1, 3), dtype=np.uint8)
        
        img[y_img, x_img, 0] = normalize(
            r_points, min=0.0, max=self.lidar_range) # Channel 0: depth
        
        img[y_img, x_img, 1] = normalize(x_points, min=-self.lidar_range, max=self.lidar_range) # Channel 1: X

        img[y_img, x_img, 2] = normalize(
            y_points, min=-self.lidar_range, max=self.lidar_range) # Channel 2: Y

        return img

    @ measure_execution_time
    def stack_image(self, img1, img2):
        return np.concatenate((img1, img2), axis=2)

    def main(self, show=False, save=True):
        global cnt

        while self.pcd is None:
            rospy.loginfo('PointCloud2 is empty, wait for data...')

        rospy.loginfo('Pointcloud2 is ready, now subscribing...')

        while not rospy.is_shutdown():
            img = self.pointCloud2ParnomaicView(msg=self.pcd)

            if show:    
                cv2.imshow('Depth', img[:, :, 0])
                cv2.imshow('x', img[:, :, 1])
                cv2.imshow('y', img[:, :, 2])
                cv2.imshow('Image', img)
                cv2.waitKey(1)

            if save:
                cv2.imwrite('/home/smeet/catkin_ws/src/PointCloud-Odometry/lidar_projection/src/dataset/custom_sequence/seq00/img/seq00_{0}.jpg'.format(cnt), img)
                print(f'{cnt}/{kitti_time[0][-1]} of projection image')
                cnt += 1
                
                if(cnt > kitti_time[0][-1]):
                    rospy.signal_shutdown('End of time')
                    exit(0)

            self.rate.sleep()

if __name__ == '__main__':
    cnt = 0
    lp = lidar_projection(lidar_model="HDL64E")
    lp.main(show=False, save=True)