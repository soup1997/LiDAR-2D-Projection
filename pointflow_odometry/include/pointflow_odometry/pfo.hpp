#ifndef POINTFLOW_ODOMETRY_PFO_HPP
#define POINTFLOW_ODOMETRY_PFO_HPP

#include <iostream>
#include <string>
#include <queue>
#include <cmath>

#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>

#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>
#include <pcl/conversions.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <pointflow_odometry/ieskf.hpp>

class PFO{
    private:
        /*---------ROS Definition---------*/
        ros::Subscriber pcd_sub; // PointCloud2 Subscriber
        ros::Subscriber imu_sub; // Imu Subscriber
        std::string _point_cloud_topic;
        std::string _imu_topic;

        /*---------LiDAR Parameters---------*/
        double _hres, _vres;                // horizontal, vertical resolution
        double _hres_rad, _vres_rad;        // for radian representation
        double _vfov, _vfov_min, _vfov_max; // vertical fov
        double _range;                      // max range
        double _yfudge;                     // yfudge factor

        /*---------IMU Variables---------*/
        double _gyroX, _gyroY, _gyroZ;         // angular velocity x, y, z
        double _accX, _accY, _accZ;            // acceleration x, y, z
        double _prev_imu_time, _curr_imu_time; // prev time stamp, current time stamp
        double _dt;                            // elapsed time

        /*---------Projected img Parameters---------*/
        int xmax_, ymax_;
        cv::Mat *stacked_img_;
        std::queue<cv::Mat> imgq_;

        /*---------Pretrained model---------*/
        torch::jit::script::Module pointflow_net;
    
    public:
        PFO(ros::NodeHandle nh, ros::NodeHandle private_nh, std::string model_path);
        ~PFO(){};

        void cloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg);
        void imuCallback(const sensor_msgs::Imu::ConstPtr &msg);
        
        void pointCloud2ParnomaicView(const pcl::PointCloud<pcl::PointXYZ> &pcd, bool show);
        void stack_image(void);
        int normalize(const double &x, double &xmin, double &xmax);
};

#endif
