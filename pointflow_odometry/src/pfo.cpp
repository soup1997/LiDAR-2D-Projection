#include <pointflow_odometry/pfo.hpp>

PFO::PFO(ros::NodeHandle nh, ros::NodeHandle private_nh){
    /*------------Image params-----------*/
    _xmax = 1800; _ymax = 77;

    /*------------Imu variables-----------*/
    _gyroX = 0.0; _gyroY=0.0; _gyroZ=0.0;
    _accX = 0.0; _accY = 0.0; _accZ=0.0;
    _prev_imu_time=0.0; _curr_imu_time=0.0;
    _dt=0.0;

    /*------------Sensor params-----------*/
    private_nh.param<std::string>("point_cloud_topic", _point_cloud_topic, "kitti/velo/pointcloud");
    ROS_INFO("PointCloud2 topic: %s", _point_cloud_topic.c_str());

    private_nh.param<std::string>("imu_topic", _imu_topic, "kitti/oxts/imu");
    ROS_INFO("Imu topic: %s", _imu_topic.c_str());

    private_nh.getParam("HDL64E/HRES", _hres);
    ROS_INFO("HDL64E horizontal resolution: %.2f", _hres);
    _hres_rad = _hres * (CV_PI / 180.0);

    private_nh.getParam("HDL64E/VRES", _vres);
    ROS_INFO("HDL64E vertical resolution: %.2f", _vres);
    _vres_rad = _vres * (CV_PI / 180.0);

    private_nh.getParam("HDL64E/VFOV", _vfov);
    ROS_INFO("HDL64E vertical field of view: %.2f", _vfov);

    private_nh.getParam("HDL64E/VFOV_MIN", _vfov_min);
    ROS_INFO("HDL64E minimim vertical field of view: %.2f", _vfov_min);

    private_nh.getParam("HDL64E/VFOV_MAX", _vfov_max);
    ROS_INFO("HDL64E maximum vertical field of view: %.2f", _vfov_max);

    private_nh.getParam("HDL64E/RANGE", _range);
    ROS_INFO("HDL64E maximum range: %.2f", _range);

    private_nh.getParam("HDL64E/YFUDGE", _yfudge);
    ROS_INFO("HDL64E y axis fudge: %.2f", _yfudge);
    
    /*------------Subscriber Definition-----------*/
    pcd_sub = nh.subscribe(_point_cloud_topic, 10, &PFO::cloudCallback, this);
    imu_sub = nh.subscribe(_imu_topic, 10, &PFO::imuCallback, this);
}

void PFO::cloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg){
    if(msg->data.size() == 0) return;

    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *pcd_ptr);

    cv::Mat img = pointCloud2ParnomaicView(*pcd_ptr, true);
    //_imgq.push(img);
    //stack_image();
}


cv::Mat PFO::pointCloud2ParnomaicView(const pcl::PointCloud<pcl::PointXYZ> &pcd, bool show){
    cv::Mat projected_img(_ymax+1, _xmax+1, CV_8UC3, cv::Scalar(0, 0, 0));

    # pragma omp parallel for
    for(auto &point: pcd.points){
        // extract x, y, z, r points
        double x = point.x;
        double y = point.y;
        double z = point.z;
        double r = std::sqrt(x*x + y*y);

        // mapping to cylinder
        int x_idx = static_cast<int>(std::atan2(y, x) / _hres_rad);
        int y_idx = static_cast<int>(-std::atan2(z, r) / _vres_rad);

        // theoritical max height for image
        double d_plane = (_vfov / _vres) / (_vfov * (CV_PI / 180.0));
        double h_below = d_plane * std::tan(_vfov_min * (CV_PI / 180.0));
        double h_above = d_plane * std::tan(_vfov_max * (CV_PI / 180.0));

        // shift coordinates to make 0, 0 the origin
        int x_min = static_cast<int>(-(360.0 / _hres) / 2.0);
        x_idx = std::trunc(x_idx - x_min);

        int y_min = static_cast<int>(-((_vfov_max / _vres) + _yfudge));
        y_idx = std::trunc(y_idx - y_min);

        // fill the image
        if (x_idx >= 0 && x_idx <= _xmax && y_idx >= 0 && y_idx <= _ymax){
            projected_img.at<cv::Vec3b>(y_idx, x_idx)[0] = normalize(r, 0.0, _range);
            projected_img.at<cv::Vec3b>(y_idx, x_idx)[1] = normalize(y, -_range, _range);
            projected_img.at<cv::Vec3b>(y_idx, x_idx)[2] = normalize(z, -50.5, 4.2);
        }       
    }

    if(show) {
        cv::imshow("projected image", projected_img);
        cv::waitKey(1);
    }

    return projected_img;
}

int PFO::normalize(double &x, double xmin, double xmax){
    double x_new = (x - xmin) / (xmax - xmin) * 255.0;
    return static_cast<int>(x_new);
}


void PFO::stack_image(void){
    if(_imgq.size() >= 2) {
        cv::Mat *stacked_img;

        auto img1 = _imgq.front();
        auto img2 = _imgq.back();

        // stack the two consecutive image
        std::vector<cv::Mat> img1_channels, img2_channels, stacked_channels;

        cv::split(img1, img1_channels);
        cv::split(img2, img2_channels);

        stacked_channels.push_back(img1_channels[0]);
        stacked_channels.push_back(img1_channels[1]);
        stacked_channels.push_back(img1_channels[2]);

        stacked_channels.push_back(img2_channels[0]);
        stacked_channels.push_back(img2_channels[1]);
        stacked_channels.push_back(img2_channels[2]);

        cv::merge(stacked_channels, *stacked_img);
        std::cout << stacked_img->size() << std::endl;
        _imgq.pop(); // after stacking the image, pop the img1       
    }
}

void PFO::imuCallback(const sensor_msgs::Imu::ConstPtr &msg){
    if(msg == nullptr) return;

    _curr_imu_time = msg->header.stamp.toSec();
    _gyroX = msg->angular_velocity.x;
    _gyroY = msg->angular_velocity.y;
    _gyroZ = msg->angular_velocity.z;

    _accX = msg->linear_acceleration.x;
    _accY = msg->linear_acceleration.y;
    _accZ = msg->linear_acceleration.z;

    _dt = _curr_imu_time - _prev_imu_time;

    _prev_imu_time = _curr_imu_time;
}
