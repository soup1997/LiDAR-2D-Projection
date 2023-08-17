#include <pointflow_odometry/pfo.hpp>

PFO::PFO(ros::NodeHandle nh, ros::NodeHandle private_nh, const std::string model_path) : _xmax(1800), _ymax(77), _curr_time(0.0), _prev_time(0.0), _translation(Eigen::Vector3d::Zero()), _orientation(Eigen::Quaterniond::Identity())
{
    /*------------Sensor Parameters-----------*/
    private_nh.param<std::string>("point_cloud_topic", point_cloud_topic, "kitti/velo/pointcloud");
    ROS_INFO("PointCloud2 topic: %s", point_cloud_topic.c_str());

    private_nh.param<std::string>("imu_topic", imu_topic, "kitti/oxts/imu");
    ROS_INFO("Imu topic: %s", imu_topic.c_str());

    private_nh.param<std::string>("img_topic", img_topic, "kitti/camera_color_right/image_raw");
    ROS_INFO("Img topic: %s", img_topic.c_str());

    private_nh.getParam("HDL64E/HRES", _hres);
    ROS_INFO("HDL64E horizontal resolution: %.2f", _hres);

    private_nh.getParam("HDL64E/VRES", _vres);
    ROS_INFO("HDL64E vertical resolution: %.2f", _vres);

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

    private_nh.getParam("OXTS_RT3003/ACC_STD", _acc_bias);
    ROS_INFO("OXTS RT3003 acceleration bias: %.2f [m/s^2]", _acc_bias);

    private_nh.getParam("OXTS_RT3003/ACC_BIAS", _acc_std);
    ROS_INFO("OXTS RT3003 acceleration std: %.2f [m/s^2]", _acc_std);

    private_nh.getParam("OXTS_RT3003/GYRO_BIAS", _gyro_bias);
    ROS_INFO("OXTS RT3003 gyro bias: %.2f [deg/s]", _gyro_bias);
    _gyro_bias *= (CV_PI / 180.0); //  need to convert [deg/s] to [rad/s]

    private_nh.getParam("OXTS_RT3003/GYRO_STD", _gyro_std);
    ROS_INFO("OXTS RT3003 gyro std: %.2f [deg/s]", _gyro_std);
    _gyro_std *= (CV_PI / 180.0); //  need to convert [deg/s] to [rad/s]

    /*------------ROS Pub-Sub Definition-----------*/
    //pcd_sub = nh.subscribe(point_cloud_topic, 10, &PFO::cloudCallback, this);
    imu_sub = nh.subscribe(imu_topic, 10, &PFO::imuCallback, this);
    path_pub = nh.advertise<nav_msgs::Path>("path", 10);

    /*------------6DOF Variables-----------*/
    _translation = Eigen::Vector3d::Zero();
    _orientation.setIdentity();
    _pose = Eigen::Matrix3d::Identity();

    /*------------Pytorch Model-----------*/
    _model = load_model(model_path);
    _model.eval();
    ROS_INFO("Pretrained model loaded sucessfully");

    /*------------IESKF-----------*/
    // IESKF ieskf(_acc_std, _acc_bias, _gyro_std, _gyro_std);
    // ieskf.init();
}

void PFO::stack_image(void)
{
    auto img1 = _img_queue.front();
    auto img2 = _img_queue.back();

    // stack the two consecutive image
    std::vector<cv::Mat> img1_channels, img2_channels, stacked_channels;

    cv::split(img1, img1_channels);
    cv::split(img2, img2_channels);

    stacked_channels.emplace_back(img1_channels[0]);
    stacked_channels.emplace_back(img1_channels[1]);
    stacked_channels.emplace_back(img1_channels[2]);

    stacked_channels.emplace_back(img2_channels[0]);
    stacked_channels.emplace_back(img2_channels[1]);
    stacked_channels.emplace_back(img2_channels[2]);

    cv::merge(stacked_channels, _stacked_img); // make 6 channel image for model input
    _img_queue.pop();                          // after stacking the image, pop the img 1
}

void PFO::pathPublisher(const Eigen::Vector3f &translation, const Eigen::Quaternionf &orientation)
{
    static nav_msgs::Path path;
    static int seq = 0;

    // transform local pose to global pose
    _translation += translation;
    _orientation *= orientation;
    _orientation.normalize();


    path.header.seq = seq;
    path.header.frame_id = "world";
    path.header.stamp = ros::Time::now();

    geometry_msgs::PoseStamped pose;
    pose.header.seq = seq;
    //path.header.frame_id = "";
    pose.header.stamp = ros::Time::now();
    pose.pose.position.x = _translation(0);
    pose.pose.position.y = _translation(1);
    pose.pose.position.z = 0.0;

    pose.pose.orientation.w = _orientation.w();
    pose.pose.orientation.x = _orientation.x();
    pose.pose.orientation.y = _orientation.y();
    pose.pose.orientation.z = _orientation.z();

    path.poses.push_back(pose);
    path_pub.publish(path);
    
    seq++;
}

void PFO::pointCloud2ParnomaicView(const pcl::PointCloud<pcl::PointXYZ> &pcd, const bool &show)
{
    cv::Mat projected_img(_ymax + 1, _xmax + 1, CV_8UC3, cv::Scalar(0, 0, 0));
    cv::Mat resized_img(64, 1024, CV_8UC3, cv::Scalar(0, 0, 0));

#pragma omp parallel for
    for (auto &point : pcd.points)
    {
        // extract x, y, z, r points
        double x = point.x;
        double y = point.y;
        double z = point.z;
        double r = std::sqrt(x * x + y * y);

        // mapping to cylinder
        int x_idx = static_cast<int>(std::atan2(y, x) / (_hres * (CV_PI / 180.0)));
        int y_idx = static_cast<int>(-std::atan2(z, r) / (_vres * (CV_PI / 180.0)));

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
        if (x_idx >= 0 && x_idx <= _xmax && y_idx >= 0 && y_idx <= _ymax)
        {
            projected_img.at<cv::Vec3b>(y_idx, x_idx)[0] = normalize(r, 0.0, _range);
            projected_img.at<cv::Vec3b>(y_idx, x_idx)[1] = normalize(x, -_range, _range);
            projected_img.at<cv::Vec3b>(y_idx, x_idx)[2] = normalize(y, -_range, _range);
        }
    }

    cv::resize(projected_img, resized_img, cv::Size(1024, 64));
    _img_queue.push(resized_img);

    if (show)
    {
        cv::imshow("projected image", projected_img);
        cv::imshow("resized image", resized_img);
        cv::waitKey(1);
    }
}

void PFO::cloudCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
    if (msg == nullptr)
        return;

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    /*------------cloudCallback Main-----------*/
    pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_ptr(new pcl::PointCloud<pcl::PointXYZ>);
    pcl::fromROSMsg(*msg, *pcd_ptr);
    pointCloud2ParnomaicView(*pcd_ptr, false);

    if (_img_queue.size() >= 2)
    {
        stack_image();
        torch::Tensor tensor_img(matToTensor(_stacked_img));
        torch::Tensor output(_model.forward({tensor_img}).toTensor()); 
        std::cout << "1: " << output << std::endl;
        
        Eigen::VectorXf output_vec(tensorToEigen(output)); // output이랑 tensorToEigen결과랑 다름, 수정 필요
        Eigen::Vector3f translation(output_vec(0), output_vec(1), output_vec(2));
        Eigen::Quaternionf orientation(output_vec(3), output_vec(4), output_vec(5), output_vec(6));
        orientation.normalize();

        std::cout << "2: " << translation << " " << orientation.coeffs() << std::endl;

        pathPublisher(translation, orientation);
    }
    /*-----------------------------------------*/

    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::milliseconds sec(std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
    std::cout << "Execution Time on CloudCallback: " << sec.count() << " ms\n";
}
void PFO::imgCallback(const sensor_msgs::Image::ConstPtr &msg){
    if (msg == nullptr) return;

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
    
    cv_bridge::CvImagePtr cv_ptr(cv_bridge::toCvCopty(*msg, enc::BGR8);
    
    if (_img_queue.size() >= 2)
    {
        stack_image();
        torch::Tensor tensor_img(matToTensor(_stacked_img));
        torch::Tensor output(_model.forward({tensor_img}).toTensor()); 
        std::cout << "1: " << output << std::endl;
        
        Eigen::VectorXf output_vec(tensorToEigen(output)); // output이랑 tensorToEigen결과랑 다름, 수정 필요
        Eigen::Vector3f translation(output_vec(0), output_vec(1), output_vec(2));
        Eigen::Quaternionf orientation(output_vec(3), output_vec(4), output_vec(5), output_vec(6));
        orientation.normalize();

        std::cout << "2: " << translation << " " << orientation.coeffs() << std::endl;

        pathPublisher(translation, orientation);
    }
    
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::milliseconds sec(std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
    std::cout << "Execution Time on imgCallback: " << sec.count() << " ms\n";
}
void PFO::imuCallback(const sensor_msgs::Imu::ConstPtr &msg)
{
    if (msg == nullptr)
        return;

    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();

    /*------------imuCallback Main-----------*/
    _curr_time = msg->header.stamp.toSec();
    _gyro_val = Eigen::Vector3d(msg->angular_velocity.x, msg->angular_velocity.y, msg->angular_velocity.z);
    _acc_val = Eigen::Vector3d(msg->linear_acceleration.x, msg->linear_acceleration.y, msg->linear_acceleration.z);
    _prev_time = _curr_time;
    /*-----------------------------------------*/

    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    std::chrono::milliseconds sec(std::chrono::duration_cast<std::chrono::milliseconds>(end - start));
    std::cout << "Execution Time on imuCallback: " << sec.count() << " ms\n";
}
