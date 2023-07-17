#include <iostream>
#include <ros/ros.h>
#include "pointflow_odometry/pfo.hpp"

int main(int argc, char** argv){

    ros::init(argc, argv, "pointflow_odometry");

    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~"); // using private parameters in node
    ros::Rate loop_rate(10);

    PFO pfo(nh, private_nh);
    
    while (ros::ok()){
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}