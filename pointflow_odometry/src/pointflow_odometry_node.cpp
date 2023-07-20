#include <pointflow_odometry/pfo.hpp>

int main(int argc, char** argv){

    ros::init(argc, argv, "pointflow_odometry");
    const std::string model_path = "/home/smeet/catkin_ws/src/PointFlow-Odometry/trained_model/Pointflow_model_scripted.pt";
    
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~"); // using private parameters in node
    ros::Rate loop_rate(10);

    PFO pfo(nh, private_nh, model_path);
    
    while (ros::ok()){
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;
}