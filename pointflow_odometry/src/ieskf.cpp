#include <pointflow_odometry/ieskf.hpp>

IESKF::IESKF(const Eigen::Vector3d &pos, const Eigen::Vector3d &vel, const Eigen::Quaterniond &quat, const Eigen::Vector3d &acc_b, const Eigen::Vector3d &gyro_b, const Eigen::Vector3d &grav_vec){
    pos_ = pos;
    vel_ = vel;
    quat_ = quat;
    acc_b_ = acc_b;
    gyro_b_ = gyro_b;
    grav_vec_ = grav_vec;
} // 외부에서는 행렬을 변수로 주지말고 오직 데이터값이랑 바이어스 값만 건내주기, 수정 필요

// make state/covariance matrix
void IESKF::init(){
    x_ << pos_, vel_, quat_, acc_b_, gyro_b_, grav_vec_;

    P_.setZero();
    P_.block<3, 3>(0, 0) = acc_bias_ * Eigen::Matrix3d::Identity();
    P_.block<3, 3>(3, 3) = 0.1 * Eigen::Matrix3d::Identity();
    P_.block<4, 4>(6, 6) = gyro_bias_ * Eigen::Matrix3d::Identity(); 
    P_.block<3, 3>(10, 10) = 0.001 * Eigen::Matrix3d::Identity();
    P_.block<3, 3>(13, 13) = 0.001 * Eigen::Matrix3d::Identity();
    P_.block<3, 3>(16, 16) = Eigen::Matrix3d::Zero();
}

