#include <pointflow_odometry/ieskf.hpp>

IESKF::IESKF(const double &a_b, const double &w_b)
{
    pos_ = Eigen::Vector3d::Zero();
    vel_ = Eigen::Vector3d::Zero();
    quat_ = Eigen::Quaterniond::Identity();
    a_b_ = a_b * Eigen::Vector3d::Ones();
    w_b_ = w_b * Eigen::Vector3d::Ones();
    g_v_ = Eigen::Vector3d::Zero();
    g_v_(2) = -GRAVITY;
    R_ = Eigen::Matrix3d::Identity(); // there is no rotation difference between inertial and body


    dpos_ = Eigen::Vector3d::Zero();
    dvel_ = Eigen::Vector3d::Zero();
    dtheta_ = Eigen::Vector3d::Zero();
    da_b_ = a_b * Eigen::Vector3d::Ones();
    dw_b_ = w_b * Eigen::Vector3d::Ones();
    dg_v_ = Eigen::Vector3d::Zero();
    dg_v_(2) = 0.006;
    R_inv_ = R_.transpose(); // there is no rotation difference between inertial and body

    // random impulse 초기화 필요(datasheet 뒤지거나 실험적으로 돌려보기)
}

// make state/covariance matrix
void IESKF::init(void)
{
    x_ << pos_, vel_, quat_, a_b_, w_b_, g_v_;

    P_.setZero();
    P_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();                          // position covariance
    P_.block<3, 3>(3, 3) = std::pow(a_b_(0), 2) * Eigen::Matrix3d::Identity();   // velocity covariance
    P_.block<4, 4>(6, 6) = std::pow(w_b_(0), 2) * Eigen::Matrix3d::Identity();   // orientation covariance
    P_.block<3, 3>(10, 10) = std::pow(a_b_(0), 2) * Eigen::Matrix3d::Identity(); // acceleration covariance
    P_.block<3, 3>(13, 13) = std::pow(w_b_(0), 2) * Eigen::Matrix3d::Identity(); // gyro covariance
    P_.block<3, 3>(16, 16) = Eigen::Matrix3d::Zero();                            // gravity vector covariance
}

void IESKF::predict_nominal_state(const Eigen::Vector3d &a_m, const Eigen::Vector3d &w_m, const double &dt)
{
    pos_ += (vel_ * dt) + ((0.5 * R_ * (a_m - a_b_) + g_v_) * std::pow(dt, 2));
    vel_ += vel_ + (R_ * (a_m - a_b_));
    quat_ *= to_quaternion(((w_m - w_b_) * dt)).normalized();
}

void IESKF::predict_error_state(const Eigen::Vector3d &a_m, const Eigen::Vector3d &w_m, const double &dt){
    dpos_ += dvel_ * dt;
    dvel_ += ((-R_ * to_skew(a_m - a_b_) * dtheta_) + ((-R_ * da_b_ + dg_v_)) * dt) + v_i_;
    dtheta_ = (R_inv_ * ((w_m - w_b_) * dt)) - (dw_b_ * dt) + theta_i_;
    da_b_ += a_i_;
    dw_b_ += w_i_;
}

Eigen::Matrix3d IESKF::to_skew(const Eigen::Vector3d &in)
{
    Eigen::Matrix3d out;

    out << 0, -in(2), in(1),
        in(2), 0, -in(0),
        -in(1), in(0), 0;

    return out;
}

Eigen::Quaterniond IESKF::to_quaternion(const Eigen::Vector3d &euler){
    Eigen::AngleAxisd roll(euler(0), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(euler(1), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(euler(2), Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yaw * pitch * roll;

    return q;
}