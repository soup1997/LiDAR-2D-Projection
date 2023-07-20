#include <pointflow_odometry/ieskf.hpp>

IESKF::IESKF(const double &a_std, const double &a_b, const double &w_std, const double &w_b)
{
    a_std_ = a_std;
    w_std_ = w_std;
    
    // nominal state
    pos_ = Eigen::Vector3d::Zero();
    vel_ = Eigen::Vector3d::Zero();
    quat_ = Eigen::Quaterniond::Identity();
    a_b_ = a_b * Eigen::Vector3d::Ones();
    w_b_ = w_b * Eigen::Vector3d::Ones();
    grav_vec_ << 0, 0, -GRAVITY;
    R_ = Eigen::Matrix3d::Identity(); // there is no rotation difference between inertial and body
    x_ << pos_, vel_, quat_, a_b_, w_b_, grav_vec_; // make nominal state matrix

    // error state
    dpos_ = Eigen::Vector3d::Zero();
    dvel_ = Eigen::Vector3d::Zero();
    dtheta_ = Eigen::Vector3d::Zero();
    da_b_ = a_b * Eigen::Vector3d::Ones();
    dw_b_ = w_b * Eigen::Vector3d::Ones();
    dgrav_vec_ << 0, 0, 0.0001;
    R_inv_ = R_.transpose(); // there is no rotation difference between inertial and body
    dx_ << dpos_, dvel_, dtheta_, da_b_, dw_b_, dgrav_vec_; // make error state matrix

    P_.setZero();
    Q_.setZero();
    Fx_.setZero(); 
    
    Fi_.setZero(); // (14 X 12) matrix
    Fi_.block<3, 3>(1, 0) = Eigen::Matrix3d::Identity();
    Fi_.block<3, 3>(4, 3) = Eigen::Matrix3d::Identity();
    Fi_.block<3, 3>(7, 6) = Eigen::Matrix3d::Identity();
    Fi_.block<3, 3>(10, 9) = Eigen::Matrix3d::Identity();
}

void IESKF::set_nominal_state(const Eigen::Vector3d &a_m, const Eigen::Vector3d &w_m, const double &dt)
{
    pos_ += (vel_ * dt) + ((0.5 * R_ * (a_m - a_b_) + w_b_) * std::pow(dt, 2));
    vel_ += (R_ * (a_m - a_b_));
    quat_ *= to_quaternion(((w_m - w_b_) * dt)).normalized();

    x_ << pos_, vel_, quat_, a_b_, w_b_, grav_vec_;
}

void IESKF::set_error_state(const Eigen::Vector3d &a_m, const Eigen::Vector3d &w_m, const double &dt)
{
    Fx_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();
    Fx_.block<3, 3>(0, 3) = dt * Eigen::Matrix3d::Identity();
    
    Fx_.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity();
    Fx_.block<3, 3>(3, 6) = -R_ * to_skew(a_m - a_b_) * dt;
    Fx_.block<3, 3>(3, 9) = -R_ * dt;
    Fx_.block<3, 3>(3, 15) = dt * Eigen::Matrix3d::Identity();
    
    Fx_.block<3, 3>(6, 6) = R_inv_ * ((w_m - w_b_) * dt);
    Fx_.block<3, 3>(6, 12) = -dt * Eigen::Matrix3d::Identity();

    Fx_.block<3, 3>(9, 9) = Eigen::Matrix3d::Identity();

    Fx_.block<3, 3>(12, 12) = Eigen::Matrix3d::Identity();

    Fx_.block<3, 3>(15, 15) = Eigen::Matrix3d::Identity();
}

void IESKF::set_covariance(const double &dt)
{
    P_.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();                            // position covariance
    P_.block<3, 3>(3, 3) = std::pow(a_std_ * dt, 2) * Eigen::Matrix3d::Identity(); // velocity covariance
    P_.block<4, 4>(6, 6) = std::pow(w_std_ * dt, 2) * Eigen::Matrix3d::Identity(); // orientation covariance
    P_.block<3, 3>(10, 10) = std::pow(a_std_, 2) * Eigen::Matrix3d::Identity();    // acceleration covariance
    P_.block<3, 3>(13, 13) = std::pow(w_std_, 2) * Eigen::Matrix3d::Identity();    // gyro covariance
    P_.block<3, 3>(16, 16) = Eigen::Matrix3d::Zero();                              // gravity vector covariance

    // random impulse covariance
    v_i_ = std::pow(a_std_ * dt, 2) * Eigen::Vector3d::Identity();     // velocity random impulse
    theta_i_ = std::pow(w_std_ * dt, 2) * Eigen::Vector3d::Identity(); // angle random impulse
    a_i_ = std::pow(a_b_(0), 2) * dt * Eigen::Vector3d::Identity();    // acceleration bias random impulse
    w_i_ = std::pow(w_b_(0), 2) * dt * Eigen::Vector3d::Identity();    // gyro bias random impulse
    
    Q_.block<3, 3>(0, 0) = v_i_;
    Q_.block<3, 3>(3, 3) = theta_i_;
    Q_.block<3, 3>(6, 6) = a_i_;
    Q_.block<3, 3>(9, 9) = w_i_;
}

void IESKF::prediction(const Eigen::Vector3d &a_m, const Eigen::Vector3d &w_m, const double &dt)
{
    set_nominal_state(a_m, w_m, dt);
    set_error_state(a_m, w_m, dt);
    set_covariance(dt);
    dx_ = Fx_ * dx_;
    P_ = (Fx_*P_*Fx_.transpose()) + (Fi_*Q_*Fi_.transpose());
}

void IESKF::correction(const Eigen::VectorXd &z){
    
}

Eigen::Matrix3d IESKF::to_skew(const Eigen::Vector3d &in)
{
    Eigen::Matrix3d out;

    out << 0, -in(2), in(1),
        in(2), 0, -in(0),
        -in(1), in(0), 0;

    return out;
}

Eigen::Quaterniond IESKF::to_quaternion(const Eigen::Vector3d &euler)
{
    Eigen::AngleAxisd roll(euler(0), Eigen::Vector3d::UnitX());
    Eigen::AngleAxisd pitch(euler(1), Eigen::Vector3d::UnitY());
    Eigen::AngleAxisd yaw(euler(2), Eigen::Vector3d::UnitZ());

    Eigen::Quaterniond q = yaw * pitch * roll;

    return q;
}