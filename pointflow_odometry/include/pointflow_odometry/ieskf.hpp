#ifndef POINTFLOW_ODOMETRY_IESKF_HPP
#define POINTFLOW_ODOMETRY_IESKF_HPP

#define STATE_SIZE 18
#define GRAVITY 9.8

#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

class IESKF
{
private:
    double a_std_, w_std_; // acceleration std, gyro std

    /*------------Nominal State Variables-----------*/
    Eigen::Vector3d pos_, vel_; // position, velocity
    Eigen::Quaterniond quat_;   // quaternion
    Eigen::Vector3d a_b_, w_b_; // acceleration bias, gyro bias
    Eigen::Vector3d grav_vec_;  // gravity vector
    Eigen::Matrix3d R_;         // from inertial to body frame

    /*------------Error State Variables-----------*/
    Eigen::Vector3d dpos_, dvel_; // delta position, delta velocity
    Eigen::Vector3d dtheta_;      // delta theta
    Eigen::Vector3d da_b_, dw_b_; // delta acceleration bias, delta gyro bias
    Eigen::Vector3d dgrav_vec_;   // delta gravity vector
    Eigen::Matrix3d R_inv_;       // from body to inertial frame

    Eigen::Vector3d v_i_;     // velocity random impulse
    Eigen::Vector3d theta_i_; // angle random impulse
    Eigen::Vector3d a_i_;     // acceleration bias random impulse
    Eigen::Vector3d w_i_;     // angular velocity bias random impulse

    /*------------State Propagation Variables-----------*/
    Eigen::Matrix<double, STATE_SIZE + 1, 1> x_;      // nominal state
    Eigen::Matrix<double, STATE_SIZE, 1> dx_;         // error state
    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> P_; // error state covariance
    Eigen::Matrix<double, 12, 12> Q_;                 // random impulse covariance

    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> Fx_; // error state transition matrix
    Eigen::Matrix<double, 14, 12> Fi_;                 // error state covariance transition matrix

    Eigen::Matrix<double, 7, STATE_SIZE> H_;
    Eigen::Matrix<double, 7, 7> V_; // covariance matrix of measurement(x, y, z, qw, qx, qy, qz)
    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> G_;

    std::vector<Eigen::Matrix<double, STATE_SIZE, 1>> state_history_;

public:
    IESKF(const double &a_std, const double &a_b, const double &w_std, const double &w_b);
    ~IESKF(){};

    void set_nominal_state(const Eigen::Vector3d &a_m, const Eigen::Vector3d &w_m, const double &dt);
    void set_error_state(const Eigen::Vector3d &a_m, const Eigen::Vector3d &w_m, const double &dt);
    void set_covariance(const double &dt);
    void prediction(const Eigen::Vector3d &a_m, const Eigen::Vector3d &w_m, const double &dt);
    void correction(const Eigen::VectorXd &z);
    void reset();

    Eigen::Matrix3d to_skew(const Eigen::Vector3d &in);
    Eigen::Quaterniond to_quaternion(const Eigen::Vector3d &euler);
};
#endif