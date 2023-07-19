#ifndef POINTFLOW_ODOMETRY_IESKF_HPP
#define POINTFLOW_ODOMETRY_IESKF_HPP

#define POS_SIZE 3
#define VEL_SIZE 3
#define QUAT_SIZE 4
#define AB_SIZE 3
#define GB_SIZE 3
#define GV_SIZE 3
#define STATE_SIZE (POS_SIZE + VEL_SIZE + QUAT_SIZE + AB_SIZE + GB_SIZE + GV_SIZE)

#define GRAVITY 9.8

#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <Eigen/Geometry>

class IESKF
{
private:
    /*------------Nominal State Variables-----------*/
    Eigen::Vector3d pos_, vel_; // position, velocity
    Eigen::Quaterniond quat_;   // quaternion
    Eigen::Vector3d a_b_, w_b_; // acceleration bias, gyro bias
    Eigen::Vector3d g_v_;       // gravity vector
    Eigen::Matrix3d R_;         // from inertial to body frame

    /*------------Error State Variables-----------*/
    Eigen::Vector3d dpos_, dvel_; // delta position, delta velocity
    Eigen::Vector3d dtheta_;      // delta theta
    Eigen::Vector3d da_b_, dw_b_; // delta acceleration bias, delta gyro bias
    Eigen::Vector3d dg_v_;        // delta gravity vector
    Eigen::Matrix3d R_inv_;       // from body to inertial frame
    Eigen::Vector3d v_i_;     // velocity random impulse
    Eigen::Vector3d theta_i_; // angle random impulse
    Eigen::Vector3d a_i_;     // acceleration bias random impulse
    Eigen::Vector3d w_i_;     // angular velocity bias random impulse

    /*------------State Propagation Variables-----------*/
    Eigen::Matrix<double, STATE_SIZE, 1> x_hat_;                 // estimated state
    Eigen::Matrix<double, STATE_SIZE, 1> x_;                     // nominal state
    Eigen::Matrix<double, STATE_SIZE, 1> dx_;                    // error state
    Eigen::Matrix<double, STATE_SIZE, 1> P_hat_;                 // estimated covariance
    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> P_;            // nominal covariance
    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> rand_impulse_; // error covariance

    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> F_; // state transition matrix
    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> G_; // input matrix
    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> H_; // measurement matrix

    double current_time_;
    double prev_time_;

    std::vector<Eigen::Matrix<double, STATE_SIZE, 1>> state_history_;

public:
    IESKF(const double &a_b, const double &w_b);
    ~IESKF(){};

    void init();
    void predict_nominal_state(const Eigen::Vector3d &a_m, const Eigen::Vector3d &w_m, const double &dt);
    void predict_error_state(const Eigen::Vector3d &a_m, const Eigen::Vector3d &w_m, const double &dt);
    void prediction();

    void update(const Eigen::VectorXd &z);
    void reset();

    Eigen::Matrix3d to_skew(const Eigen::Vector3d &in);
    Eigen::Quaterniond to_quaternion(const Eigen::Vector3d &euler);
    void compute_jacobian();
};
#endif