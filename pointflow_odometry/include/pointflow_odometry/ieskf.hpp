#ifndef POINTFLOW_ODOMETRY_IESKF_HPP
#define POINTFLOW_ODOMETRY_IESKF_HPP

#define POS_SIZE 3
#define VEL_SIZE 3
#define QUAT_SIZE 4
#define AB_SIZE 3
#define GB_SIZE 3
#define GV_SIZE 3
#define STATE_SIZE (POS_SIZE + VEL_SIZE + QUAT_SIZE + AB_SIZE + GB_SIZE + GV_SIZE)

#define GRAVITY 9.812

#include <Eigen/Dense>
#include <Eigen/Core>
#include <cmath>

class IESKF{
private:
    double acc_bias_;
    double gyro_bias_;

    // Initial value
    static Eigen::Vector3d pos_, vel_;
    static Eigen::Quaterniond quat_;
    static Eigen::Vector3d acc_b_, gyro_b_;
    static Eigen::Vector3d grav_vec_;

    static Eigen::Matrix<double, STATE_SIZE, 1> x_hat_; // estimated state
    static Eigen::Matrix<double, STATE_SIZE, 1> x_;     // nominal state
    static Eigen::Matrix<double, STATE_SIZE, 1> dx_;    // error state

    static Eigen::Matrix<double, STATE_SIZE, 1> P_hat_; // estimated covariance
    static Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> P_; // nominal covariance
    static Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> rand_impulse_; // error covariance

    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> F_; // state transition matrix
    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> G_; // input matrix
    Eigen::Matrix<double, STATE_SIZE, STATE_SIZE> H_; // measurement matrix

    double current_time;
    double prev_time;

public:
    IESKF(const Eigen::Vector3d &pos,
          const Eigen::Vector3d &vel,
          const Eigen::Quaterniond &quat,
          const Eigen::Vector3d &acc_b,
          const Eigen::Vector3d &gyro_b,
          const Eigen::Vector3d &grav_vec);

    ~IESKF(){};
    void init();
    void prediction(const Eigen::VectorXd &u);
    void update(const Eigen::VectorXd &z);
    void reset();

    void to_skew();
    void compute_jacobian();
};
#endif