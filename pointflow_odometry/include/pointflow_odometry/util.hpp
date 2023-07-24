#ifndef POINTFLOW_ODOMETRY_UTIL_HPP
#define POINTFLOW_ODOMETRY_UTIL_HPP

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/core.hpp>
#include <Eigen/Dense>

int normalize(const double &x, double xmin, double xmax);
torch::jit::script::Module load_model(const std::string &model_path);
torch::Tensor matToTensor(const cv::Mat &stacked_img);
Eigen::VectorXd tensorToEigen(const torch::Tensor& tensor);
#endif