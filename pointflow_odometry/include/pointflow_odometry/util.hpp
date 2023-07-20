#ifndef POINTFLOW_ODOMETRY_UTIL_HPP
#define POINTFLOW_ODOMETRY_UTIL_HPP

#include <torch/torch.h>
#include <torch/script.h>

int normalize(const double &x, double &xmin, double &xmax);
torch::jit::script::Module load_model(const std::string &model_path);
void cvt2Tensor(void);


#endif