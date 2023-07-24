#include <pointflow_odometry/util.hpp>

int normalize(const double &x, double xmin, double xmax){
    double x_new = (x - xmin) / (xmax - xmin) * 255.0;
    return static_cast<int>(x_new);
}

torch::jit::script::Module load_model(const std::string &model_path) {
    torch::jit::script::Module pointflow_net;

    try{
        pointflow_net = torch::jit::load(model_path);
        torch::jit::setGraphExecutorOptimize(false);
        return pointflow_net;
    }
    catch (const c10::Error& e){
        std::cerr <<"Error occured while loading the model:" << e.msg() << std::endl;
        exit(-1);
    }
}

torch::Tensor matToTensor(const cv::Mat &stacked_img){
    static torch::TensorOptions options(torch::kFloat32);

    cv::Mat stacked_img_float;
    stacked_img.convertTo(stacked_img_float, CV_32FC3);
    stacked_img_float /= 255.0;
    torch::Tensor tensor_img(torch::from_blob(stacked_img_float.data, {1, stacked_img_float.channels(), stacked_img_float.rows, stacked_img_float.cols}, options));

    return tensor_img;
}

Eigen::VectorXd tensorToEigen(const torch::Tensor& tensor) {
    auto tensor_1d = tensor.reshape({-1}); // Reshape to a 1D tensor
    int numel = tensor_1d.numel();

    std::vector<double> data(numel);
    auto tensor_data = tensor_1d.accessor<float, 1>();

    for (int i = 0; i < numel; ++i) {
        data[i] = static_cast<double>(tensor_data[i]);
    }

    return Eigen::Map<Eigen::VectorXd>(data.data(), numel);
}