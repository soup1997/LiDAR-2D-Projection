#include <pointflow_odometry/util.hpp>

int normalize(const double &x, double &xmin, double &xmax){
    double x_new = (x - xmin) / (xmax - xmin) * 255.0;
    return static_cast<int>(x_new);
}

torch::jit::script::Module load_model(const std::string &model_path) {
    torch::jit::script::Module pointflow_net;

    try{
        pointflow_net = torch::jit::load(model_path);
        torch::jit::setGraphExecutorOptimize(false);
        std::cout << "Pretrained model loaded successfully" << std::endl;
        return pointflow_net;
    }
    catch (const c10::Error& e){
        std::cerr <<"Error occured while loading the model:" << e.msg() << std::endl;
        exit(-1);
    }
}

void cvt2Tensor(void);
