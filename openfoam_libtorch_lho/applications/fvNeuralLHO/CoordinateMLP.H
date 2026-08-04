#ifndef CoordinateMLP_H
#define CoordinateMLP_H

#include "TorchCompat.H"
#include <string>

class CoordinateMLP : public torch::nn::Module
{
public:
    CoordinateMLP(int inputSize, int hiddenWidth = 64, int hiddenLayers = 3);

    torch::Tensor forward(torch::Tensor xi);

    void initialize();

private:
    torch::nn::Linear linear1_;
    torch::nn::Linear linear2_;
    torch::nn::Linear linear3_;
    torch::nn::Linear linear4_;
};

#endif
