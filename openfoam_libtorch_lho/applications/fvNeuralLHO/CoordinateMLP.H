#ifndef CoordinateMLP_H
#define CoordinateMLP_H

#include "TorchCompat.H"
#include <string>
#include <vector>

class CoordinateMLP : public torch::nn::Module
{
public:
    // Architecture: Linear(inputSize, hiddenWidth) -> tanh, then
    // (hiddenLayers - 1) x [Linear(hiddenWidth, hiddenWidth) -> tanh], then
    // Linear(hiddenWidth, 1).  hiddenLayers=3 reproduces the original fixed
    // four-layer network.
    CoordinateMLP(int inputSize, int hiddenWidth = 64, int hiddenLayers = 3);

    torch::Tensor forward(torch::Tensor xi);

    void initialize();

private:
    torch::nn::Linear input_;
    std::vector<torch::nn::Linear> hidden_;
    torch::nn::Linear output_;
};

#endif
