#ifndef ShallowFlowMLP_H
#define ShallowFlowMLP_H

#include "TorchCompat.H"
#include <vector>
#include <stdexcept>

// Single-topology pointwise MLP (brief section 14):
//   input [N, 7]  (x*, y*, lambda, sigma/h, fluidMask, interfaceMask, solidMask)
//   Linear(7, W) -> tanh -> [Linear(W, W) -> tanh] x (hiddenLayers-1) -> Linear(W, 3)
// Output per cell: (u_x*, u_y*, p~*).  u_z = 0 exactly (not learned).
class ShallowFlowMLP : public torch::nn::Module
{
public:
    ShallowFlowMLP(int inputSize = 7, int hiddenWidth = 32, int hiddenLayers = 2)
    :
        input_(register_module("input", torch::nn::Linear(inputSize, hiddenWidth))),
        output_(nullptr)
    {
        if (hiddenLayers < 1)
        {
            throw std::invalid_argument("ShallowFlowMLP: hiddenLayers must be >= 1");
        }
        for (int i = 0; i < hiddenLayers - 1; ++i)
        {
            hidden_.push_back(register_module(
                "hidden" + std::to_string(i),
                torch::nn::Linear(hiddenWidth, hiddenWidth)));
        }
        output_ = register_module("output", torch::nn::Linear(hiddenWidth, 3));
        initialize();
    }

    void initialize()
    {
        for (auto& p : this->parameters(true))
        {
            if (p.dim() >= 2) torch::nn::init::xavier_uniform_(p);
            else              torch::nn::init::zeros_(p);
        }
        this->to(torch::kFloat64);
    }

    torch::Tensor forward(torch::Tensor x)
    {
        auto h = torch::tanh(input_(x));
        for (auto& layer : hidden_)
        {
            h = torch::tanh(layer->forward(h));
        }
        return output_->forward(h);
    }

private:
    torch::nn::Linear input_;
    std::vector<torch::nn::Linear> hidden_;
    torch::nn::Linear output_;
};

#endif
