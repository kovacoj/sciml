#include "CoordinateMLP.H"
#include <stdexcept>

CoordinateMLP::CoordinateMLP(int inputSize, int hiddenWidth, int hiddenLayers)
:
    // input channels [coords..., 1]: the constant channel is required so the
    // base network is not exactly odd at initialization (zero biases + tanh
    // => f(-c) = -f(c), which would make even-parity symmetrization
    // identically zero and kill all gradients).
    input_(register_module("input", torch::nn::Linear(inputSize, hiddenWidth))),
    output_(nullptr)  // registered below, after the hidden layers
{
    if (hiddenLayers < 1)
    {
        throw std::invalid_argument("CoordinateMLP: hiddenLayers must be >= 1");
    }

    for (int i = 0; i < hiddenLayers - 1; ++i)
    {
        hidden_.push_back(register_module(
            "hidden" + std::to_string(i),
            torch::nn::Linear(hiddenWidth, hiddenWidth)));
    }
    output_ = register_module("output", torch::nn::Linear(hiddenWidth, 1));

    initialize();
}

void CoordinateMLP::initialize()
{
    // Deterministic Xavier initialization with zero biases.  Modules were
    // registered in forward order (input, hidden..., output), so iterating
    // over all parameters reproduces the exact random-draw sequence of the
    // original fixed-depth network when hiddenLayers == 3.
    for (auto& p : this->parameters(/*recurse=*/true))
    {
        if (p.dim() >= 2)
        {
            torch::nn::init::xavier_uniform_(p);
        }
        else
        {
            torch::nn::init::zeros_(p);
        }
    }

    // float64 so the eigensolver runs in double precision
    this->to(torch::kFloat64);
}

torch::Tensor CoordinateMLP::forward(torch::Tensor xi)
{
    auto x = torch::tanh(input_(xi));
    for (auto& layer : hidden_)
    {
        x = torch::tanh(layer->forward(x));
    }
    x = output_->forward(x);
    return x;
}
