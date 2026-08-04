#include "CoordinateMLP.H"

CoordinateMLP::CoordinateMLP(int inputSize, int hiddenWidth, int hiddenLayers)
:
    // input channels [coords..., 1]: the constant channel is required so the
    // base network is not exactly odd at initialization (zero biases + tanh
    // => f(-c) = -f(c), which would make even-parity symmetrization
    // identically zero and kill all gradients).
    linear1_(register_module("linear1", torch::nn::Linear(inputSize, hiddenWidth))),
    linear2_(register_module("linear2", torch::nn::Linear(hiddenWidth, hiddenWidth))),
    linear3_(register_module("linear3", torch::nn::Linear(hiddenWidth, hiddenWidth))),
    linear4_(register_module("linear4", torch::nn::Linear(hiddenWidth, 1)))
{
    initialize();
}

void CoordinateMLP::initialize()
{
    // Deterministic Xavier initialization with zero biases
    torch::nn::init::xavier_uniform_(linear1_->weight);
    torch::nn::init::zeros_(linear1_->bias);

    torch::nn::init::xavier_uniform_(linear2_->weight);
    torch::nn::init::zeros_(linear2_->bias);

    torch::nn::init::xavier_uniform_(linear3_->weight);
    torch::nn::init::zeros_(linear3_->bias);

    torch::nn::init::xavier_uniform_(linear4_->weight);
    torch::nn::init::zeros_(linear4_->bias);

    // float64 so the eigensolver runs in double precision
    this->to(torch::kFloat64);
}

torch::Tensor CoordinateMLP::forward(torch::Tensor xi)
{
    auto x = torch::tanh(linear1_(xi));
    x = torch::tanh(linear2_(x));
    x = torch::tanh(linear3_(x));
    x = linear4_(x);
    return x;
}
