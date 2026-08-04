#ifndef EigenTraining_H
#define EigenTraining_H

#include "TorchCompat.H"
#include <vector>
#include <string>
#include <functional>
#include "CoordinateMLP.H"

struct TrainingState
{
    int step;
    double energy;
    double massNorm;
    double gradNorm;
    double orthogonalityError;
    double elapsedSeconds;
};

class EigenTraining
{
public:
    EigenTraining
    (
        const torch::Tensor& K,
        const torch::Tensor& M,
        int numberOfStates = 6,
        double epsilon = 1e-30
    );

    // Train a single state with the cell values as trainable coefficients
    std::pair<torch::Tensor, std::vector<TrainingState>> trainCoefficientState
    (
        int stateIndex,
        const std::vector<torch::Tensor>& previousStates,
        int adamSteps = 3000,
        double adamLr = 1e-2,
        int lbfgsMaxIter = 300
    );

    // Train a coordinate neural network state
    // Schedule: pretrainSteps (Adam, higher LR) -> adamSteps (Adam)
    //           -> lbfgsMaxIter (LBFGS), with `restarts` restarts (1 = none)
    std::pair<torch::Tensor, std::vector<TrainingState>> trainNeuralState
    (
        CoordinateMLP& model,
        int stateIndex,
        const std::vector<torch::Tensor>& previousStates,
        const torch::Tensor& coords,
        bool enforceParity = true,
        int pretrainSteps = 150,
        int adamSteps = 500,
        double adamLr = 1e-3,
        int lbfgsMaxIter = 20,
        int restarts = 1,
        long baseSeed = 1234
    );

    // M-weighted Rayleigh quotient (scalar)
    double rayleighQuotient(const torch::Tensor& psi);

    // Mass norm squared ||psi||_M^2
    double massNorm(const torch::Tensor& psi) const;

    // Max |<psi, psi_j>_M| over previously accepted states
    double orthogonalityError
    (
        const torch::Tensor& psi,
        const std::vector<torch::Tensor>& previousStates
    ) const;

private:
    // Differentiable projection + M-normalisation (never detaches)
    torch::Tensor computePsi
    (
        const torch::Tensor& raw,
        const std::vector<torch::Tensor>& previousStates
    );

    // Differentiable Rayleigh-quotient tensor
    torch::Tensor lossTensor(const torch::Tensor& psi);

    // One zero_grad + forward + backward sweep used by optimizers
    double closureLoss
    (
        torch::optim::Optimizer& optimizer,
        const std::function<torch::Tensor()>& forwardPsi,
        const std::vector<torch::Tensor>& previousStates
    );

    torch::Tensor K_;
    torch::Tensor M_;
    int nCells_;
    double epsilon_;
};

#endif
