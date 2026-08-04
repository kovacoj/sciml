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
    // The numerator functor evaluates psi^T K psi (e.g. the O(nFaces) face
    // stencil of LhoFvOperator); it must be differentiable.
    EigenTraining
    (
        std::function<torch::Tensor(const torch::Tensor&)> numerator,
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
        int lbfgsMaxIter = 300,
        int lbfgsHistorySize = 100,
        int logEvery = 20
    );

    // Train a coordinate neural network state
    // Schedule: pretrainSteps (Adam, higher LR) -> adamSteps (Adam)
    //           -> ONE LBFGS run with at most lbfgsMaxIter internal iterations
    //           (tolerance-controlled), with `restarts` restarts (1 = none)
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
        long baseSeed = 1234,
        int lbfgsHistorySize = 100,
        int logEvery = 25
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

    // Per-phase wall-clock and closure-evaluation accounting, printed at the
    // end of each state so the optimization budget is auditable.
    struct TimingStats
    {
        double forward = 0.0;   // network forward passes
        double rayleigh = 0.0;  // projection + quotient evaluation
        double backward = 0.0;
        double optimizer = 0.0; // optimizer.step() (incl. LBFGS line search)
        long closureCalls = 0;  // LBFGS closure evaluations

        void print(int stateIndex) const;
    };

    torch::Tensor M_;
    std::function<torch::Tensor(const torch::Tensor&)> numerator_;
    int nCells_;
    double epsilon_;
};

#endif
