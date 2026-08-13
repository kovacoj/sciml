#include "EigenTraining.H"
#include "CoordinateMLP.H"
#include <iostream>
#include <chrono>
#include <cmath>
#include <stdexcept>

// NOTE: This translation unit is Torch-only.  No OpenFOAM headers are
// included, so OpenFOAM's macro set cannot interfere with LibTorch here.

using Clock = std::chrono::high_resolution_clock;

static double durationSince(const Clock::time_point& a, const Clock::time_point& b)
{
    return std::chrono::duration<double>(b - a).count();
}

void EigenTraining::TimingStats::print(int stateIndex) const
{
    std::cout << "    [timing] state " << stateIndex
              << ": forward=" << forward << "s"
              << ", rayleigh-eval=" << rayleigh << "s"
              << ", backward=" << backward << "s"
              << ", optimizer.step=" << optimizer << "s"
              << " (the LBFGS step entry includes its internal closure evaluations)"
              << ", LBFGS closure calls=" << closureCalls
              << std::endl;
}

EigenTraining::EigenTraining
(
    std::function<torch::Tensor(const torch::Tensor&)> numerator,
    const torch::Tensor& M,
    int numberOfStates,
    double epsilon
)
:
    M_(M),
    numerator_(std::move(numerator)),
    nCells_(M.size(0)),
    epsilon_(epsilon)
{
}

torch::Tensor EigenTraining::computePsi
(
    const torch::Tensor& raw,
    const std::vector<torch::Tensor>& previousStates
)
{
    auto z = raw;
    for (const auto& psi_j : previousStates)
    {
        auto coeff = (M_ * psi_j * z).sum();
        z = z - coeff * psi_j;
    }
    auto normSq = (M_ * z * z).sum() + epsilon_;
    return z / torch::sqrt(normSq);
}

torch::Tensor EigenTraining::lossTensor(const torch::Tensor& psi)
{
    auto num = numerator_(psi);
    auto den = (M_ * psi * psi).sum() + epsilon_;
    return num / den;
}

double EigenTraining::rayleighQuotient(const torch::Tensor& psi)
{
    return lossTensor(psi.detach()).item<double>();
}

double EigenTraining::massNorm(const torch::Tensor& psi) const
{
    return (M_ * psi * psi).sum().item<double>();
}

double EigenTraining::orthogonalityError
(
    const torch::Tensor& psi,
    const std::vector<torch::Tensor>& previousStates
) const
{
    double maxErr = 0.0;
    for (const auto& psi_j : previousStates)
    {
        double overlap = (M_ * psi_j * psi).sum().item<double>();
        maxErr = std::max(maxErr, std::abs(overlap));
    }
    return maxErr;
}

std::pair<torch::Tensor, std::vector<TrainingState>> EigenTraining::trainCoefficientState
(
    int stateIndex,
    const std::vector<torch::Tensor>& previousStates,
    int adamSteps,
    double adamLr,
    int lbfgsMaxIter,
    int lbfgsHistorySize,
    int logEvery
)
{
    torch::manual_seed(1234 + stateIndex * 1000);

    auto z = torch::randn(
        {nCells_},
        torch::TensorOptions().dtype(torch::kFloat64)
    );
    z.set_requires_grad(true);

    TimingStats stats;

    std::vector<TrainingState> history;
    auto t0 = Clock::now();

    auto record = [&](int step)
    {
        auto psi = computePsi(z.detach(), previousStates);
        auto t1 = Clock::now();
        TrainingState ts;
        ts.step = step;
        ts.energy = rayleighQuotient(psi);
        ts.massNorm = massNorm(psi);
        ts.gradNorm = (z.grad().defined())
            ? z.grad().abs().max().item<double>() : 0.0;
        ts.orthogonalityError = orthogonalityError(psi, previousStates);
        ts.elapsedSeconds = durationSince(t0, t1);
        history.push_back(ts);
    };

    // --- Adam phase (two scopes so the LR can drop mid-training) --------
    int firstHalf = adamSteps / 2;

    auto adamLoop = [&](torch::optim::Optimizer& optimizer, int start, int end)
    {
        for (int step = start; step < end; step++)
        {
            optimizer.zero_grad();

            auto tR0 = Clock::now();
            auto psi = computePsi(z, previousStates);
            auto energy = lossTensor(psi);
            auto tR1 = Clock::now();
            stats.rayleigh += durationSince(tR0, tR1);

            double loss = energy.item<double>();

            if (!std::isfinite(loss))
            {
                throw std::runtime_error(
                    "NaN/Inf in coefficient training at step "
                    + std::to_string(step));
            }

            auto tB0 = Clock::now();
            energy.backward();
            auto tB1 = Clock::now();
            stats.backward += durationSince(tB0, tB1);

            auto tO0 = Clock::now();
            optimizer.step();
            auto tO1 = Clock::now();
            stats.optimizer += durationSince(tO0, tO1);

            if (step % logEvery == 0 || step == adamSteps - 1)
            {
                record(step);
            }
        }
    };

    {
        torch::optim::Adam opt1(
            std::vector<torch::Tensor>{z},
            torch::optim::AdamOptions(adamLr));
        adamLoop(opt1, 0, firstHalf);
    }
    {
        torch::optim::Adam opt2(
            std::vector<torch::Tensor>{z},
            torch::optim::AdamOptions(adamLr * 0.1));
        adamLoop(opt2, firstHalf, adamSteps);
    }

    // --- LBFGS phase ------------------------------------------------------
    // ONE optimizer.step() call: LBFGSOptions::max_iter bounds the total
    // number of internal iterations; tolerance_grad/change terminate early.
    // (Previously an outer loop of step() calls multiplied the budget.)
    if (lbfgsMaxIter > 0)
    {
        torch::optim::LBFGSOptions opts(1.0);
        opts.max_iter(lbfgsMaxIter);
        opts.history_size(lbfgsHistorySize);
        opts.tolerance_grad(1e-12);
        opts.tolerance_change(1e-14);

        torch::optim::LBFGS optimizer(std::vector<torch::Tensor>{z}, opts);

        auto closure = [&]() -> torch::Tensor
        {
            ++stats.closureCalls;
            optimizer.zero_grad();

            auto tR0 = Clock::now();
            auto psi = computePsi(z, previousStates);
            auto energy = lossTensor(psi);
            auto tR1 = Clock::now();
            stats.rayleigh += durationSince(tR0, tR1);

            auto tB0 = Clock::now();
            energy.backward();
            auto tB1 = Clock::now();
            stats.backward += durationSince(tB0, tB1);

            return energy;
        };

        auto tO0 = Clock::now();
        optimizer.step(closure);
        auto tO1 = Clock::now();
        stats.optimizer += durationSince(tO0, tO1);

        record(adamSteps + lbfgsMaxIter);
    }

    stats.print(stateIndex);

    return {computePsi(z.detach(), previousStates), history};
}

std::pair<torch::Tensor, std::vector<TrainingState>> EigenTraining::trainNeuralState
(
    CoordinateMLP& model,
    int stateIndex,
    const std::vector<torch::Tensor>& previousStates,
    const torch::Tensor& coords,
    bool enforceParity,
    int pretrainSteps,
    int adamSteps,
    double adamLr,
    int lbfgsMaxIter,
    int restarts,
    long baseSeed,
    int lbfgsHistorySize,
    int logEvery
)
{
    torch::Tensor bestPsi;
    double bestEnergy = INFINITY;
    int bestRestart = 0;
    std::vector<TrainingState> bestHistory;

    TimingStats stats;

    // Normalised coordinates ~[-1, 1] (1D: [N,1]; 2D: [N,2]).  Augmented
    // with a constant channel so the input tensor is [N, dim+1] = [coords, 1]
    // (see CoordinateMLP note).
    auto xi1d = (coords / (coords.abs().max().item<double>() + epsilon_));
    auto xi = torch::cat({xi1d, torch::ones_like(xi1d.select(1, 0)).unsqueeze(1)}, /*dim=*/1);

    for (int r = 0; r < restarts; r++)
    {
        torch::manual_seed(baseSeed + stateIndex * 1000 + r);
        model.initialize();

        // Negative-parity evaluation: flip ONLY the coordinate channel,
        // keeping the constant channel positive (otherwise the base network
        // is exactly odd again: -[xi, 1] = [-xi, -1]).
        auto xiNeg = torch::cat({-xi1d, torch::ones_like(xi1d.select(1, 0)).unsqueeze(1)}, /*dim=*/1);

        auto forwardPsi = [&]() -> torch::Tensor
        {
            auto raw = model.forward(xi).squeeze(1);
            if (enforceParity)
            {
                auto rawNeg = model.forward(xiNeg).squeeze(1);
                if (stateIndex % 2 == 0) raw = (raw + rawNeg) * 0.5;
                else                     raw = (raw - rawNeg) * 0.5;
            }
            return raw;
        };

        std::vector<TrainingState> history;
        auto t0 = Clock::now();

        auto record = [&](int step)
        {
            double gradMax = 0.0;
            for (const auto& p : model.parameters())
            {
                if (p.grad().defined())
                {
                    gradMax = std::max(gradMax, p.grad().abs().max().item<double>());
                }
            }

            auto psi = computePsi(forwardPsi().detach(), previousStates);
            auto t1 = Clock::now();

            TrainingState ts;
            ts.step = step;
            ts.energy = rayleighQuotient(psi);
            ts.massNorm = massNorm(psi);
            ts.gradNorm = gradMax;
            ts.orthogonalityError = orthogonalityError(psi, previousStates);
            ts.elapsedSeconds = durationSince(t0, t1);
            history.push_back(ts);
        };

        // One timed optimizer sweep: zero_grad assumed already handled
        auto sweep = [&](torch::optim::Optimizer& optimizer) -> double
        {
            auto tF0 = Clock::now();
            auto raw = forwardPsi();
            auto tF1 = Clock::now();
            stats.forward += durationSince(tF0, tF1);

            auto psi = computePsi(raw, previousStates);
            auto energy = lossTensor(psi);
            auto tR1 = Clock::now();
            stats.rayleigh += durationSince(tF1, tR1);

            double loss = energy.item<double>();
            if (!std::isfinite(loss)) return loss;

            auto tB0 = Clock::now();
            energy.backward();
            auto tB1 = Clock::now();
            stats.backward += durationSince(tB0, tB1);

            auto tO0 = Clock::now();
            optimizer.step();
            auto tO1 = Clock::now();
            stats.optimizer += durationSince(tO0, tO1);

            return loss;
        };

        bool diverged = false;

        int stepBase = 0;

        // --- Phase 1: pretraining Adam (10x LR) ---------------------------
        if (pretrainSteps > 0 && !diverged)
        {
            torch::optim::Adam optimizer(
                model.parameters(),
                torch::optim::AdamOptions(std::min(1.0, adamLr * 10.0)));

            for (int step = 0; step < pretrainSteps; step++)
            {
                optimizer.zero_grad();
                double loss = sweep(optimizer);
                if (!std::isfinite(loss)) { diverged = true; break; }

                if (step % logEvery == 0 || step == pretrainSteps - 1)
                {
                    record(stepBase + step);
                }
            }
            stepBase += pretrainSteps;
        }

        // --- Phase 2: main Adam -------------------------------------------
        if (!diverged)
        {
            torch::optim::Adam optimizer(
                model.parameters(),
                torch::optim::AdamOptions(adamLr));

            for (int step = 0; step < adamSteps; step++)
            {
                optimizer.zero_grad();
                double loss = sweep(optimizer);
                if (!std::isfinite(loss)) { diverged = true; break; }

                if (step % logEvery == 0 || step == adamSteps - 1)
                {
                    record(stepBase + step);
                }
            }
        }
        stepBase += adamSteps;

        // --- Phase 3: LBFGS ------------------------------------------------
        // ONE optimizer.step() call with max_iter bounding the total internal
        // iteration count (see coefficient-mode note above).
        if (!diverged && lbfgsMaxIter > 0)
        {
            torch::optim::LBFGSOptions opts(0.5);
            opts.max_iter(lbfgsMaxIter);
            opts.history_size(lbfgsHistorySize);
            opts.tolerance_grad(1e-12);
            opts.tolerance_change(1e-14);

            torch::optim::LBFGS optimizer(model.parameters(), opts);

            auto closure = [&]() -> torch::Tensor
            {
                ++stats.closureCalls;
                optimizer.zero_grad();

                auto tF0 = Clock::now();
                auto raw = forwardPsi();
                auto tF1 = Clock::now();
                stats.forward += durationSince(tF0, tF1);

                auto psi = computePsi(raw, previousStates);
                auto energy = lossTensor(psi);
                auto tR1 = Clock::now();
                stats.rayleigh += durationSince(tF1, tR1);

                auto tB0 = Clock::now();
                energy.backward();
                auto tB1 = Clock::now();
                stats.backward += durationSince(tB0, tB1);

                return energy;
            };

            auto tO0 = Clock::now();
            optimizer.step(closure);
            auto tO1 = Clock::now();
            stats.optimizer += durationSince(tO0, tO1);

            record(stepBase + lbfgsMaxIter);
        }

        if (!diverged && !history.empty())
        {
            auto psiFinal = computePsi(forwardPsi().detach(), previousStates);
            double finalEnergy = rayleighQuotient(psiFinal);

            if (finalEnergy < bestEnergy && std::isfinite(finalEnergy))
            {
                bestEnergy = finalEnergy;
                bestPsi = psiFinal;
                bestHistory = history;
                bestRestart = r;
            }
        }
    }

    std::cout << "  Best restart for state " << stateIndex << ": " << bestRestart
              << " with energy " << bestEnergy << std::endl;

    stats.print(stateIndex);

    return {bestPsi, bestHistory};
}
