#ifndef PhysicsTrainer_H
#define PhysicsTrainer_H

#include "TorchCompat.H"
#include "FvMeshData.H"
#include "DifferentiableFvResidual.H"
#include "ShallowFlowMLP.H"
#include <string>
#include <vector>
#include <map>

// Torch-only training/testing engine.  OpenFOAM never appears here; the
// driver passes everything in as plain FvMeshData and receives plain data
// back for field writing / CSV output.
class PhysicsTrainer
{
public:
    PhysicsTrainer(const FvMeshData& md, const PhysicsConfig& cfg);

    // Oracle path (Gate 3): dimensionless Stokes residuals for the same
    // values OpenFOAM's fvc evaluated, returned as plain vectors.
    void stokesResidualPlain(const std::vector<double>& u,   // [2N]
                             const std::vector<double>& p,   // [N]
                             std::vector<double>& rm,        // [2N] out
                             std::vector<double>& rc) const; // [N] out

    // Gate 2 gradient test: J = |U_ib|^2; AD vs centred FD on velocity
    // components feeding the interpolation map.
    bool ibGradientCheck(int samples, const std::vector<double>& epsilons);

    // Gate 5 gradient test: full path network -> fields -> fluxes -> HFDIB
    // -> loss; AD vs centred FD on >=10 parameters across layers.
    bool paramGradientCheck(ShallowFlowMLP& model, int sampleCount,
                            const std::vector<double>& epsilons);

    struct TrainReport
    {
        std::vector<std::string> csvLines;  // header not included
        std::vector<double> finalU;         // [2N] dimensionless
        std::vector<double> finalP;         // [N]
        double totalLoss;
        double momentumLoss;
        double continuityLoss;
        double ibLoss;
        DifferentiableFvResidual::Metrics finalMetrics;
        double elapsedSeconds;
    };

    TrainReport trainCoefficients(int adamSteps, double lr, int logEvery,
                                  long seed);
    TrainReport trainNeural(ShallowFlowMLP& model, int adamSteps, double lr,
                            int logEvery, long seed);

    const DifferentiableFvResidual& residual() const { return res_; }

private:
    static constexpr const char* kCsvColumns =
        "step,total,momentum,continuity,ib,rFull,maxSolid,maxIface,"
        "inletFlux,outletFlux,massImbalance,pressureDrop,gradNorm,elapsed";

    std::string metricsLine(int step, const DifferentiableFvResidual::Losses& L,
                            const DifferentiableFvResidual::Metrics& m,
                            double gradNorm, double elapsed) const;

    DifferentiableFvResidual res_;
    PhysicsConfig cfg_;
    int64_t n_;
};

#endif
