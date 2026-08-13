#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "IOdictionary.H"
#include "OFstream.H"
#include "OSspecific.H"
#include "LhoFvOperator.H"
#include "CoordinateMLP.H"
#include "EigenTraining.H"
#include "Diagnostics.H"
#include "TorchCompat.H"
#include <fstream>
#include <cstdlib>

using namespace Foam;

enum class RunMode { DIRECT, COEFFICIENTS, NEURAL };

int main(int argc, char* argv[])
{
    argList::addOption
    (
        "mode",
        "name",
        "Run mode: direct | coefficients | neural"
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    word caseDir = runTime.path();

    IOdictionary lhoProperties
    (
        IOobject
        (
            "lhoProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    scalar kineticScale = readScalar(lhoProperties.lookup("kineticScale"));
    scalar potentialScale = readScalar(lhoProperties.lookup("potentialScale"));
    int numberOfStates = readInt(lhoProperties.lookup("numberOfStates"));
    label dimension = 1;
    word potentialMode = "harmonic";
    if (lhoProperties.found("potentialMode"))
    {
        potentialMode = lhoProperties.lookupOrDefault<word>("potentialMode", "harmonic");
    }
    if (potentialMode == "anisotropic" || potentialMode == "laplacian")
    {
        dimension = 2;
    }
    scalar omegaY = lhoProperties.lookupOrDefault<scalar>("omegaY", 1.0);
    int hiddenWidth = readInt(lhoProperties.lookup("hiddenWidth"));
    int hiddenLayers = readInt(lhoProperties.lookup("hiddenLayers"));
    bool enforceParity = readBool(lhoProperties.lookup("enforceParity"));
    label baseSeed = readLabel(lhoProperties.lookup("baseSeed"));
    int restarts = readInt(lhoProperties.lookup("restarts"));
    int pretrainSteps = 150;
    if (lhoProperties.found("pretrainSteps"))
    {
        pretrainSteps = readInt(lhoProperties.lookup("pretrainSteps"));
    }
    int adamSteps = readInt(lhoProperties.lookup("adamSteps"));
    scalar adamLearningRate = readScalar(lhoProperties.lookup("adamLearningRate"));
    int lbfgsMaxIterations = readInt(lhoProperties.lookup("lbfgsMaxIterations"));
    // Coefficient mode (Gate 2) has no network preconditioning: Adam plateaus
    // quickly and LBFGS does the real convergence work, so it gets its own
    // explicit internal-iteration budget (historically ~20x lbfgsMaxIterations
    // due to the nested-step bug).
    int lbfgsCoefficientMaxIterations = lhoProperties.lookupOrDefault<int>(
        "lbfgsCoefficientMaxIterations", lbfgsMaxIterations);
    int lbfgsHistorySize = readInt(lhoProperties.lookup("lbfgsHistorySize"));
    int logEvery = readInt(lhoProperties.lookup("logEvery"));
    scalar epsilon = readScalar(lhoProperties.lookup("epsilon"));
    bool writeMatrices = readBool(lhoProperties.lookup("writeMatrices"));
    // The full dense O(N^3) eigendecomposition is validation, not part of the
    // training mechanism; make it skippable for large meshes.
    bool computeDirectReference =
        lhoProperties.lookupOrDefault<bool>("computeDirectReference", true);

    RunMode mode = RunMode::DIRECT;
    if (args.optionFound("mode"))
    {
        word modeStr = args.optionRead<word>("mode");
        if (modeStr == "direct") mode = RunMode::DIRECT;
        else if (modeStr == "coefficients") mode = RunMode::COEFFICIENTS;
        else if (modeStr == "neural") mode = RunMode::NEURAL;
    }

    Info << "\n=== OpenFOAM-LibTorch LHO Eigensolver ===" << nl;
    Info << "Kinetic scale: " << kineticScale << nl;
    Info << "Potential scale: " << potentialScale << nl;
    Info << "Potential mode: " << potentialMode << " (" << dimension << "D";
    if (potentialMode == "anisotropic") Info << ", omegaY=" << omegaY;
    Info << ")" << nl;
    Info << "Number of states: " << numberOfStates << nl;
    Info << "Mode: " << (mode == RunMode::DIRECT ? "direct" : 
                         mode == RunMode::COEFFICIENTS ? "coefficients" : "neural") << nl;
    Info << "==========================================" << nl << nl;

    torch::manual_seed(baseSeed);

    // Torch intra-op threads: the training loop is dominated by micro-ops
    // on tiny tensors, where intra-op threading costs more than it gains.
    // (The dense LAPACK eigendecomposition still parallelises via BLAS /
    // OMP_NUM_THREADS in env.sh.)
    int nThreads = 1;
    if (const char* s = std::getenv("TORCH_NUM_THREADS"))
    {
        nThreads = std::atoi(s);
    }
    torch::set_num_threads(nThreads);
    Info << "Torch intra-op threads: " << nThreads << nl;

    auto options = torch::TensorOptions()
        .dtype(torch::kFloat64)
        .device(torch::kCPU)
        .requires_grad(false);

    LhoFvOperator fvOp(mesh, dimension, potentialMode, omegaY, kineticScale, potentialScale);

    const auto& K = fvOp.K();
    const auto& M = fvOp.mass();
    const auto& x = fvOp.x();
    const auto& y = fvOp.y();

    fileName postProcDir = caseDir / "postProcessing" / "fvNeuralLHO";
    mkDir(postProcDir);

    if (writeMatrices)
    {
        fvOp.writeDiagnostics(postProcDir);
    }

    Diagnostics diag(K, M);

    std::vector<double> energiesDirect;
    std::vector<torch::Tensor> eigenstatesDirect;
    if (computeDirectReference)
    {
        std::tie(energiesDirect, eigenstatesDirect) =
            diag.solveDirectFV(numberOfStates);
    }
    else
    {
        Info << "computeDirectReference=false: skipping dense O(N^3) "
             << "direct eigensolve" << nl;
    }

    // Exact spectrum (harmonic => 1D n+0.5; anisotropic => sorted 2D;
    // laplacian => unit-disk Bessel zeros squared)
    std::vector<double> exactEnergies;
    if (potentialMode == "laplacian")
    {
        exactEnergies = Diagnostics::laplacianDiskSpectrum(numberOfStates);
    }
    else
    {
        exactEnergies = Diagnostics::exactSpectrum(numberOfStates, dimension, omegaY);
    }

    if (computeDirectReference)
    {
        Info << "\n=== Direct FV Eigenvalues ===" << nl;
        for (int n = 0; n < std::min((int)eigenstatesDirect.size(), numberOfStates); n++)
        {
            double E_exact = exactEnergies[n];
            Info << "State " << n << ": E_FV = " << energiesDirect[n]
                 << ", E_exact = " << E_exact
                 << ", error = " << std::abs(energiesDirect[n] - E_exact) << nl;
        }
    }

    if (mode == RunMode::DIRECT)
    {
        Info << "\nDirect mode complete." << nl;
        return 0;
    }

    std::vector<torch::Tensor> neuralStates;
    std::vector<std::vector<TrainingState>> trainingHistories;

    if (mode == RunMode::COEFFICIENTS || mode == RunMode::NEURAL)
    {
        // Rayleigh numerator psi^T K psi via the O(nFaces) face stencil
        // (identical to K.matmul(psi).dot(psi); checked at assembly).
        EigenTraining trainer(
            [&fvOp](const torch::Tensor& psi)
            {
                return fvOp.rayleighNumerator(psi);
            },
            M, numberOfStates, epsilon);

        // Normalised coordinate tensor [N, dim]: 1D = [x], 2D = [x, y]
        torch::Tensor coords;
        if (dimension == 2)
        {
            coords = torch::cat({x.unsqueeze(1), y.unsqueeze(1)}, /*dim=*/1);
        }
        else
        {
            coords = x.unsqueeze(1);
        }

        for (int n = 0; n < numberOfStates; n++)
        {
            Info << "\n=== Training state " << n << " ===" << nl;

            torch::Tensor psi_n;
            std::vector<TrainingState> history;

            if (mode == RunMode::COEFFICIENTS)
            {
                std::tie(psi_n, history) = trainer.trainCoefficientState
                (
                    n,
                    neuralStates,
                    adamSteps,
                    adamLearningRate,
                    lbfgsCoefficientMaxIterations,
                    lbfgsHistorySize,
                    logEvery
                );
            }
            else
            {
                CoordinateMLP model(dimension + 1, hiddenWidth, hiddenLayers);
                std::tie(psi_n, history) = trainer.trainNeuralState
                (
                    model,
                    n,
                    neuralStates,
                    coords,
                    enforceParity,
                    pretrainSteps,
                    adamSteps,
                    adamLearningRate,
                    lbfgsMaxIterations,
                    restarts,
                    baseSeed,
                    lbfgsHistorySize,
                    logEvery
                );
            }

            neuralStates.push_back(psi_n);
            trainingHistories.push_back(history);

            double E_NN = trainer.rayleighQuotient(psi_n);

            Info << "State " << n << " complete:" << nl;
            Info << "  E_NN = " << E_NN << nl;
            if (computeDirectReference)
            {
                // Note: for degenerate states (e.g. unit-disk doublets), a
                // state-by-state overlap is basis-dependent and NOT a valid
                // accuracy measure; use the subspace diagnostics in
                // scripts/plot_eigenfunctions_2d.py instead.
                double overlap = std::abs(diag.computeOverlap(psi_n, eigenstatesDirect[n]));
                double fieldError = diag.computeFieldError(psi_n, eigenstatesDirect[n]);

                Info << "  E_direct = " << energiesDirect[n] << nl;
                Info << "  |E_NN - E_direct| = " << std::abs(E_NN - energiesDirect[n]) << nl;
                Info << "  Overlap with direct = " << overlap << nl;
                Info << "  Field error = " << fieldError << nl;
            }
        }
    }

    if (!neuralStates.empty())
    {
        Info << "\n=== Writing output fields ===" << nl;

        volScalarField potential
        (
            IOobject
            (
                "potential",
                Foam::Time::timeName(runTime.value()),
                mesh,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh,
            dimensionedScalar("zero", dimless, 0)
        );

        for (Foam::label cellI = 0; cellI < x.size(0); cellI++)
        {
            potential[cellI] = fvOp.potential()[cellI].item<double>();
        }
        potential.correctBoundaryConditions();
        potential.write();

        for (int n = 0; n < std::min((int)neuralStates.size(), numberOfStates); n++)
        {
            word psiNNName = "psiNN_" + std::to_string(n);
            volScalarField psiNN
            (
                IOobject
                (
                    psiNNName,
                    Foam::Time::timeName(runTime.value()),
                    mesh,
                    IOobject::NO_READ,
                    IOobject::AUTO_WRITE
                ),
                mesh,
                dimensionedScalar("zero", dimless, 0)
            );

            for (Foam::label cellI = 0; cellI < neuralStates[n].size(0); cellI++)
            {
                psiNN[cellI] = neuralStates[n][cellI].item<double>();
            }
            psiNN.correctBoundaryConditions();
            psiNN.write();

            if (computeDirectReference)
            {
                word psiDirectName = "psiDirectFV_" + std::to_string(n);
                volScalarField psiDirect
                (
                    IOobject
                    (
                        psiDirectName,
                        Foam::Time::timeName(runTime.value()),
                        mesh,
                        IOobject::NO_READ,
                        IOobject::AUTO_WRITE
                    ),
                    mesh,
                    dimensionedScalar("zero", dimless, 0)
                );

                for (Foam::label cellI = 0; cellI < eigenstatesDirect[n].size(0); cellI++)
                {
                    psiDirect[cellI] = eigenstatesDirect[n][cellI].item<double>();
                }
                psiDirect.correctBoundaryConditions();
                psiDirect.write();
            }
        }

        fileName csvPath = postProcDir / "eigenvalues.csv";
        std::vector<double> energiesNN;
        for (const auto& psi : neuralStates)
        {
            // Face-stencil M-weighted Rayleigh quotient (O(nFaces))
            double e = (fvOp.rayleighNumerator(psi).item<double>()
                        / ((M * psi * psi).sum() + epsilon).item<double>());
            energiesNN.push_back(e);
        }
        diag.writeEigenvaluesCSV(csvPath, energiesNN, energiesDirect, exactEnergies, mesh.nCells());

        // Training histories as CSV
        for (int n = 0; n < std::min((int)trainingHistories.size(), numberOfStates); n++)
        {
            fileName trainPath =
                postProcDir / ("training_state_" + std::to_string(n) + ".csv");
            OFstream ths(trainPath);
            ths << "step,energy,massNorm,gradNorm,orthogonalityError,elapsedSeconds"
                << endl;
            for (const auto& ts : trainingHistories[n])
            {
                ths << ts.step << ","
                    << ts.energy << ","
                    << ts.massNorm << ","
                    << ts.gradNorm << ","
                    << ts.orthogonalityError << ","
                    << ts.elapsedSeconds << endl;
            }
        }

        fileName profilesPath = postProcDir / "profiles.csv";
        {
            OFstream os(profilesPath);
            os << "cell,x";
            if (dimension == 2) os << ",y";
            os << ",volume,potential";
            for (int n = 0; n < std::min((int)neuralStates.size(), numberOfStates); n++)
            {
                os << ",psiNN_" << n;
                if (computeDirectReference) os << ",psiDirectFV_" << n;
            }
            os << endl;

            for (Foam::label cellI = 0; cellI < x.size(0); cellI++)
            {
                os << cellI << ","
                   << x[cellI].item<double>() << ",";
                if (dimension == 2)
                {
                    os << y[cellI].item<double>() << ",";
                }
                os << M[cellI].item<double>() << ","
                   << fvOp.potential()[cellI].item<double>();

                for (int n = 0; n < std::min((int)neuralStates.size(), numberOfStates); n++)
                {
                    os << "," << neuralStates[n][cellI].item<double>();
                    if (computeDirectReference)
                    {
                        os << "," << eigenstatesDirect[n][cellI].item<double>();
                    }
                }
                os << endl;
            }
        }
    }

    Info << "\n=== Run complete ===" << nl;
    return 0;
}
