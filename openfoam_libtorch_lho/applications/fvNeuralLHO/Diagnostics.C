#include "Diagnostics.H"
#include "EigenTraining.H"
#include "OFstream.H"
#include <cmath>
#include <algorithm>
#include <utility>

using namespace Foam;

Diagnostics::Diagnostics(const torch::Tensor& K, const torch::Tensor& M)
:
    K_(K),
    M_(M)
{
}

std::pair<std::vector<double>, std::vector<torch::Tensor>>
Diagnostics::solveDirectFV(int numStates)
{
    // H = M^{-1/2} K M^{-1/2}:  M_ holds the diagonal mass vector [N]
    auto invSqrtMass = M_.rsqrt();

    auto H = invSqrtMass.unsqueeze(1) * K_ * invSqrtMass.unsqueeze(0);

    auto result = torch::linalg_eigh(H);
    auto eigenvalues = std::get<0>(result);
    auto eigenvectorsY = std::get<1>(result);

    std::vector<double> energies;
    std::vector<torch::Tensor> eigenstates;

    for (int i = 0; i < numStates && i < eigenvalues.size(0); i++)
    {
        double E = eigenvalues[i].item<double>();
        energies.push_back(E);

        auto y_i = eigenvectorsY.select(1, i);
        auto psi_i = invSqrtMass * y_i;

        // Explicit M-normalisation (redundant but harmless)
        double normSq = (M_ * psi_i * psi_i).sum().item<double>();
        psi_i = psi_i / std::sqrt(normSq);

        eigenstates.push_back(psi_i.detach());
    }

    return {energies, eigenstates};
}

torch::Tensor Diagnostics::hermitePolynomial(int n, const torch::Tensor& x)
{
    if (n == 0)
    {
        return torch::ones_like(x);
    }
    else if (n == 1)
    {
        return 2.0 * x;
    }

    auto H_prev2 = torch::ones_like(x);
    auto H_prev1 = 2.0 * x.clone();

    for (int k = 1; k < n; k++)
    {
        auto H_curr = 2.0 * x * H_prev1 - 2.0 * k * H_prev2;
        H_prev2 = H_prev1.clone();
        H_prev1 = H_curr.clone();
    }

    return H_prev1;
}

torch::Tensor Diagnostics::analyticalEigenfunction(int n, const torch::Tensor& x)
{
    auto H_n = hermitePolynomial(n, x);

    double normFactor =
        std::pow(M_PI, 0.25) * std::sqrt(std::pow(2, n) * std::tgamma(n + 1));

    auto expTerm = torch::exp(-x.square() / 2.0);
    return H_n * expTerm / normFactor;
}

std::vector<double> Diagnostics::exactSpectrum(int numStates, double omegaY)
{
    if (omegaY <= 1.0 + 1e-14)
    {
        // 1D harmonic oscillator (also isotropic-2D ordered list happens to be
        // n + 0.5 only in the non-degenerate 1D case).
        std::vector<double> E(numStates);
        for (int n = 0; n < numStates; n++) E[n] = n + 0.5;
        return E;
    }

    // 2D anisotropic: generate pairs (nx, ny) and sort ascending.
    std::vector<std::pair<double, int>> entries;
    int maxN = 4 * numStates + 8;
    for (int nx = 0; nx <= maxN; nx++)
    {
        for (int ny = 0; ny <= maxN; ny++)
        {
            double E = (nx + 0.5) + omegaY * (ny + 0.5);
            int tot = nx + ny;
            entries.emplace_back(E, tot);
        }
    }
    std::sort(entries.begin(), entries.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<double> E(numStates);
    for (int i = 0; i < numStates && i < (int)entries.size(); i++)
    {
        E[i] = entries[i].first;
    }
    return E;
}

std::vector<double> Diagnostics::laplacianDiskSpectrum(int numStates)
{
    // Squares of Bessel-function zeros j_{m,k}^2 (distinct, ascending).
    static const double diskEigs[] = {
        5.783185962947,
        14.681970642124,
        26.374616427163,
        30.471262343662,
        40.706465818200,
        49.218456321695,
        57.582940903291,
        70.849998919096,
        74.887006790695,
        76.938928333647,
        95.277572544037,
        98.726272477249,
        103.499453895137,
        122.427796064928,
        122.907600203616,
        135.020708865970
    };

    std::vector<double> E;
    for (int i = 0; i < numStates && i < (int)(sizeof(diskEigs) / sizeof(double)); i++)
    {
        E.push_back(diskEigs[i]);
    }
    return E;
}

double Diagnostics::computeOverlap
(
    const torch::Tensor& psi1,
    const torch::Tensor& psi2
)
{
    return (M_ * psi1 * psi2).sum().item<double>();
}

double Diagnostics::computeFieldError
(
    const torch::Tensor& psi1,
    const torch::Tensor& psi2
)
{
    // Eigenfunctions have arbitrary sign: align via M-overlap first
    double overlap = computeOverlap(psi1, psi2);
    auto diff = (overlap < 0) ? (psi1 + psi2) : (psi1 - psi2);
    return std::sqrt((M_ * diff * diff).sum().item<double>());
}

double Diagnostics::computeResidual
(
    const torch::Tensor& Kpsi,
    const torch::Tensor& Mpsi,
    const torch::Tensor& psi,
    double E
)
{
    auto r = Kpsi.matmul(psi) - E * Mpsi * psi;
    return r.norm().item<double>();
}

void Diagnostics::writeEigenvaluesCSV
(
    const Foam::fileName& path,
    const std::vector<double>& energiesNN,
    const std::vector<double>& energiesDirect,
    const std::vector<double>& energiesExact,
    int N
)
{
    OFstream os(path);
    os << "state,E_NN,E_directFV,E_exact,dE_NN_direct,dE_direct_exact,dE_NN_exact"
       << endl;

    for (size_t n = 0; n < std::min(energiesNN.size(), energiesDirect.size()); n++)
    {
        double E_exact = (n < energiesExact.size()) ? energiesExact[n] : (n + 0.5);
        double dNN_Direct = std::abs(energiesNN[n] - energiesDirect[n]);
        double dDirect_Exact = std::abs(energiesDirect[n] - E_exact);
        double dNN_Exact = std::abs(energiesNN[n] - E_exact);

        os << n << ","
           << energiesNN[n] << ","
           << energiesDirect[n] << ","
           << E_exact << ","
           << dNN_Direct << ","
           << dDirect_Exact << ","
           << dNN_Exact << endl;
    }
}
