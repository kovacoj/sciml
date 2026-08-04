#ifndef Diagnostics_H
#define Diagnostics_H

#include "fileName.H"
#include "TorchCompat.H"
#include <string>
#include <vector>

struct TrainingState;

class Diagnostics
{
public:
    Diagnostics(const torch::Tensor& K, const torch::Tensor& M);

    // Direct FV eigensolver using M^(-1/2) transformation
    std::pair<std::vector<double>, std::vector<torch::Tensor>> solveDirectFV(int numStates = 6);

    // Sorted exact oscillator spectrum for the first `numStates` states.
    // dimension 1: E_n = n + 0.5
    // dimension 2: E = (nx + 0.5) + omegaY*(ny + 0.5), sorted ascending.
    //              For omegaY == 1 this is the isotropic 2D spectrum
    //              E = nx + ny + 1 with multiplicities 1, 2, 3, ...
    static std::vector<double> exactSpectrum(
        int numStates,
        int dimension,
        double omegaY = 1.0);

    // Dirichlet eigenvalues of -Laplacian on the unit disk: squares of Bessel
    // zeros j_{m,k}^2 (distinct values, sorted ascending).
    static std::vector<double> laplacianDiskSpectrum(int numStates);

    // Analytical harmonic oscillator eigenfunctions (physicists' Hermite polynomials)
    torch::Tensor analyticalEigenfunction(int n, const torch::Tensor& x);

    // Compute overlap between two states in M inner product
    double computeOverlap(const torch::Tensor& psi1, const torch::Tensor& psi2);

    // Compute mass-normalized field error  
    double computeFieldError(const torch::Tensor& psi1, const torch::Tensor& psi2);

    // Compute residual ||Kψ - EMψ||_2
    double computeResidual(const torch::Tensor& K, const torch::Tensor& M,
                          const torch::Tensor& psi, double E);

    // Write CSV files
    void writeEigenvaluesCSV(const Foam::fileName& path, 
                            const std::vector<double>& energiesNN,
                            const std::vector<double>& energiesDirect,
                            const std::vector<double>& energiesExact,
                            int N);

private:
    torch::Tensor K_;
    torch::Tensor M_;
    torch::Tensor hermitePolynomial(int n, const torch::Tensor& x);
};

#endif
