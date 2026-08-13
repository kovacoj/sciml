#include "LhoFvOperator.H"
#include "fvc.H"
#include "IFstream.H"
#include "OFstream.H"
#include "surfaceFields.H"
#include "OSspecific.H"
#include <iostream>
#include <cmath>
#include <vector>

using namespace Foam;

LhoFvOperator::LhoFvOperator
(
    const fvMesh& mesh,
    label dimension,
    const word& potentialMode,
    scalar omegaY,
    scalar kineticScale,
    scalar potentialScale
)
:
    mesh_(mesh),
    dim_(dimension),
    potentialMode_(potentialMode),
    omegaY_(omegaY),
    kineticScale_(kineticScale),
    potentialScale_(potentialScale),
    nCells_(mesh.nCells()),
    K_(torch::zeros({nCells_, nCells_}, torch::kFloat64)),
    mass_(torch::zeros(nCells_, torch::kFloat64)),
    x_(torch::zeros(nCells_, torch::kFloat64)),
    y_(torch::zeros(nCells_, torch::kFloat64)),
    potential_(torch::zeros(nCells_, torch::kFloat64))
{
    Info << "Assembling finite-volume Hamiltonian (" << potentialMode_
         << ", " << dim_ << "D) for " << nCells_ << " cells" << nl;

    assembleKinetic();
    assemblePotential();

    // Validate
    auto symmetryError = (K_ - K_.transpose(0, 1)).abs().max().item<double>();
    Info << "Matrix symmetry error: " << symmetryError << nl;

    if (symmetryError > 1e-12)
    {
        FatalErrorInFunction
            << "K is not symmetric to required tolerance"
            << exit(FatalError);
    }

    // Check for positive volumes and finite values
    auto minMass = mass_.min().item<double>();
    if (minMass <= 0)
    {
        FatalErrorInFunction
            << "Non-positive cell volume detected: " << minMass
            << exit(FatalError);
    }

    auto kAbsMax = K_.abs().max().item<double>();
    if (!std::isfinite(kAbsMax))
    {
        FatalErrorInFunction
            << "Non-finite value in K matrix"
            << exit(FatalError);
    }

    // Cross-check the face-based quadratic form against the dense matrix on a
    // deterministic probe vector (regression guard for the training path).
    {
        auto probe = torch::zeros({nCells_}, torch::kFloat64);
        for (label c = 0; c < nCells_; ++c)
        {
            probe[c] = 1.0 + static_cast<double>(c % 7);
        }
        const double qDense = probe.dot(K_.matmul(probe)).item<double>();
        const double qFace = rayleighNumerator(probe).item<double>();
        const double relErr = std::abs(qFace - qDense)
                            / (std::abs(qDense) + SMALL);
        Info << "Face quadratic form vs dense K: relative error " << relErr << nl;
        if (relErr > 1e-12)
        {
            FatalErrorInFunction
                << "Face-based quadratic form disagrees with dense K"
                << exit(FatalError);
        }
    }

    Info << "Finite-volume operator assembled successfully" << nl;
}

torch::Tensor LhoFvOperator::rayleighNumerator(const torch::Tensor& psi) const
{
    // Dirichlet energy psi^T K psi assembled from the face stencil:
    //   sum_int g_f (psi_n - psi_o)^2  +  sum_bnd g_b psi_c^2
    //   + sum_c m_c V_c psi_c^2  (potential, zero for "laplacian" mode)
    auto jumps = psi.index_select(0, faceNeighbour_)
               - psi.index_select(0, faceOwner_);
    auto energy = (faceConductance_ * jumps.square()).sum();

    if (boundaryCell_.numel() > 0)
    {
        auto psiB = psi.index_select(0, boundaryCell_);
        energy = energy + (boundaryConductance_ * psiB.square()).sum();
    }

    energy = energy + (mass_ * potential_ * psi.square()).sum();
    return energy;
}

void LhoFvOperator::assembleKinetic()
{
    const auto& Sf = mesh_.Sf();
    const auto& Cc = mesh_.C();
    const auto& CfFaces = mesh_.Cf();
    const auto& owner = mesh_.owner();
    const auto& neighbour = mesh_.neighbour();

    // Internal faces
    std::vector<int64_t> ownerIdx(owner.size());
    std::vector<int64_t> neighbourIdx(neighbour.size());
    std::vector<double> intConductance(owner.size());

    forAll(owner, faceI)
    {
        label o = owner[faceI];
        label n = neighbour[faceI];

        vector Sf_vec = Sf[faceI];
        scalar A_f = mag(Sf_vec);
        vector n_f = Sf_vec / A_f;

        vector Co = Cc[o];
        vector Cn = Cc[n];
        scalar d_f = mag((Cn - Co) & n_f);

        if (d_f <= SMALL)
        {
            FatalErrorInFunction
                << "Zero or negative face distance detected"
                << exit(FatalError);
        }

        scalar g_f = kineticScale_ * A_f / d_f;

        K_[o][o] += g_f;
        K_[n][n] += g_f;
        K_[o][n] -= g_f;
        K_[n][o] -= g_f;

        ownerIdx[faceI] = o;
        neighbourIdx[faceI] = n;
        intConductance[faceI] = g_f;
    }

    auto intOptsI = torch::TensorOptions().dtype(torch::kInt64);
    auto intOptsD = torch::TensorOptions().dtype(torch::kFloat64);
    faceOwner_ = torch::from_blob(
        ownerIdx.data(), {(int64_t)ownerIdx.size()}, intOptsI).clone();
    faceNeighbour_ = torch::from_blob(
        neighbourIdx.data(), {(int64_t)neighbourIdx.size()}, intOptsI).clone();
    faceConductance_ = torch::from_blob(
        intConductance.data(), {(int64_t)intConductance.size()}, intOptsD).clone();

    // Boundary faces
    std::vector<int64_t> bndCellIdx;
    std::vector<double> bndConductance;

    const auto& boundary = mesh_.boundary();
    forAll(boundary, patchI)
    {
        const auto& patch = boundary[patchI];
        const word& patchType = patch.type();

        // Only process left and right patches (Dirichlet)
        if (patchType == "patch")
        {
            forAll(patch, i)
            {
                label faceI = patch.start() + i;
                label c = patch.faceCells()[i];

                vector Sf_vec = Sf[faceI];
                scalar A_f = mag(Sf_vec);
                vector n_f = Sf_vec / A_f;

                vector Cfv = CfFaces[faceI];
                vector CcC = Cc[c];
                scalar d_b = mag((Cfv - CcC) & n_f);

                if (d_b <= SMALL)
                {
                    FatalErrorInFunction
                        << "Zero boundary distance on patch " << patch.name()
                        << exit(FatalError);
                }

                scalar g_b = kineticScale_ * A_f / d_b;
                K_[c][c] += g_b;

                bndCellIdx.push_back(c);
                bndConductance.push_back(g_b);
            }
        }
    }

    auto bndOptsI = torch::TensorOptions().dtype(torch::kInt64);
    auto bndOptsD = torch::TensorOptions().dtype(torch::kFloat64);
    if (bndCellIdx.empty())
    {
        boundaryCell_ = torch::empty({0}, bndOptsI);
        boundaryConductance_ = torch::empty({0}, bndOptsD);
    }
    else
    {
        boundaryCell_ = torch::from_blob(
            bndCellIdx.data(), {(int64_t)bndCellIdx.size()}, bndOptsI).clone();
        boundaryConductance_ = torch::from_blob(
            bndConductance.data(), {(int64_t)bndConductance.size()}, bndOptsD).clone();
    }
}

void LhoFvOperator::assemblePotential()
{
    const auto& C = mesh_.C();
    const auto& V = mesh_.V();

    if (potentialMode_ == "laplacian")
    {
        // Pure -Laplacian: no potential; record coordinates and mass only.
        for (label c = 0; c < nCells_; c++)
        {
            x_[c] = C[c].x();
            y_[c] = C[c].y();
            mass_[c] = V[c];
            potential_[c] = 0.0;
        }
    }
    else if (potentialMode_ == "anisotropic")
    {
        for (label c = 0; c < nCells_; c++)
        {
            scalar x_c = C[c].x();
            scalar y_c = C[c].y();
            scalar V_c = V[c];

            x_[c] = x_c;
            y_[c] = y_c;
            mass_[c] = V_c;
            scalar pot = potentialScale_ * (x_c * x_c + omegaY_ * omegaY_ * y_c * y_c);
            potential_[c] = pot;

            K_[c][c] += pot * V_c;
        }
    }
    else // harmonic (1D)
    {
        for (label c = 0; c < nCells_; c++)
        {
            scalar x_c = C[c].x();
            scalar V_c = V[c];

            x_[c] = x_c;
            mass_[c] = V_c;
            potential_[c] = potentialScale_ * x_c * x_c;

            K_[c][c] += potential_[c] * V_c;
        }
    }
}

void LhoFvOperator::writeDiagnostics(const fileName& path) const
{
    Foam::mkDir(path);

    fileName kFile = path / "K.csv";
    fileName massFile = path / "mass.csv";
    fileName xFile = path / "x.csv";

    // Only write matrices for small meshes
    if (nCells_ <= 64)
    {
        OFstream kOs(kFile);
        kOs << "i,j,K_ij" << endl;
        for (label i = 0; i < nCells_; i++)
        {
            for (label j = 0; j < nCells_; j++)
            {
                kOs << i << "," << j << "," << K_[i][j].item<double>() << endl;
            }
        }

        OFstream massOs(massFile);
        massOs << "cell,mass" << endl;
        for (label c = 0; c < nCells_; c++)
        {
            massOs << c << "," << mass_[c].item<double>() << endl;
        }

        OFstream xOs(xFile);
        xOs << "cell,x" << endl;
        for (label c = 0; c < nCells_; c++)
        {
            xOs << c << "," << x_[c].item<double>() << endl;
        }

        if (dim_ == 2)
        {
            fileName yFile = path / "y.csv";
            OFstream yOs(yFile);
            yOs << "cell,y" << endl;
            for (label c = 0; c < nCells_; c++)
            {
                yOs << c << "," << y_[c].item<double>() << endl;
            }
        }
    }
    else
    {
        Info << "Skipping dense matrix output for N=" << nCells_ << nl;
    }
}
