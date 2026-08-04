#ifndef DifferentiableFvResidual_H
#define DifferentiableFvResidual_H

#include "TorchCompat.H"
#include "FvMeshData.H"
#include <vector>
#include <string>

// Torch side of the demo: dimensionless finite-volume Stokes/HFDIB residual
// evaluated from tensor ops only (no dense matrices anywhere).
//
// Scaling: the ctor scales lengths by 1/Lref. Areas scale L^-2, volumes
// L^-3, distances L^-1. Velocities in Uref units, pressure in
// Pref = nu*Uref/Lref units, so the momentum residual is dimensionless in
// units of nu*Uref/Lref^2 and the continuity residual in Uref/Lref.
struct PhysicsConfig
{
    double nu = 0.01;
    double inletU = 0.1;
    double Uref = 0.1;
    double Lref = 0.128;
    double gammaContinuity = 1.0;
    double gammaIb = 10.0;

    double nuStar() const { return nu / (Uref * Lref); }
    double pRef() const { return nu * Uref / Lref; }
};

class DifferentiableFvResidual
{
public:
    DifferentiableFvResidual(const FvMeshData& md, const PhysicsConfig& cfg);

    // HFDIB sparse map: u_ib values for the K target cells [K, 2]
    torch::Tensor applyIb(const torch::Tensor& U) const;

    // Full-N field with interface cells REPLACED by their HFDIB values and
    // solid cells zeroed (fluid cells keep the original value).
    torch::Tensor imposedField(const torch::Tensor& U) const;

    // Dimensionless cell residuals:
    //   Rm [N,2] = -(1/V) sum_f nu* (u_ngh - u_own)|Sf|/d   (+ boundary rules)
    //              + (1/V) sum_f p_f Sf
    //   Rc [N]   = (1/V) sum_f u_f . Sf
    std::pair<torch::Tensor, torch::Tensor> stokesResidual(
        const torch::Tensor& U, const torch::Tensor& p) const;

    struct Losses
    {
        torch::Tensor total;
        torch::Tensor momentum;
        torch::Tensor continuity;
        torch::Tensor ib;
        torch::Tensor rFull;  // paper forced-balance diagnostic norm
    };
    Losses losses(const torch::Tensor& U, const torch::Tensor& p) const;

    struct Metrics
    {
        double maxSolidSpeed;
        double maxInterfaceError;
        double inletFlux;
        double outletFlux;
        double massImbalance;   // |out - in| / in
        double pressureDrop;    // mean p@inlet owners - mean p@outlet owners
        double rFullNorm;
    };
    Metrics metrics(const torch::Tensor& U, const torch::Tensor& p) const;

    // dimensionless feature tensor for the shallow MLP [N, 7]:
    // (x*, y*, lambda, sigma/h, fluidMask, interfaceMask, solidMask)
    const torch::Tensor& features() const { return features_; }

    int64_t nCells() const { return nCells_; }
    double inletUStar() const { return inletUStar_; }

private:
    int64_t nCells_;
    double nuStar_;
    double inletUStar_;   // dimensionless inlet speed (= inletU/Uref)
    double cfgGammaC_ = 1.0;
    double cfgGammaIb_ = 10.0;
    torch::Tensor xStar_;         // scaled cell centres [N, 2]
    torch::Tensor vol_;             // [N]
    torch::Tensor owners_, neighs_; // [F] int64
    torch::Tensor sfInt_;           // [F, 2]
    torch::Tensor magSfInt_;        // [F]
    torch::Tensor deltaInt_;        // [F]
    int64_t nIn_, nOut_, nWall_;    // boundary role counts (concatenated)
    torch::Tensor bndOwner_;        // [B] int64
    torch::Tensor bndSf_;           // [B, 2]
    torch::Tensor bndMagSf_;        // [B]
    torch::Tensor bndDelta_;        // [B]
    torch::Tensor fluidMask_, interfaceMask_, solidMask_;  // [N]
    torch::Tensor chiMask_;         // [N] ceil(lambda): 1 on solid+interface
    torch::Tensor ibTarget_;        // [K] int64
    torch::Tensor ibRowIdx_;        // [M] int64 (row of each term)
    std::vector<int64_t> ibRowStartHost_;
    torch::Tensor ibSrc_;           // [M] int64
    torch::Tensor ibW_;             // [M]
    torch::Tensor features_;        // [N, 7]
    torch::Tensor inletOwnerIdx_;   // [nIn]
    torch::Tensor outletOwnerIdx_;  // [nOut]
    torch::Tensor outletSf_;        // [nOut, 2]
};

#endif
