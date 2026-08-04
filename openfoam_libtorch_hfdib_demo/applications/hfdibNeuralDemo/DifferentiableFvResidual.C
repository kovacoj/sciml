#include "DifferentiableFvResidual.H"
#include <cmath>
#include <stdexcept>
#include <cstring>

// NOTE: Torch-only translation unit.  No OpenFOAM headers.

static torch::Tensor toD(const std::vector<double>& v)
{
    return torch::from_blob(
        const_cast<double*>(v.data()), {(int64_t)v.size()}, torch::kFloat64)
        .clone();
}

static torch::Tensor toI(const std::vector<int64_t>& v)
{
    return torch::from_blob(
        const_cast<int64_t*>(v.data()), {(int64_t)v.size()}, torch::kInt64)
        .clone();
}

DifferentiableFvResidual::DifferentiableFvResidual(
    const FvMeshData& md, const PhysicsConfig& cfg)
:
    nCells_(static_cast<int64_t>(md.nCells)),
    nuStar_(cfg.nuStar()),
    inletUStar_(cfg.inletU / cfg.Uref),
    cfgGammaC_(cfg.gammaContinuity),
    cfgGammaIb_(cfg.gammaIb)
{
    const double s = 1.0 / cfg.Lref;  // length scale

    // --- geometry, scaled to dimensionless space --------------------------
    {
        std::vector<double> v(md.cellVolumes);
        for (auto& x : v) x *= s * s * s;
        vol_ = toD(v);
        std::vector<double> m(md.faceMagSfInt);
        for (auto& x : m) x *= s * s;
        magSfInt_ = toD(m);
        std::vector<double> d(md.faceDeltaInt);
        for (auto& x : d) x *= s;
        deltaInt_ = toD(d);
        std::vector<double> sfa(md.faceAreaVecInt);
        for (auto& x : sfa) x *= s * s;
        sfInt_ = toD(sfa).view({-1, 2});
        std::vector<double> bsfa(md.bndFaceAreaVec);
        for (auto& x : bsfa) x *= s * s;
        bndSf_ = toD(bsfa).view({-1, 2});
        std::vector<double> bm(md.bndFaceMagSf);
        for (auto& x : bm) x *= s * s;
        bndMagSf_ = toD(bm);
        std::vector<double> bd(md.bndDelta);
        for (auto& x : bd) x *= s;
        bndDelta_ = toD(bd);
    }
    owners_ = toI(md.owners);
    neighs_ = toI(md.neighbours);
    bndOwner_ = toI(md.bndOwner);
    nIn_ = static_cast<int64_t>(md.inletFaces.size());
    nOut_ = static_cast<int64_t>(md.outletFaces.size());
    nWall_ = static_cast<int64_t>(md.wallFaces.size());

    // per-role boundary slices (concatenation order: inlet, outlet, walls)
    inletOwnerIdx_ = bndOwner_.slice(0, 0, nIn_);
    outletOwnerIdx_ = bndOwner_.slice(0, nIn_, nIn_ + nOut_);
    outletSf_ = bndSf_.slice(0, nIn_, nIn_ + nOut_);

    // fixed-U boundary mask: 1 for inlet+wall rows, 0 for outlet rows
    {
        std::vector<double> fm(nIn_ + nOut_ + nWall_, 1.0);
        for (int64_t i = nIn_; i < nIn_ + nOut_; ++i) fm[i] = 0.0;
        bndFixedMask_ = toD(fm);
    }

    // --- masks / HFDIB ----------------------------------------------------
    fluidMask_ = toD(md.fluidMask);
    interfaceMask_ = toD(md.interfaceMask);
    solidMask_ = toD(md.solidMask);
    chiMask_ = interfaceMask_ + solidMask_;  // ceil(lambda) in {0,1}

    ibTarget_ = toI(md.ibTargetCells);
    ibSrc_ = toI(md.ibSourceCells);
    ibW_ = toD(md.ibWeights);
    {
        std::vector<int64_t> rowIdx(md.ibWeights.size());
        for (std::size_t k = 0; k + 1 < md.ibRowStart.size(); ++k)
        {
            for (int64_t j = md.ibRowStart[k]; j < md.ibRowStart[k + 1]; ++j)
            {
                rowIdx[static_cast<std::size_t>(j)] = static_cast<int64_t>(k);
            }
        }
        ibRowIdx_ = toI(rowIdx);
    }

    // --- NN feature tensor [N, 7] ----------------------------------------
    {
        auto cc = toD(md.cellCenters);  // [2N] dimensional
        std::vector<double> feat(md.nCells * 7);
        for (std::size_t c = 0; c < md.nCells; ++c)
        {
            const double hEff = std::cbrt(md.cellVolumes[c]);
            // coordinates centred to roughly [-1, 1]
            feat[c * 7 + 0] = 2.0 * (md.cellCenters[2 * c] * s) - 1.0;
            feat[c * 7 + 1] = 2.0 * (md.cellCenters[2 * c + 1] * s) - 1.0;
            feat[c * 7 + 2] = md.lambda[c];
            feat[c * 7 + 3] = md.signedDistance[c] / hEff;
            feat[c * 7 + 4] = md.fluidMask[c];
            feat[c * 7 + 5] = md.interfaceMask[c];
            feat[c * 7 + 6] = md.solidMask[c];
        }
        features_ = toD(feat).view({-1, 7});
        xStar_ = cc.view({-1, 2}) * s;  // scaled cell centres [N,2]
    }
}

torch::Tensor DifferentiableFvResidual::applyIb(const torch::Tensor& U) const
{
    if (ibW_.numel() == 0)
    {
        return torch::zeros({ibTarget_.size(0), 2}, U.options());
    }
    auto gathered = U.index_select(0, ibSrc_);              // [M, 2]
    auto terms = gathered * ibW_.unsqueeze(1);              // [M, 2]
    // functional index_add keeps the result connected to U's graph
    return torch::zeros({ibTarget_.size(0), 2}, U.options())
        .index_add(0, ibRowIdx_, terms);
}

torch::Tensor DifferentiableFvResidual::imposedField(const torch::Tensor& U) const
{
    // fluid cells keep U; interface cells take their HFDIB value; solid = 0
    torch::Tensor uib = torch::zeros({nCells_, (int64_t)2}, U.options());
    if (ibTarget_.numel() > 0)
    {
        uib = uib.index_copy(0, ibTarget_, applyIb(U));
    }
    return fluidMask_.unsqueeze(1) * U + interfaceMask_.unsqueeze(1) * uib;
}

std::pair<torch::Tensor, torch::Tensor> DifferentiableFvResidual::stokesResidual(
    const torch::Tensor& U, const torch::Tensor& p) const
{
    // ---- internal faces ------------------------------------------------
    auto uO = U.index_select(0, owners_);   // [F,2]
    auto uN = U.index_select(0, neighs_);
    auto pO = p.index_select(0, owners_);   // [F]
    auto pN = p.index_select(0, neighs_);

    // viscous flux (uncorrected snGrad on orthogonal mesh), nu* folded in:
    auto diffFlux = nuStar_ * (uN - uO)
                  * (magSfInt_ / deltaInt_).unsqueeze(1);       // [F,2]
    // pressure face value (linear interpolation = arithmetic mean here)
    auto pFace = 0.5 * (pO + pN);                               // [F]
    auto pGradFlux = pFace.unsqueeze(1) * sfInt_;               // [F,2]
    // continuity face flux: u_f = 0.5(uO + uN), phi = u_f . Sf
    auto phi = (0.5 * (uO + uN) * sfInt_).sum(1);               // [F]

    auto momVisc = torch::zeros({nCells_, (int64_t)2}, U.options())
        .index_add(0, owners_, diffFlux)
        .index_add(0, neighs_, -diffFlux);
    auto gradP = torch::zeros({nCells_, (int64_t)2}, U.options())
        .index_add(0, owners_, pGradFlux)
        .index_add(0, neighs_, pGradFlux);
    auto cont = torch::zeros({nCells_}, U.options())
        .index_add(0, owners_, phi)
        .index_add(0, neighs_, -phi);

    // ---- boundary faces (roles: inlet [0,nIn), outlet [nIn,nIn+nOut), walls)
    auto uB = U.index_select(0, bndOwner_);                     // [B,2]
    auto pB = p.index_select(0, bndOwner_);                     // [B]

    // boundary face velocity values, built WITHOUT in-place ops so outlet
    // gradients survive: inlet fixed (u*,0), outlet owner (zeroGradient),
    // walls zero.
    torch::Tensor uBc;
    {
        auto inletRows = torch::zeros({nIn_, (int64_t)2}, U.options());
        if (nIn_ > 0)
        {
            inletRows = inletRows
                + torch::tensor({inletUStar_, 0.0}, U.options())
                      .unsqueeze(0).expand({nIn_, 2});
        }
        auto outRows = U.index_select(0, bndOwner_.slice(0, nIn_, nIn_ + nOut_));
        auto wallRows = torch::zeros({nWall_, (int64_t)2}, U.options());
        uBc = torch::cat({inletRows, outRows, wallRows}, 0);
    }

    // viscous boundary flux on fixed-U patches (inlet, walls); the outlet
    // patch (zeroGradient u) has zero diffusive flux -> masked out.
    auto bcDiff = nuStar_ * (uBc - uB)
        * (bndMagSf_ / bndDelta_).unsqueeze(1)
        * bndFixedMask_.unsqueeze(1);
    auto momBnd = torch::zeros({nCells_, (int64_t)2}, U.options())
        .index_add(0, bndOwner_, bcDiff);

    // pressure at boundary faces: inlet/walls zeroGradient -> p_owner;
    // outlet fixedValue 0 -> face value 0 (this pins the pressure gauge).
    auto pFaceB = pB * (1.0 - (1.0 - bndFixedMask_)) * 0.0
        + pB * bndFixedMask_;  // = p_owner for inlet/wall rows
    // (outlet rows get 0)
    auto pBGradFlux = pFaceB.unsqueeze(1) * bndSf_;
    gradP = gradP.index_add(0, bndOwner_, pBGradFlux);

    // continuity boundary flux
    auto phiB = (uBc * bndSf_).sum(1);
    cont = cont.index_add(0, bndOwner_, phiB);

    auto invV = (1.0 / vol_).unsqueeze(1);
    auto Rm = (gradP - (momVisc + momBnd)) * invV;  // grad p - nu lap u
    auto Rc = cont / vol_;
    return {Rm, Rc};
}

DifferentiableFvResidual::Losses DifferentiableFvResidual::losses(
    const torch::Tensor& U, const torch::Tensor& p) const
{
    auto res = stokesResidual(U, p);
    auto& Rm = res.first;
    auto& Rc = res.second;

    auto wFluid = fluidMask_ * vol_;            // [N]
    Losses L;
    L.momentum = ((Rm * Rm).sum(1) * wFluid).sum() / wFluid.sum();

    auto wCont = (fluidMask_ + interfaceMask_) * vol_;
    L.continuity = ((Rc * Rc) * wCont).sum() / wCont.sum();

    // HFDIB constraint: pure solid -> u = 0; interface -> u = u_ib
    auto pen = (U * U).sum(1) * solidMask_;
    if (ibTarget_.numel() > 0)
    {
        auto uib = applyIb(U);  // [K,2]
        auto uT = U.index_select(0, ibTarget_);
        pen = pen.index_add(0, ibTarget_, ((uT - uib) * (uT - uib)).sum(1));
    }
    auto wIb = solidMask_ + interfaceMask_;
    L.ib = (pen * vol_).sum() / (wIb * vol_).sum();

    L.total = L.momentum + cfgGammaC_ * L.continuity + cfgGammaIb_ * L.ib;

    // paper forced-balance diagnostic: R(U) - chi * R(U_imposed)
    auto resS = stokesResidual(imposedField(U), p);
    auto rFull = res.first - chiMask_.unsqueeze(1) * resS.first;
    L.rFull = torch::sqrt(((rFull * rFull).sum(1) * vol_).sum() / vol_.sum());
    return L;
}

DifferentiableFvResidual::Metrics DifferentiableFvResidual::metrics(
    const torch::Tensor& U, const torch::Tensor& p) const
{
    torch::NoGradGuard ng;
    auto speed = (U * U).sum(1).sqrt();
    Metrics m{};
    m.maxSolidSpeed = 0.0;
    m.maxInterfaceError = 0.0;
    if (solidMask_.sum().item<double>() > 0)
    {
        m.maxSolidSpeed = (speed * solidMask_).max().item<double>();
    }
    if (ibTarget_.numel() > 0)
    {
        auto err = (U.index_select(0, ibTarget_) - applyIb(U));
        m.maxInterfaceError = (err * err).sum(1).sqrt().max().item<double>();
    }
    // inlet flux is exact by construction (fixed face values)
    m.inletFlux = inletUStar_
                * bndMagSf_.slice(0, 0, nIn_).sum().item<double>();
    m.outletFlux = (U.index_select(0, outletOwnerIdx_) * outletSf_)
        .sum(1).sum().item<double>();
    m.massImbalance = std::abs(m.outletFlux - m.inletFlux)
                    / (std::abs(m.inletFlux) + 1e-30);
    auto pIn = p.index_select(0, inletOwnerIdx_).mean().item<double>();
    auto pOut = p.index_select(0, outletOwnerIdx_).mean().item<double>();
    m.pressureDrop = pIn - pOut;
    m.rFullNorm = losses(U, p).rFull.item<double>();
    return m;
}
