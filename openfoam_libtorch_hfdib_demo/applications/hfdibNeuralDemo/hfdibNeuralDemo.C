#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "IOdictionary.H"
#include "OFstream.H"
#include "OSspecific.H"
#include "BitmapTopology.H"
#include "HfdibGeometryBuilder.H"
#include "HfdibInterpolationBuilder.H"
#include "FvMeshDataBuilder.H"
#include "TorchCompat.H"
#include <cmath>
#include <vector>
#include <numeric>

using namespace Foam;

namespace
{

// Build the 4 always-fluid port rectangles from the inlet/outlet patches
// (split by side and by y below/above the chamber centreline).
std::vector<BitmapTopology::Rect> portRectsFromPatches(const fvMesh& mesh)
{
    std::vector<BitmapTopology::Rect> rects;
    const surfaceVectorField& Cf = mesh.Cf();
    for (const word pName : {word("inlet"), word("outlet")})
    {
        const auto& bny = mesh.boundary();
        label patchI = -1;
        forAll(bny, bi)
        {
            if (bny[bi].name() == pName)
            {
                patchI = bi;
                break;
            }
        }
        const auto& pp = bny[patchI];
        // split into below/above chamber half
        double loMin[2] = {1e300, 1e300}, loMax[2] = {-1e300, -1e300};
        double upMin[2] = {1e300, 1e300}, upMax[2] = {-1e300, -1e300};
        double yMid = 0.0;
        {
            double yMinAll = 1e300, yMaxAll = -1e300;
            forAll(mesh.C().primitiveField(), c)
            {
                yMinAll = std::min(yMinAll, static_cast<double>(mesh.C().primitiveField()[c].y()));
                yMaxAll = std::max(yMaxAll, static_cast<double>(mesh.C().primitiveField()[c].y()));
            }
            yMid = 0.5 * (yMinAll + yMaxAll);
        }
        const double hHalf = 0.5 * std::cbrt(static_cast<double>(mesh.V()[0]));
        forAll(pp, i)
        {
            const vector f = Cf[pp.start() + i];
            double* mn = (f.y() < yMid) ? loMin : upMin;
            double* mx = (f.y() < yMid) ? loMax : upMax;
            mn[0] = std::min(mn[0], f.x() - hHalf);
            mn[1] = std::min(mn[1], f.y() - hHalf);
            mx[0] = std::max(mx[0], f.x() + hHalf);
            mx[1] = std::max(mx[1], f.y() + hHalf);
        }
        rects.push_back({loMin[0], loMax[0], loMin[1], loMax[1]});
        rects.push_back({upMin[0], upMax[0], upMin[1], upMax[1]});
    }
    return rects;
}

} // anonymous namespace

int main(int argc, char* argv[])
{
    argList::addOption("mode", "name",
        "smoke | setup | testLambda | testInterpolation | residualCheck | "
        "coefficients | neural");

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    IOdictionary physicsProperties
    (
        IOobject
        (
            "physicsProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    const scalar nu = readScalar(physicsProperties.lookup("nu"));
    const scalar inletU = readScalar(physicsProperties.lookup("inletVelocity"));
    const scalar Uref = readScalar(physicsProperties.lookup("Uref"));
    const scalar Lref = readScalar(physicsProperties.lookup("Lref"));
    const scalar d1Factor = physicsProperties.lookupOrDefault<scalar>("d1Factor", 1.5);
    const scalar d2Factor = physicsProperties.lookupOrDefault<scalar>("d2Factor", 1.0);

    word mode = "smoke";
    if (args.optionFound("mode")) mode = args.optionRead<word>("mode");

    fileName postProcDir = runTime.path() / "postProcessing" / "hfdibNeuralDemo";

    // ---------------------------------------------------------------- smoke
    if (mode == "smoke")
    {
        Info << "Torch version: " << TORCH_VERSION << nl;
        torch::set_num_threads(1);

        // autograd smoke: y = x^2 at x = [1,2,3] -> dy/dx = [2,4,6]
        auto x = torch::tensor({1.0, 2.0, 3.0},
                               torch::TensorOptions().dtype(torch::kFloat64)
                                                     .requires_grad(true));
        auto y = (x * x).sum();
        y.backward();
        const double g0 = x.grad()[0].item<double>();
        const double g1 = x.grad()[1].item<double>();
        const double g2 = x.grad()[2].item<double>();

        // tensor from OpenFOAM-derived array
        std::vector<double> cx(mesh.nCells());
        for (label c = 0; c < mesh.nCells(); ++c) cx[c] = mesh.C().primitiveField()[c].x();
        auto xc = torch::from_blob(cx.data(), {(int64_t)cx.size()},
                                   torch::kFloat64).clone();

        const bool ok =
            std::abs(g0 - 2.0) < 1e-14 &&
            std::abs(g1 - 4.0) < 1e-14 &&
            std::abs(g2 - 6.0) < 1e-14 &&
            std::abs(xc.sum().item<double>() -
                     std::accumulate(cx.begin(), cx.end(), 0.0)) < 1e-14;
        Info << "autograd grads: " << g0 << " " << g1 << " " << g2 << nl;
        Info << "OF-array tensor sum check: " << xc.sum().item<double>() << nl;
        Info << (ok ? "Gate 0 SMOKE PASS" : "Gate 0 SMOKE FAIL") << nl;
        return ok ? 0 : 1;
    }

    // ------------------------------------------------- shared geometry data
    // AOI width (dataset: 64 * 0.002 m); used both as Lref and bitmap width
    const double chamberW = static_cast<double>(Lref);
    BitmapTopology topo(
        runTime.path() / "constant" / "topology8x8.csv",
        chamberW,
        portRectsFromPatches(mesh));

    HfdibGeometryBuilder geomBuilder(mesh, topo);

    // ---------------------------------------------------------------- setup
    if (mode == "setup")
    {
        geomBuilder.writeFields(postProcDir, Foam::Time::timeName(runTime.value()));
        geomBuilder.writeGeometryCsv(postProcDir);

        HfdibInterpolationBuilder interpBuilder(
            mesh, topo, geomBuilder.geometry(), geomBuilder.lambda(),
            d1Factor, d2Factor);

        // acceptance: lambda in [0,1], masks partition, interface nonempty
        bool ok = geomBuilder.geometry().size() > 0;
        for (double l : geomBuilder.lambda())
        {
            ok = ok && std::isfinite(l) && l >= 0.0 && l <= 1.0;
        }
        for (label c = 0; c < mesh.nCells(); ++c)
        {
            const double s = geomBuilder.fluidMask()[c]
                           + geomBuilder.interfaceMask()[c]
                           + geomBuilder.solidMask()[c];
            ok = ok && std::abs(s - 1.0) < 1e-14;
        }
        Info << "lambda/masks acceptance: " << (ok ? "PASS" : "FAIL") << nl;
        Info << "Setup complete. Fields + CSVs in " << postProcDir << nl;
        return ok ? 0 : 1;
    }

    // ---------------------------------------------------------- testLambda
    if (mode == "testLambda")
    {
        const auto& lam = geomBuilder.lambda();
        const auto& sig = geomBuilder.sigma();
        const auto& fm = geomBuilder.fluidMask();
        const auto& im = geomBuilder.interfaceMask();
        const auto& sm = geomBuilder.solidMask();
        const auto& V = mesh.V();
        const double hLoc = std::cbrt(static_cast<double>(V[0]));

        int failures = 0;
        auto check = [&](bool cond, const std::string& name)
        {
            Info << "  [" << (cond ? "PASS" : "FAIL") << "] " << name << nl;
            if (!cond) ++failures;
        };

        // 1. known fluid location has lambda == 0
        bool foundFluid = false;
        for (label c = 0; c < mesh.nCells(); ++c)
        {
            if (mesh.C().primitiveField()[c].x() < 0.0)  // deep inside a port corridor
            {
                check(lam[c] == 0.0, "port-corridor cell has lambda = 0");
                foundFluid = true;
                break;
            }
        }
        check(foundFluid, "found a port-corridor cell");
        bool foundDeepFluid = false;
        for (label c = 0; c < mesh.nCells(); ++c)
        {
            if (sig[c] > 3.0 * hLoc)
            {
                check(lam[c] == 0.0, "deep-fluid chamber cell has lambda = 0");
                foundDeepFluid = true;
                break;
            }
        }
        check(foundDeepFluid, "found a deep-fluid cell");

        // 2. known solid location has lambda == 1
        bool foundSolid = false;
        for (label c = 0; c < mesh.nCells(); ++c)
        {
            if (sig[c] < -3.0 * hLoc)
            {
                check(lam[c] == 1.0, "deep-solid cell has lambda = 1");
                foundSolid = true;
                break;
            }
        }
        check(foundSolid, "found a deep-solid cell");

        // 3. (1 - lambda) masking zeros the solid and retains the fluid
        bool maskOk = true;
        const double Cv = 0.12345;
        for (label c = 0; c < mesh.nCells(); ++c)
        {
            const double masked = (1.0 - lam[c]) * Cv;
            if (sm[c] == 1.0 && masked != 0.0)   maskOk = false;
            if (fm[c] == 1.0 && masked != Cv)    maskOk = false;
        }
        check(maskOk, "velocity masking by (1 - lambda): solid 0, fluid kept");

        // 4. ceil(lambda) is zero only in pure fluid cells
        bool ceilOk = true;
        for (label c = 0; c < mesh.nCells(); ++c)
        {
            const double chi = std::ceil(lam[c]);
            if (fm[c] == 1.0 && chi != 0.0) ceilOk = false;
            if ((sm[c] == 1.0 || im[c] == 1.0) && chi != 1.0) ceilOk = false;
        }
        check(ceilOk, "ceil(lambda) = 0 only in pure fluid cells");

        // generic acceptance
        bool rangeOk = true, partOk = true, interfaceNonEmpty = false;
        for (label c = 0; c < mesh.nCells(); ++c)
        {
            rangeOk = rangeOk && std::isfinite(lam[c])
                    && lam[c] >= 0.0 && lam[c] <= 1.0;
            partOk = partOk &&
                std::abs(fm[c] + im[c] + sm[c] - 1.0) < 1e-14;
            if (im[c] == 1.0) interfaceNonEmpty = true;
        }
        check(rangeOk, "lambda in [0, 1] everywhere");
        check(partOk, "fluidMask + interfaceMask + solidMask == 1");
        check(interfaceNonEmpty, "interfaceMask is nonempty");

        Info << "test_lambda_convention: "
             << (failures == 0 ? "PASS" : "FAIL") << nl;
        return failures == 0 ? 0 : 1;
    }

    if (mode == "testInterpolation")
    {
        FatalErrorInFunction << "mode testInterpolation: implemented in stage 2"
                             << exit(FatalError);
    }

    FatalErrorInFunction << "unknown mode '" << mode << "'" << exit(FatalError);
    return 1;
}
