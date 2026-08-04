#include "HfdibGeometryBuilder.H"
#include "OFstream.H"
#include "OSspecific.H"
#include <cmath>
#include <algorithm>

using namespace Foam;

HfdibGeometryBuilder::HfdibGeometryBuilder(
    const fvMesh& mesh,
    const BitmapTopology& topo)
:
    mesh_(mesh),
    topo_(topo)
{
    compute();
}

void HfdibGeometryBuilder::compute()
{
    const label nCells = mesh_.nCells();
    const auto& C = mesh_.C().primitiveField();
    const scalarField& V = mesh_.V();

    lambda_.assign(nCells, 0.0);
    sigma_.assign(nCells, 0.0);
    fluidMask_.assign(nCells, 0.0);
    interfaceMask_.assign(nCells, 0.0);
    solidMask_.assign(nCells, 0.0);

    const auto& cellPoints = mesh_.cellPoints();
    const pointField& pts = mesh_.points();

    for (label cellI = 0; cellI < nCells; ++cellI)
    {
        const vector ci = C[cellI];
        vector q(Zero), n(Zero);
        const double sig = topo_.signedDistance(ci, q, n);
        sigma_[cellI] = sig;

        // A cell is intersected by the interface iff its in-plane corner
        // vertices are not all on the same side of the solid.  Hex corner
        // vertices appear twice (top/bottom), hence nTotal = 8.
        const labelList& cp = cellPoints[cellI];
        label nInside = 0;
        for (const label pi : cp)
        {
            const point& pp = pts[pi];
            if (topo_.insideSolid(vector(pp.x(), pp.y(), 0))) ++nInside;
        }
        const label nTotal = cp.size();
        const bool mixed = (nInside > 0 && nInside < nTotal);

        if (!mixed)
        {
            lambda_[cellI] = (nInside == nTotal) ? 1.0 : 0.0;
        }
        else
        {
            const double hEff = std::cbrt(static_cast<double>(V[cellI]));
            lambda_[cellI] = 0.5 * (1.0 - std::tanh(sig / hEff));
            geom_.cells.push_back(cellI);
            geom_.surfacePoint.push_back(q.x());
            geom_.surfacePoint.push_back(q.y());
            geom_.normal.push_back(n.x());
            geom_.normal.push_back(n.y());
            geom_.ds.push_back(std::abs(sig));
            geom_.sigma.push_back(sig);
        }
    }

    for (label cellI = 0; cellI < nCells; ++cellI)
    {
        const double lam = lambda_[cellI];
        if (lam <= 0.0)      fluidMask_[cellI] = 1.0;
        else if (lam >= 1.0) solidMask_[cellI] = 1.0;
        else                 interfaceMask_[cellI] = 1.0;
    }

    const label nSolid = static_cast<label>(
        std::count(solidMask_.begin(), solidMask_.end(), 1.0));
    Info << "HFDIB geometry: " << nCells << " cells, interface cells "
         << geom_.size() << ", solid cells " << nSolid << nl;
}

void HfdibGeometryBuilder::writeFields(const fileName& postProcDir, const Foam::word& timeName) const
{
    mkDir(postProcDir);

    auto writeVol = [&](const word& name, const std::vector<double>& vals)
    {
        volScalarField f
        (
            IOobject
            (
                name,
                timeName,
                mesh_,
                IOobject::NO_READ,
                IOobject::AUTO_WRITE
            ),
            mesh_,
            dimensionedScalar("zero", dimless, 0)
        );
        for (label cellI = 0; cellI < mesh_.nCells(); ++cellI)
        {
            f[cellI] = vals[cellI];
        }
        // keep the boundary consistent so the written field is loadable
        forAll(f.boundaryField(), patchI)
        {
            if (mesh_.boundary()[patchI].type() != "empty")
            {
                f.boundaryFieldRef()[patchI] ==
                    f.boundaryField()[patchI].patchInternalField();
            }
        }
        f.write();
    };

    writeVol("sigma", sigma_);
    writeVol("lambda", lambda_);
    writeVol("fluidMask", fluidMask_);
    writeVol("interfaceMask", interfaceMask_);
    writeVol("solidMask", solidMask_);

    // plain CSV for plotting scripts
    OFstream os(postProcDir / "fields.csv");
    os << "cell,x,y,sigma,lambda,fluidMask,interfaceMask,solidMask" << endl;
    const auto& C = mesh_.C().primitiveField();
    for (label cellI = 0; cellI < mesh_.nCells(); ++cellI)
    {
        os << cellI << ","
           << C[cellI].x() << ","
           << C[cellI].y() << ","
           << sigma_[cellI] << ","
           << lambda_[cellI] << ","
           << fluidMask_[cellI] << ","
           << interfaceMask_[cellI] << ","
           << solidMask_[cellI] << endl;
    }
}

void HfdibGeometryBuilder::writeGeometryCsv(const fileName& postProcDir) const
{
    mkDir(postProcDir);
    OFstream os(postProcDir / "geometry.csv");
    os << "cell,qx,qy,nx,ny,ds,sigma" << endl;
    for (std::size_t k = 0; k < geom_.size(); ++k)
    {
        os << geom_.cells[k] << ","
           << geom_.surfacePoint[2 * k] << ","
           << geom_.surfacePoint[2 * k + 1] << ","
           << geom_.normal[2 * k] << ","
           << geom_.normal[2 * k + 1] << ","
           << geom_.ds[k] << ","
           << geom_.sigma[k] << endl;
    }
}
