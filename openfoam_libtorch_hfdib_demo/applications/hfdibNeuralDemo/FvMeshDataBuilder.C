#include "FvMeshDataBuilder.H"
#include "volFields.H"
#include <cmath>

using namespace Foam;

FvMeshData FvMeshDataBuilder::build(const fvMesh& mesh)
{
    FvMeshData md;
    const label nCells = mesh.nCells();
    md.nCells = static_cast<std::size_t>(nCells);

    const surfaceVectorField& Sf = mesh.Sf();
    const surfaceScalarField& magSf = mesh.magSf();
    const auto& C = mesh.C().primitiveField();
    const scalarField& V = mesh.V();
    const labelUList& owner = mesh.owner();
    const labelUList& neighbour = mesh.neighbour();

    md.cellVolumes.assign(V.begin(), V.end());
    md.cellCenters.resize(2 * nCells);
    for (label c = 0; c < nCells; ++c)
    {
        md.cellCenters[2 * c] = C[c].x();
        md.cellCenters[2 * c + 1] = C[c].y();
    }

    // internal faces
    const label nInt = owner.size();
    md.owners.assign(owner.begin(), owner.end());
    md.neighbours.assign(neighbour.begin(), neighbour.end());
    md.faceAreaVecInt.resize(2 * nInt);
    md.faceMagSfInt.resize(nInt);
    md.faceDeltaInt.resize(nInt);
    for (label f = 0; f < nInt; ++f)
    {
        const vector sf = Sf[f];
        const double a = mag(sf);
        const vector n = sf / a;
        const label o = owner[f];
        const label nb = neighbour[f];
        const double d = (C[nb] - C[o]) & n;
        md.faceAreaVecInt[2 * f] = sf.x();
        md.faceAreaVecInt[2 * f + 1] = sf.y();
        md.faceMagSfInt[f] = a;
        md.faceDeltaInt[f] = (std::abs(d) > 1e-30) ? d : mag(C[nb] - C[o]);
        if (d <= 0)
        {
            // non-orthogonal pathological face; keep a positive fallback
            md.faceDeltaInt[f] = mag(C[nb] - C[o]);
        }
    }

    // boundary faces grouped by patch role (skip "empty" etc.)
    auto appendPatch = [&](const word& patchName, std::vector<int64_t>& faces)
    {
        label patchI = -1;
        const auto& bny = mesh.boundary();
        forAll(bny, i)
        {
            if (bny[i].name() == patchName)
            {
                patchI = i;
                break;
            }
        }
        if (patchI < 0)
        {
            FatalErrorInFunction
                << "patch '" << patchName << "' not found" << exit(FatalError);
        }
        const auto& pp = bny[patchI];
        const labelUList& fc = pp.faceCells();
        forAll(pp, i)
        {
            const label faceI = pp.start() + i;
            faces.push_back(faceI);
            md.bndOwner.push_back(fc[i]);
            const vector sf = Sf[faceI];
            md.bndFaceAreaVec.push_back(sf.x());
            md.bndFaceAreaVec.push_back(sf.y());
            md.bndFaceMagSf.push_back(mag(sf));
            const double d = (mesh.Cf()[faceI] - C[fc[i]]) & (sf / mag(sf));
            md.bndDelta.push_back(std::abs(d));
        }
    };

    appendPatch("inlet", md.inletFaces);
    appendPatch("outlet", md.outletFaces);
    appendPatch("walls", md.wallFaces);

    return md;
}
