#ifndef FvMeshData_H
#define FvMeshData_H

#include <cstddef>
#include <cstdint>
#include <vector>

// Plain-data bridge between the OpenFOAM side (mesh, fields, boundary
// metadata, HFDIB geometry) and the Torch-only training side.  Contains
// only standard-library objects (implementation brief, section 9).
//
// All lengths are DIMENSIONAL (SI).  The Torch side applies the 1/Lref
// scaling once when building tensors.
struct FvMeshData
{
    std::size_t nCells = 0;

    // Internal faces
    std::vector<int64_t> owners;
    std::vector<int64_t> neighbours;

    // Geometry (2D vectors: z dropped, z-fluxes are zero)
    std::vector<double> cellVolumes;        // [nCells]
    std::vector<double> cellCenters;        // [2*nCells]  (x,y)
    std::vector<double> faceAreaVecInt;     // [2*nInt]    Sf vector (x,y)
    std::vector<double> faceMagSfInt;       // [nInt]      |Sf|
    std::vector<double> faceDeltaInt;       // [nInt]      d_f = (Cn-Co).n

    // Boundary faces grouped by patch role (only inlet/outlet/walls;
    // "empty" faces are excluded).  faceAreaVecBnd/MagSf/Delta/Owner share
    // the same ordering as the concatenation of these lists; roleOffsets
    // marks {0, end(inlet), end(outlet), end(walls)}.
    std::vector<int64_t> inletFaces;        // global face ids
    std::vector<int64_t> outletFaces;
    std::vector<int64_t> wallFaces;
    std::vector<int64_t> bndOwner;          // owner cell per boundary face
    std::vector<double>  bndFaceAreaVec;    // Sf vector (x,y)
    std::vector<double>  bndFaceMagSf;
    std::vector<double>  bndDelta;          // distance centre->face

    // HFDIB fields (paper convention: lambda = 0 fluid, = 1 solid)
    std::vector<double> lambda;
    std::vector<double> signedDistance;
    std::vector<double> fluidMask;
    std::vector<double> interfaceMask;
    std::vector<double> solidMask;

    // HFDIB sparse interpolation map (CSR), per velocity component:
    // for target row k (an interface cell),
    //   u_ib[target_k] = sum_{j in [rowStart[k], rowStart[k+1])} w_j u[src_j]
    // Solid cells are not targets; their imposed value is identically zero.
    std::vector<int64_t> ibTargetCells;     // [K]
    std::vector<int64_t> ibRowStart;        // [K+1]
    std::vector<int64_t> ibSourceCells;     // [M]
    std::vector<double>  ibWeights;         // [M]
    std::vector<int64_t> ibOrder;           // [K] polynomial order used (0/1/2)

    // HFDIB diagnostics (per interface cell, parallel to ibTargetCells)
    std::vector<double> ibDs;               // distance cell center -> surface
    std::vector<double> ibNormal;           // [2K] fluid-side normal
    std::vector<double> ibSurfacePoint;     // [2K]
};

#endif
