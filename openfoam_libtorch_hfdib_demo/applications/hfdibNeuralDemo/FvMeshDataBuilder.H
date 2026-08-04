#ifndef FvMeshDataBuilder_H
#define FvMeshDataBuilder_H

#include "fvMesh.H"
#include "FvMeshData.H"

// Extracts the plain FvMeshData (without the HFDIB-specific arrays, which
// are filled by the geometry/interpolation builders).
class FvMeshDataBuilder
{
public:
    static FvMeshData build(const Foam::fvMesh& mesh);
};

#endif
