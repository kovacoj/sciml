#ifndef HfdibInterpolationBuilder_H
#define HfdibInterpolationBuilder_H

#include "fvMesh.H"
#include "BitmapTopology.H"
#include "HfdibGeometryData.H"
#include "HfdibInterpolationData.H"

// Builds the static line-interpolation HFDIB map for a fixed topology
// (clean reimplementation of the openHFDIB lineInt zero/linear/quadratic
// order hierarchy; see references/UPSTREAM.md).
//
// For each interface cell P:
//   x1 = q + d1*hLoc*n,  x2 = q + (d1+d2)*hLoc*n   (fluid side)
// with d1 = d1Factor and d2 = d2Factor (defaults ~1.5 and ~1 cell sizes).
// Interpolation-point velocities are evaluated from cell-centred values:
// bilinear over the 4 surrounding cell centres when all four exist and are
// not fully solid, otherwise inverse-distance weighting (IDW) over the
// usable cells of the 3x3 ring.
//
// Order hierarchy per cell (stationary wall, u_Gamma = 0):
//   2: quadratic through (0,0),(d1,u1),(d1+d2,u2) evaluated at ds
//   1: linear through (0,0),(d1,u1)
//   0: u = 0 (no fluid-side interpolation point usable)
class HfdibInterpolationBuilder
{
public:
    HfdibInterpolationBuilder(
        const Foam::fvMesh& mesh,
        const BitmapTopology& topo,
        const HfdibGeometryData& geom,
        const std::vector<double>& lambda,
        double d1Factor = 1.5,
        double d2Factor = 1.0);

    const HfdibInterpolationData& data() const { return data_; }

private:
    // Evaluate interpolation weights of a point from surrounding cells.
    // Returns false when the point is unusable (outside domain or no
    // non-solid cells around). mode: 2 bilinear, 1 IDW.
    bool pointWeights(double px, double py,
                      std::vector<std::pair<Foam::label, double>>& out,
                      int& mode) const;

    const Foam::fvMesh& mesh_;
    const BitmapTopology& topo_;
    const std::vector<double>& lambda_;

    HfdibInterpolationData data_;

    // Cartesian lookup grid: cell centres are uniform (h x h), possibly
    // with missing columns outside the port bands.  grid_[ix*gH_+iy] = cell
    // index or -1.
    double h_;
    double x0_, y0_;   // min cell-centre coordinates
    int gW_, gH_;
    std::vector<Foam::label> grid_;
};

#endif
