#ifndef HfdibGeometryData_H
#define HfdibGeometryData_H

#include <vector>
#include <cstdint>

// Plain-data description of the static HFDIB geometry (no OpenFOAM or Torch
// types).  One record per interface cell (0 < lambda < 1).
struct HfdibGeometryData
{
    std::vector<int64_t> cells;     // interface cell indices        [K]
    std::vector<double> surfacePoint;  // q, 2 comps (x,y)           [2K]
    std::vector<double> normal;        // n, fluid-side unit         [2K]
    std::vector<double> ds;            // distance cell-centre->q    [K]
    std::vector<double> sigma;         // signed distance            [K]

    std::size_t size() const { return cells.size(); }
};

#endif
