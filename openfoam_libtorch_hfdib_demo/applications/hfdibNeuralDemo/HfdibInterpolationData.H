#ifndef HfdibInterpolationData_H
#define HfdibInterpolationData_H

#include <vector>
#include <cstdint>

// Sparse CSR interpolation map for the immersed velocity (per velocity
// component): for target row k (an interface cell),
//   u_ib[cell_k] = sum_j weight_j * u[src_j]
// plus per-target diagnostic records.
struct HfdibInterpolationData
{
    std::vector<int64_t> targetCells;   // [K]
    std::vector<int64_t> rowStart;      // [K+1]
    std::vector<int64_t> sourceCells;   // [M]
    std::vector<double>  weights;       // [M]

    // diagnostics, parallel to targetCells
    std::vector<int64_t> order;         // polynomial order used (0/1/2)
    std::vector<double>  ds;            // centre -> surface distance
    std::vector<double>  normal;        // [2K]
    std::vector<double>  surfacePoint;  // [2K]
    std::vector<double>  point1;        // [2K] first interp point  (= q + d1 n)
    std::vector<double>  point2;        // [2K] second interp point (= q + (d1+d2) n)
    std::vector<int64_t> pointEvalMode; // per point, 2 comps: 2=bilinear,1=IDW,(-1)=invalid
};

#endif
