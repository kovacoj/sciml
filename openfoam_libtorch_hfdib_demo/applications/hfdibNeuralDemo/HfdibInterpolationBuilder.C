#include "HfdibInterpolationBuilder.H"
#include "volFields.H"
#include <cmath>
#include <algorithm>
#include <map>

using namespace Foam;

HfdibInterpolationBuilder::HfdibInterpolationBuilder(
    const fvMesh& mesh,
    const BitmapTopology& topo,
    const HfdibGeometryData& geom,
    const std::vector<double>& lambda,
    double d1Factor,
    double d2Factor)
:
    mesh_(mesh),
    topo_(topo),
    lambda_(lambda)
{
    const label nCells = mesh_.nCells();
    const auto& C = mesh_.C().primitiveField();
    const scalarField& V = mesh_.V();

    // uniform Cartesian lookup grid
    h_ = std::cbrt(static_cast<double>(V[0]));
    x0_ = 1e300; y0_ = 1e300;
    double xMax = -1e300, yMax = -1e300;
    for (label c = 0; c < nCells; ++c)
    {
        x0_ = std::min(x0_, static_cast<double>(C[c].x()));
        y0_ = std::min(y0_, static_cast<double>(C[c].y()));
        xMax = std::max(xMax, static_cast<double>(C[c].x()));
        yMax = std::max(yMax, static_cast<double>(C[c].y()));
    }
    gW_ = static_cast<int>(std::lround((xMax - x0_) / h_)) + 1;
    gH_ = static_cast<int>(std::lround((yMax - y0_) / h_)) + 1;
    grid_.assign(static_cast<std::size_t>(gW_) * gH_, -1);
    for (label c = 0; c < nCells; ++c)
    {
        const int ix = static_cast<int>(std::lround((C[c].x() - x0_) / h_));
        const int iy = static_cast<int>(std::lround((C[c].y() - y0_) / h_));
        grid_[static_cast<std::size_t>(ix) * gH_ + iy] = c;
    }

    data_.rowStart.push_back(0);
    label nOrd0 = 0, nOrd1 = 0, nOrd2 = 0;

    std::map<std::pair<int64_t, int64_t>, double> acc;  // (row, source) -> w
    int64_t row = 0;

    for (std::size_t k = 0; k < geom.size(); ++k)
    {
        const label cellI = geom.cells[k];
        const double qx = geom.surfacePoint[2 * k];
        const double qy = geom.surfacePoint[2 * k + 1];
        const double nx = geom.normal[2 * k];
        const double ny = geom.normal[2 * k + 1];
        const double ds = geom.ds[k];
        const double hLoc = std::cbrt(static_cast<double>(V[cellI]));

        const double d1 = d1Factor * hLoc;
        const double d2 = d2Factor * hLoc;

        const double x1 = qx + d1 * nx, y1 = qy + d1 * ny;
        const double x2 = qx + (d1 + d2) * nx, y2 = qy + (d1 + d2) * ny;

        auto usable = [&](double px, double py,
                          std::vector<std::pair<label, double>>& w, int& mode)
        {
            // must be strictly fluid-side and have a usable stencil
            vector q2(Zero), n2(Zero);
            const double s = topo_.signedDistance(vector(px, py, 0), q2, n2);
            if (s <= 0.0) return false;
            return pointWeights(px, py, w, mode);
        };

        std::vector<std::pair<label, double>> w1, w2;
        int mode1 = -1, mode2 = -1;
        const bool ok1 = usable(x1, y1, w1, mode1);
        const bool ok2 = ok1 && usable(x2, y2, w2, mode2);

        int order = 0;
        if (ok1 && ok2) order = 2;
        else if (ok1)    order = 1;
        else             order = 0;

        if (order == 0) ++nOrd0;
        else if (order == 1) ++nOrd1;
        else ++nOrd2;

        // polynomial coefficients (stationary wall, u_Gamma = 0)
        const double D = d1 * d2 * (d1 + d2);
        double c1 = 0.0, c2 = 0.0;
        if (order == 2)
        {
            c1 = (d1 + d2) * ds * (d1 + d2 - ds) / D;
            c2 = d1 * ds * (ds - d1) / D;
        }
        else if (order == 1)
        {
            c1 = ds / d1;
        }

        for (const auto& pw : w1)
        {
            const label src = pw.first;
            acc[std::make_pair(row, static_cast<int64_t>(src))] += c1 * pw.second;
        }
        if (order == 2)
        {
            for (const auto& pw : w2)
            {
                const label src = pw.first;
                acc[std::make_pair(row, static_cast<int64_t>(src))] += c2 * pw.second;
            }
        }

        data_.targetCells.push_back(cellI);
        data_.order.push_back(order);
        data_.ds.push_back(ds);
        data_.normal.push_back(nx);
        data_.normal.push_back(ny);
        data_.surfacePoint.push_back(qx);
        data_.surfacePoint.push_back(qy);
        data_.point1.push_back(x1);
        data_.point1.push_back(y1);
        data_.point2.push_back(x2);
        data_.point2.push_back(y2);
        data_.pointEvalMode.push_back(mode1);
        data_.pointEvalMode.push_back(mode2);

        // flush CSR row (skip numerically negligible terms)
        for (auto it = acc.begin(); it != acc.end();)
        {
            if (it->first.first != row) break;
            if (std::abs(it->second) > 1e-14)
            {
                data_.sourceCells.push_back(it->first.second);
                data_.weights.push_back(it->second);
            }
            it = acc.erase(it);
        }
        data_.rowStart.push_back(
            static_cast<int64_t>(data_.sourceCells.size()));
        ++row;
    }
    acc.clear();

    Info << "HFDIB interpolation: " << geom.size()
         << " interface cells; orders used: 0:" << nOrd0
         << " 1:" << nOrd1 << " 2:" << nOrd2 << nl;
    if (geom.size() > 0 && nOrd0 == geom.size())
    {
        FatalErrorInFunction
            << "no usable interpolation stencils at all" << exit(FatalError);
    }
}

bool HfdibInterpolationBuilder::pointWeights(
    double px, double py,
    std::vector<std::pair<label, double>>& out,
    int& mode) const
{
    const double fx = (px - x0_) / h_;
    const double fy = (py - y0_) / h_;

    const int icx = static_cast<int>(std::lround(fx));
    const int icy = static_cast<int>(std::lround(fy));

    auto at = [&](int ix, int iy) -> label
    {
        if (ix < 0 || ix >= gW_ || iy < 0 || iy >= gH_) return -1;
        return grid_[static_cast<std::size_t>(ix) * gH_ + iy];
    };

    // --- try bilinear over the four surrounding cell centres -----------
    {
        int i1 = static_cast<int>(std::floor(fx - 0.5));
        int j1 = static_cast<int>(std::floor(fy - 0.5));
        i1 = i1 < 0 ? 0 : (i1 > gW_ - 2 ? gW_ - 2 : i1);
        j1 = j1 < 0 ? 0 : (j1 > gH_ - 2 ? gH_ - 2 : j1);

        const label c00 = at(i1, j1), c10 = at(i1 + 1, j1);
        const label c01 = at(i1, j1 + 1), c11 = at(i1 + 1, j1 + 1);

        if (c00 >= 0 && c10 >= 0 && c01 >= 0 && c11 >= 0
            && lambda_[c00] < 1.0 && lambda_[c10] < 1.0
            && lambda_[c01] < 1.0 && lambda_[ c11] < 1.0)
        {
            const double cx0 = x0_ + (i1 + 0.5) * h_;
            const double cy0 = y0_ + (j1 + 0.5) * h_;
            double tx = (px - cx0) / h_;
            if (tx < 0.0) tx = 0.0; else if (tx > 1.0) tx = 1.0;
            double ty = (py - cy0) / h_;
            if (ty < 0.0) ty = 0.0; else if (ty > 1.0) ty = 1.0;
            out = {
                {c00, (1 - tx) * (1 - ty)},
                {c10, tx * (1 - ty)},
                {c01, (1 - tx) * ty},
                {c11, tx * ty}};
            mode = 2;
            return true;
        }
    }

    // --- fallback: IDW over usable cells of the 3x3 ring ---------------
    {
        double wsum = 0.0;
        std::vector<std::pair<label, double>> cand;
        for (int dx = -1; dx <= 1; ++dx)
        {
            for (int dy = -1; dy <= 1; ++dy)
            {
                const label c = at(icx + dx, icy + dy);
                if (c < 0 || lambda_[c] >= 1.0) continue;
                const double cx = x0_ + (icx + dx + 0.5) * h_;
                const double cy = y0_ + (icy + dy + 0.5) * h_;
                const double r2 = (px - cx) * (px - cx) + (py - cy) * (py - cy);
                const double w = 1.0 / (r2 + 1e-30);
                cand.emplace_back(c, w);
                wsum += w;
            }
        }
        if (cand.empty()) return false;
        for (auto& cw : cand)
        {
            out.emplace_back(cw.first, cw.second / wsum);
        }
        mode = 1;
        return true;
    }
}
