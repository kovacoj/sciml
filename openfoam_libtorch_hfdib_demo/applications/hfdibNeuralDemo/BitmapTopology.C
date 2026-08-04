#include "BitmapTopology.H"
#include "IFstream.H"
#include "error.H"
#include <cmath>
#include <algorithm>

using namespace Foam;

void BitmapTopology::distToRect(
    double px, double py, const Rect& R,
    double& d, double& qx, double& qy)
{
    // clamp the point onto the rectangle closure
    qx = (px < R.x0) ? R.x0 : ((px > R.x1) ? R.x1 : px);
    qy = (py < R.y0) ? R.y0 : ((py > R.y1) ? R.y1 : py);
    double dx = px - qx;
    double dy = py - qy;
    d = std::sqrt(dx * dx + dy * dy);
}

BitmapTopology::BitmapTopology(
    const Foam::fileName& csvPath,
    double chamberWidth,
    const std::vector<Rect>& portRects)
:
    H_(chamberWidth),
    w_(chamberWidth / 8.0)
{
    // read 8 lines of 8 comma-separated integers, row 0 = top
    IFstream is(csvPath);
    if (!is.good())
    {
        FatalErrorInFunction
            << "Cannot open topology csv: " << csvPath << exit(FatalError);
    }
    for (int r = 0; r < 8; ++r)
    {
        string line;
        is.getLine(line);
        std::vector<int> vals;
        std::size_t pos = 0;
        while (pos <= line.size())
        {
            std::size_t comma = line.find(',', pos);
            string tok = (comma == string::npos)
                ? line.substr(pos) : line.substr(pos, comma - pos);
            vals.push_back(std::stoi(tok));
            if (comma == string::npos) break;
            pos = comma + 1;
        }
        if (vals.size() != 8)
        {
            FatalErrorInFunction
                << "topology csv row " << r << " has " << vals.size()
                << " entries (expected 8)" << exit(FatalError);
        }
        for (int c = 0; c < 8; ++c)
        {
            bm_[r][c] = vals[c];
            if (bm_[r][c] != 0 && bm_[r][c] != 1)
            {
                FatalErrorInFunction
                    << "topology values must be 0 (fluid) or 1 (solid), got "
                    << bm_[r][c] << " at (r,c)=(" << r << "," << c << ")"
                    << exit(FatalError);
            }
        }
    }

    // rectangles: row r covers x in [c w, (c+1) w], y in [H-(r+1) w, H-r w]
    for (int r = 0; r < 8; ++r)
    {
        for (int c = 0; c < 8; ++c)
        {
            Rect R{c * w_, (c + 1) * w_, H_ - (r + 1) * w_, H_ - r * w_};
            if (bm_[r][c] == 1) solidRects_.push_back(R);
            else                fluidRects_.push_back(R);
        }
    }
    for (const auto& R : portRects) fluidRects_.push_back(R);

    if (solidRects_.empty())
    {
        FatalErrorInFunction
            << "topology contains no solid cells" << exit(FatalError);
    }
}

bool BitmapTopology::insideSolid(const vector& p) const
{
    const double px = p.x();
    const double py = p.y();
    if (px < 0.0 || px > H_ || py < 0.0 || py > H_)
    {
        return false;  // ports and everything outside the chamber are fluid
    }
    int c = static_cast<int>(px / w_); if (c < 0) c = 0; if (c > 7) c = 7;
    int r = static_cast<int>((H_ - py) / w_); if (r < 0) r = 0; if (r > 7) r = 7;
    return bm_[r][c] == 1;
}

double BitmapTopology::signedDistance(
    const vector& p, vector& q, vector& n) const
{
    const double px = p.x();
    const double py = p.y();

    if (!insideSolid(p))
    {
        double best = 1e300, qx = px, qy = py;
        for (const auto& R : solidRects_)
        {
            double d, bx, by;
            distToRect(px, py, R, d, bx, by);
            if (d < best) { best = d; qx = bx; qy = by; }
        }
        q = vector(qx, qy, p.z());
        double m = std::sqrt((px - qx) * (px - qx) + (py - qy) * (py - qy));
        if (m < 1e-30)
        {
            n = vector(0, 0, 0);  // degenerate: on the surface
        }
        else
        {
            n = vector((px - qx) / m, (py - qy) / m, 0);
        }
        return best;  // sigma > 0 in fluid
    }

    // inside solid: distance to the fluid set; normal points TOWARD fluid
    double best = 1e300, qx = px, qy = py;
    for (const auto& R : fluidRects_)
    {
        double d, bx, by;
        distToRect(px, py, R, d, bx, by);
        if (d < best) { best = d; qx = bx; qy = by; }
    }
    q = vector(qx, qy, p.z());
    double m = std::sqrt((px - qx) * (px - qx) + (py - qy) * (py - qy));
    if (m < 1e-30)
    {
        n = vector(0, 0, 0);
    }
    else
    {
        n = vector((qx - px) / m, (qy - py) / m, 0);
    }
    return -best;  // sigma < 0 in solid
}
