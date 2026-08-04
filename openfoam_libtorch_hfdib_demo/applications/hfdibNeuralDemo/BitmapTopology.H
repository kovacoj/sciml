#ifndef BitmapTopology_H
#define BitmapTopology_H

#include "vector.H"
#include "fileName.H"
#include <vector>

// Exact signed-distance geometry of the 8x8 topology bitmap (0 = fluid,
// 1 = solid; row 0 = top of the chamber).  The chamber is [0, H] x [0, H];
// the four short port corridors are always fluid and are appended as extra
// fluid rectangles so solid-side distances near ports are handled properly.
//
// Provides, for any point p in the domain:
//   sigma(p)   signed distance to the solid surface (>0 fluid, <0 solid)
//   q(p)       nearest point on the solid surface  (sigma(q) = 0)
//   n(p)       unit normal pointing from the surface toward the fluid
//
// For a point exactly on a boundary (measure zero: cell centres sit on odd
// multiples of h/2 while boundaries sit on multiples of the bitmap cell
// width), the normal is undefined; callers must handle |sigma| ~ 0.
class BitmapTopology
{
public:
    struct Rect
    {
        double x0, x1, y0, y1;
    };

    BitmapTopology
    (
        const Foam::fileName& csvPath,
        double chamberWidth,                 // H
        const std::vector<Rect>& portRects   // extra always-fluid rects
    );

    bool insideSolid(const Foam::vector& p) const;

    // sigma, nearest surface point q, fluid-side normal n
    double signedDistance(const Foam::vector& p, Foam::vector& q,
                          Foam::vector& n) const;

    int bitmap(int r, int c) const { return bm_[r][c]; }
    double cellWidth() const { return w_; }
    double chamberWidth() const { return H_; }

private:
    // distance from point to rectangle, point assumed OUTSIDE the rectangle
    static void distToRect(double px, double py, const Rect& R,
                           double& d, double& qx, double& qy);

    int bm_[8][8];
    double H_;
    double w_;
    std::vector<Rect> solidRects_;
    std::vector<Rect> fluidRects_;
};

#endif
