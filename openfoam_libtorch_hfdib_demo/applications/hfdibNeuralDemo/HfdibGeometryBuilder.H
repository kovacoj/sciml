#ifndef HfdibGeometryBuilder_H
#define HfdibGeometryBuilder_H

#include "fvMesh.H"
#include "volFields.H"
#include "BitmapTopology.H"
#include "HfdibGeometryData.H"

// Builds sigma / lambda / fluidMask / interfaceMask / solidMask from the
// topology bitmap and extracts the per-interface-cell surface geometry.
//
// Conventions (paper / openHFDIB, NOT the tpfm_unet README wording):
//   sigma > 0 in fluid, sigma < 0 in solid
//   lambda = 0 fluid, lambda = 1 solid, 0 < lambda < 1 interface
//   interface cells:  lambda = 0.5 * (1 - tanh(sigma / V_P^(1/3)))
//
// A cell is classified "intersected by the interface" when its 4 in-plane
// corner vertices are not all on the same side of the rectilinear solid
// (exact for this geometry: bitmap edges coincide with cell boundaries).
class HfdibGeometryBuilder
{
public:
    HfdibGeometryBuilder(const Foam::fvMesh& mesh,
                         const BitmapTopology& topo);

    const HfdibGeometryData& geometry() const { return geom_; }
    const std::vector<double>& lambda() const { return lambda_; }
    const std::vector<double>& sigma() const { return sigma_; }

    // per-cell classification masks in {0,1}
    const std::vector<double>& fluidMask() const { return fluidMask_; }
    const std::vector<double>& interfaceMask() const { return interfaceMask_; }
    const std::vector<double>& solidMask() const { return solidMask_; }

    // write OpenFOAM fields + a plain fields.csv for plotting
    void writeFields(const Foam::fileName& postProcDir, const Foam::word& timeName) const;
    void writeGeometryCsv(const Foam::fileName& postProcDir) const;

private:
    void compute();

    const Foam::fvMesh& mesh_;
    const BitmapTopology& topo_;

    HfdibGeometryData geom_;
    std::vector<double> lambda_;
    std::vector<double> sigma_;
    std::vector<double> fluidMask_;
    std::vector<double> interfaceMask_;
    std::vector<double> solidMask_;
};

#endif
