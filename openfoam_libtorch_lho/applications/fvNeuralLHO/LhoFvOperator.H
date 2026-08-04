#ifndef LhoFvOperator_H
#define LhoFvOperator_H

#include "fvMesh.H"
#include "volFields.H"
#include "TorchCompat.H"

class LhoFvOperator
{
public:
    LhoFvOperator(
        const Foam::fvMesh& mesh,
        Foam::label dimension,
        const Foam::word& potentialMode,
        Foam::scalar omegaY,
        Foam::scalar kineticScale = 0.5,
        Foam::scalar potentialScale = 0.5);

    const torch::Tensor& K() const { return K_; }
    const torch::Tensor& mass() const { return mass_; }
    const torch::Tensor& x() const { return x_; }
    const torch::Tensor& y() const { return y_; }
    const torch::Tensor& potential() const { return potential_; }
    Foam::label nCells() const { return nCells_; }
    Foam::label dim() const { return dim_; }
    const Foam::word& potentialMode() const { return potentialMode_; }
    Foam::scalar omegaY() const { return omegaY_; }

    void writeDiagnostics(const Foam::fileName& path) const;

private:
    void assembleKinetic();
    void assemblePotential();

    const Foam::fvMesh& mesh_;
    Foam::label dim_;
    Foam::word potentialMode_;
    Foam::scalar omegaY_;
    Foam::scalar kineticScale_;
    Foam::scalar potentialScale_;
    Foam::label nCells_;

    torch::Tensor K_;
    torch::Tensor mass_;
    torch::Tensor x_;
    torch::Tensor y_;
    torch::Tensor potential_;
};

#endif
