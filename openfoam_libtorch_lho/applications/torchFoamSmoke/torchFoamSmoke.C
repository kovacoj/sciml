#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "TorchCompat.H"

#ifdef TypeName
    #define AV_OPENFOAM_TYPENAME_RESTORED 1
#else
    #define AV_OPENFOAM_TYPENAME_RESTORED 0
#endif

#include <iostream>

using namespace Foam;

int main(int argc, char *argv[])
{
    argList::addNote
    (
        "torchFoamSmoke\n"
        "OpenFOAM + LibTorch compatibility test"
    );

    static_assert(
        AV_OPENFOAM_TYPENAME_RESTORED == 1,
        "TorchCompat.H failed to restore OpenFOAM TypeName macro"
    );

    #include "setRootCase.H"
    #include "createTime.H"
    #include "createMesh.H"

    torch::manual_seed(1234);
    torch::set_num_threads(1);

    auto options =
        torch::TensorOptions()
            .dtype(torch::kFloat64)
            .device(torch::kCPU)
            .requires_grad(true);

    auto x = torch::tensor({1.0, 2.0, 3.0}, options);
    auto loss = x.square().sum();
    loss.backward();

    auto expected = 2.0*x.detach();
    auto ok = torch::allclose(
        x.grad(),
        expected,
        1.0e-12,
        1.0e-12
    );

    Info << "OpenFOAM cells: " << mesh.nCells() << nl;
    std::cout << "Tensor: " << x << std::endl;
    std::cout << "Gradient: " << x.grad() << std::endl;

    if (!ok)
    {
        FatalErrorInFunction
            << "LibTorch autograd smoke test failed"
            << exit(FatalError);
    }

    Info << "OpenFOAM TypeName macro restored: yes" << nl;
    Info << "OpenFOAM + LibTorch compatibility test passed" << nl;
    return 0;
}
