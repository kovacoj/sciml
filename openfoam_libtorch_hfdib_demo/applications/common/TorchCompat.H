#ifndef AV_TORCH_COMPAT_H
#define AV_TORCH_COMPAT_H

/*
 * OpenFOAM defines a global function-like macro named TypeName.
 * LibTorch contains C++ identifiers that conflict with this macro.
 *
 * Save the OpenFOAM macro, temporarily remove it while processing
 * LibTorch headers, then restore it.
 */

#if defined(TypeName)
    #pragma push_macro("TypeName")
    #undef TypeName
    #define AV_RESTORE_OPENFOAM_TYPENAME
#endif

// Include the full LibTorch C++ frontend
#include <torch/torch.h>

#if defined(AV_RESTORE_OPENFOAM_TYPENAME)
    #pragma pop_macro("TypeName")
    #undef AV_RESTORE_OPENFOAM_TYPENAME
#endif

#endif
