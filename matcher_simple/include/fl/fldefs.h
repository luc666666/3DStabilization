/*
 * Copyright (c) ICG. All rights reserved.
 *
 * Institute for Computer Graphics and Vision
 * Graz University of Technology / Austria
 *
 *
 * This software is distributed WITHOUT ANY WARRANTY; without even
 * the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 * PURPOSE.  See the above copyright notices for more information.
 *
 *
 * Project     : VMLibraries
 * Module      : FlowLib
 * Class       : misc
 * Language    : C++/CUDA
 * Description : Definition of Macros, structs, etc.
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef FL_DEFS_H
#define FL_DEFS_H

#include <cstddef>
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <iudefs.h>
#include <vector>
#include <string>

namespace fl {

// CONSTANTS
//
static const int MAX_PYRAMIDAL_LEVELS = 100;

//-----------------------------------------------------------------------------
/* Shared lib macros for windows dlls
*/
#ifdef WIN32
#ifdef _MSC_VER
#pragma warning( disable : 4251 ) // disable the warning about exported template code from stl
#endif // _MSC_VER

#ifdef FL_USE_STATIC
#define FL_DLLAPI
#else
#ifdef FL_DLL_EXPORTS
#define FL_DLLAPI __declspec(dllexport)
#else
#define FL_DLLAPI __declspec(dllimport)
#endif
#endif
#else
#define FL_DLLAPI
#endif

// TYPEDEFS
//

//-----------------------------------------------------------------------------
/** Specification of the available models
 * @note The entries in the model_list and EnumModel MUST have the same ordering so that building the std::map structures in flowlib.cpp work!!
 * @note IF YOU MAKE CHANGES HERE, ALSO RETHINK THE CHECK IN FlowLib::checkModelCompatibilty()!
 * @note IF YOU MAKE CHANGES HERE, THE HL1 MODEL HAS TO BE THE FIRST MODEL THAT USE THE GPU BECAUSE OF SOME INTERNAL TESTS!!!
 */
static const std::string model_list[]= {
  // Huber models
  std::string("HL1"),
  std::string("FAST_HL1"),
  std::string("FAST_HL1_TENSOR"),
  std::string("INTERPOLATE_FAST_HL1"),
  std::string("HL1_PRECOND"),
  std::string("HL1_WEIGHTED_PRECOND"),
  std::string("HL1_DIRECTED_PRECOND"),
  std::string("HL1_TENSOR_PRECOND"),
  std::string("HL1_COMPENSATION"),
  std::string("HL1_COMPENSATION_PRECOND"),
  std::string("HL1_TENSOR_COMPENSATION_PRECOND"),
  std::string("HGRAD"),
  std::string("HGRAD_PRECOND"),
  std::string("HGRAD_TENSOR_PRECOND"),
  // Huber quadfit models
  std::string("HQUADFIT"),
  std::string("HQUADFIT_HESSIAN"),
  std::string("HQUADFIT_TENSOR"),
  // nonlocal models
  std::string("NLHL1"),
  std::string("NLHL1_COMPENSATION_PRECOND"),
  std::string("NLHGRAD_PRECOND"),
  std::string("NLHQUADFIT"),
  std::string("TGV2L1"),
  std::string("HCENSUS"),
  std::string("HCENSUS_TENSOR_PRECOND"),
  std::string("NLHCENSUS"),
  std::string("TGV2CENSUS")
//  //
//  std::string("HL1_EDGES_ILLUMINATION"),
//  //
//  std::string("HL1_TENSOR_GRAD"),
//  std::string("HL1_3F_SYM"),
//  std::string("HL1_3F_SYM_PRECOND"),
//  std::string("TGV2_STRTEX"),
//  std::string("TGV2_GRAD"),
//  std::string("HL1_ILLUMINATION_THRESH"),
};

typedef enum EnumModel
{
  // Huber models
  HL1, /**< Huber + L1 */
  FAST_HL1, /**< fast Huber + L1 (single kernel for primal+dual update) */
  FAST_HL1_TENSOR, /**< fast tensor directed Huber + L1 */
  INTERPOLATE_FAST_HL1, /**< fast Huber + L1 with warpint to intermediate position */
  HL1_PRECOND, /**< Huber + L1 */
  HL1_WEIGHTED_PRECOND, /**< weighted Huber + L1 */
  HL1_DIRECTED_PRECOND, /**< directed Huber + L1 */
  HL1_TENSOR_PRECOND, /**< tensor directed Huber + L1 */
  HL1_COMPENSATION, /**< Huber + L1 + compensation */
  HL1_COMPENSATION_PRECOND, /**< Huber + L1 + compensation */
  HL1_TENSOR_COMPENSATION_PRECOND, /**< tensor directed Huber + L1 + compensation */
  HGRAD, /**< Huber + grad */
  HGRAD_PRECOND, /**< Huber + grad */
  HGRAD_TENSOR_PRECOND, /**< tensor directed Huber + grad */
  // Huber quadfit models
  HQUADFIT, /**< Huber + Quadfit (approximated Hessian) */
  HQUADFIT_HESSIAN, /**< Huber + Quadfit (full Hessian) */
  HQUADFIT_TENSOR, /**< tensor directed Huber + Quadfit */
  // nonlocal huber models
  NLHL1, /**< nonlocal Huber + L1 */
  NLHL1_COMPENSATION_PRECOND,  /**< nonlocal Huber + L1 + compensation */
  NLHGRAD_PRECOND, /**< nonlocal Huber + grad */
  NLHQUADFIT, /**< nonlocal Huber + Quadfit */
  // TGV
  TGV2L1,  /**< TGV2 + L1 */
  // Census dataterm
  HCENSUS, /**< Huber + Census(L1) */
  HCENSUS_TENSOR_PRECOND, /**< tensor directed Huber + Census(L1) */
  NLHCENSUS, /**< Nonlocal Huber + Census(L1) */
  TGV2CENSUS, /**< TGV2 + Census(L1) */
  //TGV2_GRAD,
  // TODO TGV
  //TGV2_STRTEX,
  //
  //
  HL1_3F_SYM, /**< Huber-L1 model using 3 images / 2 intensity dataterms; symmetry constrained; */
  HL1_3F_SYM_PRECOND, /**< Huber-L1 model using 3 images / 2 intensity dataterms; symmetry constrained; preconditioner */
  //
  /*
    DO NOT FORGET TO MAKE ALSO AN ENTRY IN THE STRING LIST model_list
   */

  UNDEFINED_MODEL = 1000
} Model;

/** Defines the type of structure texture decomposition. */
typedef enum EnumStructureTextureDecompositionMethod {
  STR_TEX_DECOMP_OFF,   /**< No structure texture decomposition of input images. */
  STR_TEX_DECOMP_GAUSS, /**< Uses Gauss filter for structure-texture decomposition. */
  STR_TEX_DECOMP_ROF   /**< Uses ROF denoising filter for structure-texture decomposition. */
  // :TODO: STR_TEX_DECOMP_TVL1   /**< Uses TVL1 denoising filter for structure-texture decomposition. */
} StructureTextureDecompositionMethod;

/** Defines the type of dataterm used for quad fit optmiziation. */
typedef enum EnumQuadfitDataTerm {
  QUADFIT_DATATERM_AD,   /**< Absolute differences. */
  QUADFIT_DATATERM_SAD3, /**< Sum of absolute differences in a 3x3 window. */
  QUADFIT_DATATERM_SAD, /**< Sum of absolute differences. Window size is given by the radius quad_fit_windowradius. ws = 2r+1; */
  QUADFIT_DATATERM_NCC3,  /**< Cross correlation measure in a 3x3 window. */
  QUADFIT_DATATERM_NCC,  /**< Cross correlation measure. . Window size is given by the radius quad_fit_windowradius. ws = 2r+1; */
  QUADFIT_DATATERM_GRAD,  /**< Gradients dataterm. */
  QUADFIT_DATATERM_NGRAD,  /**< Normalized gradients. */
  QUADFIT_DATATERM_GRAD_INTENSITY /**< Mixed gradients and intensity dataterm. */
} QuadfitDataTerm;

/** Defines the type of confidence calculations. */
typedef enum EnumConfidenceType {
  GEOMETRIC_CONFIDENCE, /**< Compares the flow field in both(!) directions for inconsistency and maps the result to a probability distribution between 0.0f and 1.0f. */
  MAPPING_CONFIDENCE, /**< Assumes that flow vectors point to single values. If more than a single flow vector is pointing to a coordinate it is marked as occlude. */
  WARPED_CONFIDENCE /**< Compares the warped and the reference input image. */
} ConfidenceType;

/** Calculates the number of useful levels that fit \a finest_level_side on the coarsest scale.
 * @param[in] width Image width on the finest scale.
 * @param[in] height Image height on the finest scale.
 * @param[in] scale_factor Scale factor between two levels.
 * @param[in] finest_level_side Minimal size in the coarsest scale.
 * @returns Number of levels that satisfy that the shorter side on the coarsest level is smaller than finest_level_side.
 */
inline int calcNumLevels(int width, int height,
                         float scale_factor, int finest_level_side)
{
  int short_side = (width < height) ? width : height;
  float ratio = static_cast<float>(short_side) / static_cast<float>(finest_level_side);
  int nlevels = static_cast<int>(-logf(ratio)/logf(scale_factor));
  return nlevels;
}

/** Round a / b to nearest higher integer value.
 * @param[in] a Numerator
 * @param[in] b Denominator
 * @return a / b rounded up
 */
inline unsigned int divUp(unsigned int a, unsigned int b)
{
  return (a % b != 0) ? (a / b + 1) : (a / b);
}


/** define square functionality */
inline __device__ __host__ float fIUSQR(float a) { return a*a; }

} // namespace fl

#endif // FL_DEFS_H
