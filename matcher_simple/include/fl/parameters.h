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
 * Project     : FlowLib2
 * Module      : Parameters
 * Class       : fl::Parameters
 * Language    : C++
 * Description : Definition of paramter class (implemented as singleton)
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef FL_PARAMETERS_H
#define FL_PARAMETERS_H


#include <ostream>
#include <iu/iucore.h>
#include "fldefs.h"

namespace fl {
/** Parameters structure holds parameters for flow calculations.
 */
class Parameters
{
public:
  // hidden ctor, dtor, copy ctor and assign op due to singleton
  Parameters() :
    verbose(0),
    model(HL1), compute_reverse_direction(false),
    lambda(35.0f), lambda_pointwise(0), epsilon_u(0.01f), iters(10), warps(10),
    levels(fl::MAX_PYRAMIDAL_LEVELS), start_level(fl::MAX_PYRAMIDAL_LEVELS), stop_level(0),
    scale_factor(0.5f), interpolation_method(IU_INTERPOLATE_LINEAR),
    filter_median_level(1),
    regularization_tensor_weight_sigma(1.0f),
    regularization_tensor_weight_alpha(10.0f), regularization_tensor_weight_q(0.7f),
    gamma_c(0.01f), epsilon_c(0.01f),
    quadfit_dataterm(fl::QUADFIT_DATATERM_AD),
    quadfit_discretization(1.0f), quadfit_windowradius(1),
    nl_winr(3), nl_w(make_float2(0.03f,2.0f)), nl_wnl(make_float2(1.0f,0.0f)),
    alpha0(4.0f), alpha1(2.0f), census_winr(2), census_epsilon(0.005f),
    //      use_adaptive_lambda(false),
    str_tex_decomp_method(STR_TEX_DECOMP_OFF), str_tex_decomp_smoothing_amount(1.0f),
    str_tex_decomp_weight(0.8f), str_tex_decomp_rof_iterations(100),
    //      evaluate_energies(false),
    gpu_compute_capability_major(0), gpu_compute_capability_minor(0)
  {
  }

  ~Parameters()
  {
    delete lambda_pointwise;
  }

  int verbose; /**< Verbosity flag. Adjusts the amount of debug/verbose output on the console. \n Default: 0*/
  Model model; /**< Current model settings for flow calculations. */
  bool compute_reverse_direction; /**< If enabled, flow is computed in opposite direction. */
  float lambda; /**< Amount of weighting the regularization of u against the optical flow constraint. \n Default: 0.02 */
  iu::ImageGpu_32f_C1* lambda_pointwise;
  float epsilon_u; /**< Parameter for Huber regularity (for u) that models the threshold for the quadratic penalization. \n Default: 0.05 */
  unsigned int iters; /**< Number of iterations per warping step. \n Default: 10 */
  unsigned int warps; /**< Number of warps. \n Default: 5 */
  unsigned int levels; /**< Number of maximal used levels. \n Default: MAX_PYRAMIDAL_LEVELS -- auotmatically determined due to scale factor.  \n This variable effects initialization of input data!*/
  unsigned int start_level; /**< Level where calculation start. Value between levels-1 and 0 is ok. \n Default: start_level is set to coarsest level. */
  unsigned int stop_level; /** Level where calculation stop. Value between levels-1 and 0 is ok. \n Default: stop_level is set to fines level (0). */
  float scale_factor; /**< Scale factor from one level to the next. \n Default: 0.5  \n This variable effects initialization of input data!*/
  IuInterpolationType interpolation_method; /**< Interpolation method that is used for resizing input images, warping and wherever it is supported. \n Default: linear */
  unsigned int filter_median_level; /**< All levels coarser than this level use a median filter when upscaling u/v. \n Default: MAX_PYRAMIDAL_LEVELS */

  // regularization weighting
  float regularization_tensor_weight_sigma; /**< Sigma for gaussian prefilter of (fixed) input image before calculating the tensor matrix. \n Default: 1.0 */
  float regularization_tensor_weight_alpha; /**< Multiplicative weighting of edge norm for tensor calculations. \n Default: 10 */
  float regularization_tensor_weight_q; /**< Exponential weighting for edge norm for tensor calculations. \n Default: 0.5 */

  // illumination estimation
  float gamma_c; /**< Scales the value of c to be in approx the same range than the flow vectors (needed for primal-dual opt.). \n Default: 0.01 */
  float epsilon_c; /**< Parameter for Huber regularity (for c) that models the threshold for the quadratic penalization. \n Default: 0.5 */

  //  // occlusion/confidence refinement
  //  bool use_adaptive_lambda;

  // quad-fit optimization
  fl::QuadfitDataTerm quadfit_dataterm;
  float quadfit_discretization;
  int quadfit_windowradius;

  // nonlocal models
  int nl_winr; /**< The window radius(!) of the nltv neighborhood. */
  float2 nl_w; /**< 1/w.x weights the color similarity and 1/w.y the proximal fraction of the nltv weights. */
  float2 nl_wnl; /**< wnl.x weights the color/proximal weight's fraction towards a TV fraction (wnl.y) for removing small details. */

  // tgv2 weighting
  float alpha0; /**< */
  float alpha1; /**< */

  // census parameters
  int census_winr; /**< window radius to compute census transform. */
  float census_epsilon; /**< @todo: whatever this param is for. */

  // Input image preprocessing
  fl::StructureTextureDecompositionMethod str_tex_decomp_method; /**< Method of structure-texture decomposition of input images. \n Default: OFF */
  float str_tex_decomp_smoothing_amount; /**< Sets the amount of smoothing for Gauss and ROF denoising for the str-tex decomposition. \n Default: 1.0f */
  float str_tex_decomp_weight; /**< Weighting of structure vs texture for recombined images. \n Default: 0.8f */
  unsigned int str_tex_decomp_rof_iterations; /**< Sets the number of iterations when doing ROF denoising for str-tex decomposition. \n Default: 100 */

  //  // Energy calculations
  //  bool evaluate_energies;

  // GPU CONFIGURATION
  int gpu_compute_capability_major; /**< additional variable for major compute capability because npp is not aware of 2.0. */
  int gpu_compute_capability_minor; /**< additional variable for minor compute capability because npp is not aware of 2.0. */

  friend std::ostream& operator<<( std::ostream& stream, const Parameters& params );


  /** Internal parameters (nested class).
   * @warning Do not mess around with these. This will brake the calculations! (except block_size if you know what you are doing....)
   */
  class ParametersInternal
  {
  public:
    ParametersInternal() :
      linearization_step(1.0f), linearization_step_factor(1.75f),
      smallest_level_size(8), init_u(true), init_v(true), init_c(true),
      cur_level_id(UINT_MAX)
    {
    }

    static unsigned int block_size; /**< Standard block size. This is adapted when ghe gpu_compute_capability is updated. @note This is only a suggestion! */
    float linearization_step; /**< Maximum step size of linearized results. -> Should get smaller over warps. */
    float linearization_step_factor; /**< Factor that reduces linearization_step over warps. (lin_step /= factor;) */
    unsigned int smallest_level_size; /**< Denotes the minimum of the shorter side on the coarsest level. @note This one must be >= 16. */

    // flags for init variables
    bool init_u;
    bool init_v;
    bool init_c;

    unsigned int cur_level_id; /**< for internal calculations the current level id can be set here. */

  private:
    ParametersInternal(ParametersInternal const&);  // intentionally not implemented
    ParametersInternal& operator=(ParametersInternal const&);  // intentionally not implemented
  };

  ParametersInternal intern_params; /**< Inernal parameter structure. */

private:
  Parameters(Parameters const&);  // intentionally not implemented
  Parameters& operator=(Parameters const&);  // intentionally not implemented
};

} // namespace fl

#endif // FL_PARAMETERS_H
