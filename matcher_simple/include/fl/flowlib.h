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
 * Module      : FlowLib
 * Class       : FlowLib
 * Language    : C++
 * Description : Definition of the FlowLib interfaces
 *
 * Author     : Manuel Werlberger
 * EMail      : werlberger@icg.tugraz.at
 *
 */

#ifndef FLOWLIB_H
#define FLOWLIB_H

#include <vector>
#include <map>

// defines and some config stuff
#include "fldefs.h"

// LOCAL INCLUDES
//
#include <iu/iudefs.h>
#include <iu/iucontainers.h>
#include "parameters.h"

namespace fl {

// forward declarations (in fl namespace)
class Pyramid;

/**  FlowLib Interface.
 *
 * TODO: #include "XX.h" <BR>
 * TODO: -llib
 *
 * TODO: A longer description.
 *
 * \see FlowLib ...
 */
class FL_DLLAPI FlowLib
{
public:

  // LIFECYCLE
  //
  /** Default constructor.
   */
  FlowLib(int verbose = 0);

  /** Destructor.
   */
  ~FlowLib();

  /** Resets the FlowLib to an initial state.
   * \note The parameters are not reset; all images (pyramids) are deleted;
   */
  void reset();

  // OPERATIONS
  //
  /** Initializes needed data structures.
   *
   * Initializes needed data strucutres for further processing. The desired method has
   * to be set before this function is called.
   */
  void initData();

  /** Resets the information about the available GPU.
   * This function should be called if FlowLib Instance was created and afterwards the default CUDA device was changed.
   */
  void updateGpuInformation();

  /** Set two input images (either host or device) for flow calculations.
   *
   * @param input1 Pointer to the first input image.
   * @param input2 Pointer to the second input image.
   *
   * \return Ready-state of FlowLib for flow calculations.
   *
   * \note That the memory layout of the 2 input images has to be the same.
   */
  bool setInputImages(iu::Image* input1, iu::Image* input2);

  /** Sets an input image for flow calculations.
   *
   * @param input Pointer to the image (either host or device).
   *
   * \return Ready-state of FlowLib for flow calculations.
   */
  bool setInputImage(iu::Image* input1);


  /** Shifts images (and internal pyramids) -- for two images the images are swapped
   */
  bool shiftImages();

  /** Calculates the Optical Flow
   */
  bool calculate();

  /* *******
    More advanced functions follow here
   ********/

  /** Sets an occlusion (or outlier) map where no dataterm can be computed.
   * @param map Image indicating the outliers/occlusions where no data information is available; 1.0f -> occlusion.
   */
  void setOcclusionMap(iu::ImageGpu_32f_C1* map);

  /** Computes the confidence based on the given type.
   * @param confidence Confidence of calculated flow at the given pixel.
   * @param confidence_type Type of confidence computation.
   * @params level desired level for computing the motion confidence.
   */
  void computeConfidence(iu::ImageGpu_32f_C1* confidence, ConfidenceType confidence_type=WARPED_CONFIDENCE, unsigned int level=0);

  /** Computes the geometric confidence of the estimated optical flow.
   * @param confidence Confidence of calculated flow at the given pixel.
   * @param n_sigma_window Statistical area between mean+-n*sigma. (eg. n=2 should contain approx. 95% of the values should lie in this window)
   */
  void computeGeometricConfidence(iu::ImageGpu_32f_C1* confidence, float n_sigma_window=1.5, unsigned int level=0);

  /** Computes the occluded pixels in the reference frame based on the mapping uniqueness criterion.
   * @param occlusion Pointer to image where the occluded pixels are marked with 1.0f.
   */
  void computeMappingConfidence(iu::ImageGpu_32f_C1* occlusion, unsigned int level=0);

  /** Compares the warped image with the actual input image and marks the difference pixels as low confident.
   * @param confidence Confidence based on the comparison of warped and input image.
   * \note Confidence values < 0 mark the regions that get occluded and > 0 the regions that appear again.
   */
  void computeWarpedConfidence(iu::ImageGpu_32f_C1* confidence, float n_sigma_window=4.0, unsigned int level=0);

  /** Initialization of flow vecturs u (x-disparities).
   *
   * @param level Level thats going to be initialized.
   * @param u Device image (32-bit) with flow vectors used for initialization.
   * @param also_init_finer If TRUE also finer levels are going to be initialized. Otherwise only the specified level is initialized.
   */
  void initU(int level, iu::ImageGpu_32f_C1* u, bool also_init_finer = true);

  /** Initialization of flow vecturs v (y-disparities).
   *
   * @param level Level thats going to be initialized.
   * @param v Device image (32-bit) with flow vectors used for initialization.
   * @param also_init_finer If TRUE also finer levels are going to be initialized. Otherwise only the specified level is initialized.
   */
  void initV(int level, iu::ImageGpu_32f_C1* v, bool also_init_finer = true);

  /** Prior knowledge with a scale paramter that tells the algorithm when the prior u/v have to be set.
   * @param u Device image (32-bit) with flow vector x-component u for initialization.
   * @param v Device image (32-bit) with flow vector y-component v for initialization.
   * @param radius Scale in terms of blob radius where the structure should be initalized.
   */
  void setPriorFlow(iu::ImageGpu_32f_C1* u, iu::ImageGpu_32f_C1* v, iu::ImageGpu_32f_C1* radius);

  // ACCESS
  //
  /** Returns a reference to the parameter class.
   * \note This is used to change and lookup any settings for the flow calculation.
   */
  fl::Parameters& parameters();

  /** Returns a const reference to the parameter class.
   * \note This is used to lookup any settings for the flow calculation.
   */
  const fl::Parameters& const_parameters() const;

  /** Copies pixel values of the resultand (u) (x-disparities) to a gpu image.
   * @param[in] level Pyramid level that is reffered to.
   * @param[out] dst Pointer to the image where values of u are copied to (host or device).
   *
   * \return State of th copy process.
   * \note No data is copied when the level lies outside the valid range of pyramid levels or no data is initialized.
   */
  bool getU_32f_C1(int level, iu::ImageGpu_32f_C1 *dst);

  /** Copies pixel values of the resultand (v) (y-disparities) to a gpu image.
   * @param[in] level Pyramid level that is reffered to.
   * @param[out] dst Pointer to the image where values of u are copied to.
   *
   * \return State of th copy process.
   * \note No data is copied when the level lies outside the valid range of pyramid levels or no data is initialized.
   */
  bool getV_32f_C1(int level, iu::ImageGpu_32f_C1 *dst);

  /** Upsamples u to fit dst and applies a bilateral filter to enhance the edges according to the given prior image.
   * @param[out] dst Pointer to the image where values of u are copied to.
   * @param[in] prior Pointer to the image acting as a prior for enhancing the edges
   */
  void getScaledU_32f_C1(iu::ImageGpu_32f_C1 *dst, iu::ImageGpu_32f_C1 *prior=NULL,
                         const float sigma_spatial=8.0f, const float sigma_range=0.01f, const int radius=5);

  /** Upsamples v to fit dst and applies a bilateral filter to enhance the edges according to the given prior image.
   * @param[out] dst Pointer to the image where values of v are copied to.
   * @param[in] prior Pointer to the image acting as a prior for enhancing the edges
   */
  void getScaledV_32f_C1(iu::ImageGpu_32f_C1 *dst, iu::ImageGpu_32f_C1 *prior=NULL,
                         const float sigma_spatial=8.0f, const float sigma_range=0.01f, const int radius=5);


  /** Copies illumination changes (c) between the two input images to the given buffer \a dst
   * @param[in] level Pyramid level that is reffered to.
   * @param[out] dst Pointer to the image where the values of c are copied to.
   *
   * \return TRUE if an illumination estimate is available and FALSE if not.
   * \note No data is copied when the level lies outside the valid range of pyramid levels or no data is initialized.
   */
  bool getCompensation_32f_C1(int level, iu::ImageGpu_32f_C1 *dst);

  /** Returns an RGBA color representation (Middlebury color rep.) of the current optical flow.
   * @param[in] level Pyramid level that is reffered to.
   * @param[out] cflow Color-coded optical flow field.
   * @param[in] normalize_max_flow Optional parameter that normalizes the color-coded flow towards the given maximum magnitude.
   */
  bool getColorFlow_8u_C4(int level, iu::ImageGpu_8u_C4* cflow, float normalize_max_flow=0.0f);

  /** Returns a list of available models.
   */
  const std::vector<std::string>& getAvailableModels();

  /** Returns a map with model -> string correspondences.
   */
  const std::map<fl::Model, std::string>& getModelToStringMap();

  /** Returns a map with string -> model correspondences.
   */
  const std::map<std::string, fl::Model>& getStringToModelMap();

  /** Returns the model enum id from the given model string.
   * @param[in] Model as readable string.
   * \return Model as fl::Model Enum id. If model is unknown, fl::UNDEFINED_MODEL is returned.
   */
  fl::Model getModelId(std::string model_string);


  /** Returns the size of the \a n-th level.
   * @param[in] level Pyramid level that is reffered to.
   * @param[out] size Size of the n-th level.
   */
  IuSize getSize(const int& level);

  /** Returns the number of set input images (pyramids)
   */
  unsigned int getNumImages();

  /** Returns the number of set needed input images for the currently set model
   */
  unsigned int getNeededNumImages();

  /** Writes the warped (moving) image to \a dst.
   * @param[in] level Pyramid level that is reffered to.
   * @param[out] dst Pointer to the image where the warped image is written to.
   * @param[in] ill_correction Optional flag if warped image should be illumination corrected or not. [Default = true]
   * \throw IuException
   */
  void getWarpedImage_32f_C1(int level, iu::ImageGpu_32f_C1* warped_image, bool ill_correction=true);

//  /** Writes the warped (moving) image of level 0 to \a dst.
//   * @param[in] level Pyramid level that is reffered to.
//   * @param[out] dst Pointer to the image where the warped image is written to.
//   */
//  bool getWarpedImage_32f_C1(int level, iu::ImageGpu_32f_C1* warped_image);

//  /** Writes the warped  and illumination corrected(moving) image to \a dst.
//   * @param[in] level Pyramid level that is reffered to.
//   * @param[out] dst Pointer to the array where the warped image is written to.
//   * @param[in] dst_pitch Distance in bytes between starts of consecutive lines in \a dst.
//   * @param[in] size  Size of the array \dst in pixels.
//   */
//  bool getWarpedIlluminationCorrectedImage_32f_C1(int level, iu::ImageGpu_32f_C1* warped_image);

//  /** Warps the given image \a src and writes the result to \a dst.
//   * @param[in] src Input image that should be warped
//   * @param[in] src_pitch Distance in bytes between starts of consecutive lines in \a src.
//   * @param[out] dst Pointer to the array where the warped image is written to.
//   * @param[in] dst_pitch Distance in bytes between starts of consecutive lines in \a dst.
//   * @param[in] size  Size of the array \dst in pixels.
//   */
//  bool warpImage_32f_C1(float* src, size_t src_pitch,
//                        float* dst, size_t dst_pitch, IuSize size);

//  /** Warps the given image \a src and writes the result to \a dst.
//   * @param[in] src Input image that should be warped
//   * @param[in] src_pitch Distance in bytes between starts of consecutive lines in \a src.
//   * @param[out] dst Pointer to the array where the warped image is written to.
//   * @param[in] dst_pitch Distance in bytes between starts of consecutive lines in \a dst.
//   * @param[in] size  Size of the array \dst in pixels.
//   */
//  bool warpImage_32f_C3(float* src, size_t src_pitch,
//                        float* dst, size_t dst_pitch, IuSize size);

//  /** Warps the given image \a src and writes the result to \a dst.
//   * @param[in] src Input image that should be warped
//   * @param[in] src_pitch Distance in bytes between starts of consecutive lines in \a src.
//   * @param[out] dst Pointer to the array where the warped image is written to.
//   * @param[in] dst_pitch Distance in bytes between starts of consecutive lines in \a dst.
//   * @param[in] size  Size of the array \dst in pixels.
//   */
//  bool warpImage_32f_C4(float* src, size_t src_pitch,
//                        float* dst, size_t dst_pitch, IuSize size);

//  /** Warps the given image \a src, corrects its illumination and writes the result to \a dst.
//   * @param[in] src Input image that should be warped
//   * @param[in] src_pitch Distance in bytes between starts of consecutive lines in \a src.
//   * @param[out] dst Pointer to the array where the warped image is written to.
//   * @param[in] dst_pitch Distance in bytes between starts of consecutive lines in \a dst.
//   * @param[in] size  Size of the array \dst in pixels.
//   */
//  bool warpImageAndCorrectIllumination_32f_C1(float* src, size_t src_pitch,
//                                              float* dst, size_t dst_pitch, IuSize size);
  /* *************************************************************************
    BE AWARE THAT THE NEXT FEW FUNCTION JUST RETURN A POINTER TO THE IMAGES
    USED FOR CALCULATIONS. DO NOT MESS AROUND WITH THE IMAGES!
  ************************************************************************* */

  /** Returns the pointer to the NPP image at present.
   * @param[in] num Image number. Normally 0 or 1. e.g. 0 .. first image; 1 .. second image;
   * @param[in] level Level id of the desired image.
   */
  iu::ImageGpu_32f_C1* getImage(int num, int level);

  /** Returns a pointer stl::deque containing all the input images.
   */
  iu::ImageDeque* getImageDeque();

  /** Returns a pointer stl::deque containing all the image pyramids.
   */
  iu::ImagePyramidDeque* getImagePyramidDeque();

  /** Returns the pointer to the disparities u.
   * @param[in] level Level id of the desired image.
   */
  iu::ImageGpu_32f_C1* getU(int level);

  /** Returns the pointer to the disparities v.
   * @param[in] level Level id of the desired image.
   */
  iu::ImageGpu_32f_C1* getV(int level);

  /** Returns a const Pointer to the estimated illumination changes.
   * @param[in] level Pyramid level that is reffered to.
   */
  iu::ImageGpu_32f_C1* getC(int level);

  /** Returns a const Pointer to the estimated acceleration (x-direction)
   * @param[in] level Pyramid level that is reffered to.
   */
  iu::ImageGpu_32f_C1* getAx(int level);

  /** Returns a const Pointer to the estimated acceleration (y-direction)
   * @param[in] level Pyramid level that is reffered to.
   */
  iu::ImageGpu_32f_C1* getAy(int level);


  /** TODO
    */
  iu::LinearHostMemory_32f_C1* getPrimalEnergies(int level);
  iu::LinearHostMemory_32f_C1* getDualEnergies(int level);

  /** Writes the current flow field into a file.
   * @param[in] filename Filename where the flo file is saved.
   * @param[in] level Level id of the desired image.
   */
  void writeFloFile(const std::string& filename, unsigned int level);

  /** Writes a scaled version of the current flow field into a file.
   * @param[in] filename Filename where the flo file is saved.
   * @param[in] size Desired size.
   * @param[in] prior Prior for bilateral filter to enhance edges of the flow field.
   */
  void writeScaledFloFile(const std::string& filename, IuSize size, iu::ImageGpu_32f_C1* prior);

  // INQUIRY
  //
  /** Tells if FlowLib data structures are initialized.
   * \note Initialization is done automatically when first input image is set.
   */
  bool isInitialized() const;

  /** Tells if FlowLib is ready to process the provided input data.
   */
  bool isReady() const;

  /** Availability of a cuda capable GPU to parallelize the calculations.
   * The current version of the FlowLib needs a compute capability >= 1.1
   */
  bool isCompatibleGpuAvailable() const;

protected:

private:

  // Copy constructor and asignment operator kept private for the moment!
  //
  /** Copy constructor.
   * @param from The value to copy to this object.
   */
  FlowLib(const FlowLib& from);

  /** Assignment operator.
    * @param from THe value to assign to this object.
    * \return A reference to this object.
    */
  FlowLib& operator=(const FlowLib& from);

  /** Initializes GPU memory for the color gradient.
   */
  void initColorGradient();


  // MEMBER VARIABLES
  //
  fl::Parameters params_; /**< Parameters. */
  bool initialized_; /**< Flag if data structures are initialized and valid. Initialization is done automatically when first input image is set. */
  bool ready_; /**< Ready state for processing input data. */

  fl::Pyramid* pyramid_; /**< Pointer to data structure. Responsible for data handling and processing! */

  iu::LinearDeviceMemory_32f_C4* d_gradient_; /**< Color gradient to create color representation of flow field. */

  std::vector<std::string> model_string_vector_;
  std::map<std::string, fl::Model>  model_string_to_id_map_;
  std::map<fl::Model, std::string> model_id_to_string_map_;

  // stuff that must be safed for some reason
  iu::ImageGpu_32f_C1* occlusion_map_;
};

} // namespace fl

#endif // FLOWLIB_H
