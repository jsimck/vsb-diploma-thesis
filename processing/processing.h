#ifndef VSB_SEMESTRAL_PROJECT_PROCESSING_H
#define VSB_SEMESTRAL_PROJECT_PROCESSING_H

#include <opencv2/core/mat.hpp>
#include "../core/template.h"
#include "../core/match.h"

namespace tless {
    // Lookup tables
    static const int NORMAL_LUT_SIZE = 20, DEPTH_LUT_SIZE = 5;
    static const uchar DEPTH_LUT[DEPTH_LUT_SIZE] = {1, 2, 4, 8, 16};
    static const uchar NORMAL_LUT[NORMAL_LUT_SIZE][NORMAL_LUT_SIZE] = {
            {32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64, 64,  64,  64,  64,  128, 128, 128, 128, 128},
            {32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64,  64,  64,  128, 128, 128, 128, 128, 128},
            {32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64, 64,  64,  64,  128, 128, 128, 128, 128, 128},
            {32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64,  64,  128, 128, 128, 128, 128, 128, 128},
            {32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64,  64,  128, 128, 128, 128, 128, 128, 128},
            {32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64, 64,  64,  128, 128, 128, 128, 128, 128, 128},
            {16, 32, 32, 32, 32, 32, 32, 32, 32, 64, 64, 64,  128, 128, 128, 128, 128, 128, 128, 128},
            {16, 16, 16, 32, 32, 32, 32, 32, 32, 64, 64, 64,  128, 128, 128, 128, 128, 128, 1,   1},
            {16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 64, 128, 128, 128, 128, 1,   1,   1,   1,   1},
            {16, 16, 16, 16, 16, 16, 16, 16, 32, 32, 64, 128, 128, 1,   1,   1,   1,   1,   1,   1},
            {16, 16, 16, 16, 16, 16, 16, 16, 16, 16, 1,  1,   1,   1,   1,   1,   1,   1,   1,   1},
            {16, 16, 16, 16, 16, 16, 16, 16, 8,  8,  4,  2,   2,   1,   1,   1,   1,   1,   1,   1},
            {16, 16, 16, 16, 16, 16, 8,  8,  8,  8,  4,  2,   2,   2,   2,   1,   1,   1,   1,   1},
            {16, 16, 16, 8,  8,  8,  8,  8,  8,  4,  4,  4,   2,   2,   2,   2,   2,   2,   1,   1},
            {16, 8,  8,  8,  8,  8,  8,  8,  8,  4,  4,  4,   2,   2,   2,   2,   2,   2,   2,   2},
            {8,  8,  8,  8,  8,  8,  8,  8,  4,  4,  4,  4,   4,   2,   2,   2,   2,   2,   2,   2},
            {8,  8,  8,  8,  8,  8,  8,  8,  4,  4,  4,  4,   4,   2,   2,   2,   2,   2,   2,   2},
            {8,  8,  8,  8,  8,  8,  8,  8,  4,  4,  4,  4,   4,   2,   2,   2,   2,   2,   2,   2},
            {8,  8,  8,  8,  8,  8,  8,  4,  4,  4,  4,  4,   4,   4,   2,   2,   2,   2,   2,   2},
            {8,  8,  8,  8,  8,  8,  8,  4,  4,  4,  4,  4,   4,   4,   2,   2,   2,   2,   2,   2}
    };

    /**
     * @brief Computes quantized surface normals from 16-bit depth image.
     *
     * @param[in]  src           Source 16-bit depth image (in mm)
     * @param[out] dst           Destination 8-bit image, where each bit represents one bin of view cone
     * @param[in]  fx            Camera focal length in X direction
     * @param[in]  fy            Camera focal length in Y direction
     * @param[in]  maxDepth      Ignore pixels beyond this depth
     * @param[in]  maxDifference When computing surface normals, ignore contributions of
     *                           pixels whose depth difference with central pixel is above this threshold
     */
    void quantizedNormals(const cv::Mat &src, cv::Mat &dst, float fx, float fy, int maxDepth, int maxDifference);

    /**
     * @brief Generates binary image of visible depth edgels, detected in depth image within (min, max) depths.
     *
     * @param[in]  src      Source 16-bit depth image (in mm)
     * @param[out] dst      8-bit uchar binary image containing 1 where edgels arise in depth image
     * @param[in]  minDepth Ignore pixels with depth lower then this threshold
     * @param[in]  maxDepth Ignore pixels with depth higher then this threshold
     * @param[in]  minMag   Ignore pixels with edge magnitude lower than this
     */
    void depthEdgels(const cv::Mat &src, cv::Mat &dst, int minDepth, int maxDepth, int minMag);

    /**
     * @brief Finds normalization (error) factor, to help define range in which the given depth value can be.
     *
     * @param[in] depth         Depth value to get normalization factor for
     * @param[in] errorFunction Normalization function (vector of 2 values, first is upperBound, second is normalization factor)
     * @return                  Normalization factor defined in error function for given depth range
     */
    float depthNormalizationFactor(float depth, const std::vector<cv::Vec2f>& errorFunction);

    /**
     * @brief Returns quantized depth into one of 5 bins defined in ranges.
     *
     * @param[in] depth  Depth value (32-bit signed int)
     * @param[in] ranges Vector of ranges defining bounds for each quantization bin
     * @return           Quantized depth (1 | 2 | 4 | 8 | 16)
     */
    uchar quantizeDepth(int depth, const std::vector<cv::Range> &ranges);

    /**
     * @brief Remaps white and black colors of HSV image to yellow and blue for HSV recognition
     *
     * @param[in]  src        Input 8-bit HSV image to normalize
     * @param[out] dst        Normalized 8-bit 1-channel image containing hue values from HSV
     * @param[in]  value      Value threshold, values below this threshold [blacks] are mapped to blue color
     * @param[in]  saturation Saturation threshold, values below this and above value threshold [white] are mapped to yellow color
     */
    void normalizeHSV(const cv::Mat &src, cv::Mat &dst, uchar value = 22, uchar saturation = 31);

    /**
     * @brief Applies non-maxima suppression to matches, removing matches with large overlap and lower score.
     *
     * This function calculates overlap between each window, if the overlap is > than [maxOverlap]
     * we only retain a match with higher score.
     *
     * @param[in,out] matches    Input/output array of matches to apply non-maxima suppression on
     * @param[in]     maxOverlap Max allowed overlap between two matched bounding boxes
     */
    void nonMaximaSuppression(std::vector<Match> &matches, float maxOverlap);

    /**
     * @brief Finds edges in gray image using sobel operator (applies erosion and gaussian blur in pre-processing).
     *
     * @param[in]  src   8-bit gray source image
     * @param[out] dst   8-bit image with detected edges
     * @param[in]  kSize Size of a kernel used in Gaussian blur in pre-processing
     */
    void filterEdges(const cv::Mat &src, cv::Mat &dst, int kSize = 5);

    /**
     * @brief Computes and quantizes gradient orientations over RGB scene
     *
     * @param[in]  src    8-bit 3-channel RGB image to compute gradients on
     * @param[out] dst    8-bit image map of quantized gradient orientations
     * @param[in]  minMag Minimum edge magnitude to consider as valid and compute orientation for
     */
    void quantizedGradients(const cv::Mat &src, cv::Mat &dst, float minMag);

    /**
     * @brief Quantizes orientation gradients into 5 bins (0-180deg) based on their angle
     *
     * @param[in] deg Degree of the gradient orientation
     * @return        One of quantized orientations 1, 2, 4, 8, 16
     */
    uchar quantizeGradientOrientation(float deg);
}

#endif