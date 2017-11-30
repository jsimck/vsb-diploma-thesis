#ifndef VSB_SEMESTRAL_PROJECT_PROCESSING_H
#define VSB_SEMESTRAL_PROJECT_PROCESSING_H

#include <opencv2/core/mat.hpp>
#include "../core/template.h"

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
     * @brief Computes relative depths from 16-bit depth image on (triplet) given points.
     *
     * @param[in]  src    Source 16-bit depth image (in mm)
     * @param[in]  c      Center triplet point
     * @param[in]  p1     P1 triplet point
     * @param[in]  p2     P2 triplet point
     * @param[out] depths 2 value output array where: depths[0] = src.at(p1) - src.at(c); depths[1] = src.at(p2) - src.at(c);
     */
    void relativeDepths(const cv::Mat &src, cv::Point c, cv::Point p1, cv::Point p2, int *depths);

    /**
     * @brief Generates integral image of visible depth edgels, detected in depth image within (min, max) depths
     *
     * @param[in]  src      Source 16-bit depth image (in mm)
     * @param[out] sum      Destination 32-bit signed integer integral image
     * @param[in]  minDepth Ignore pixels with depth lower then this threshold
     * @param[in]  maxDepth Ignore pixels with depth higher then this threshold
     * @param[in]  minMag   Ignore pixels with edge magnitude lower than this
     */
    void filterDepthEdgels(const cv::Mat &src, cv::Mat &sum, int minDepth, int maxDepth, int minMag = 650);

    // Filters
    void filterSobel(const cv::Mat &src, cv::Mat &dst, bool xFilter = true, bool yFilter = true);

    // Computation
    void quantizedOrientationGradients(const cv::Mat &srcGray, cv::Mat &quantizedOrientations, cv::Mat &magnitude);

    // Quantization & discretization functions
    uchar quantizeOrientationGradient(float deg);
    uchar quantizeDepth(float depth, std::vector<cv::Range> &ranges);
}

#endif