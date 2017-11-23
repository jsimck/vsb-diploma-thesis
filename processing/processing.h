#ifndef VSB_SEMESTRAL_PROJECT_PROCESSING_H
#define VSB_SEMESTRAL_PROJECT_PROCESSING_H

#include <opencv2/core/mat.hpp>

// TODO move to namespaced functions rather than static

class Processing {
private:
    // Lookup tables
    static const int NORMAL_LUT_SIZE = 20, DEPTH_LUT_SIZE = 5;
    static const uchar DEPTH_LUT[DEPTH_LUT_SIZE];
    static const uchar NORMAL_LUT[NORMAL_LUT_SIZE][NORMAL_LUT_SIZE];

    /**
     * Applies bilateral filtering around each point A computing optimal gradient in b.
     *
     * Each point A represents a filtered depth to compute normal from, where
     * A[0] is origin. b represents optimal gradient computed across origin b[0]
     * and
     *
     * @param[in]  delta         Current pixel depth value
     * @param[in]  xShift        Patch shift in X direction (+/- patch), if shifted
     * @param[in]  yShift        Patch shift in Y direction (+/- patch), if shifted
     * @param[out] A             3 Points to compute bilateral filter for
     * @param[out] b             2 Values, containing optimal[1] and depth gradient[0]
     * @param[in]  maxDifference Ignore contributions of pixels whose depth difference with central
     *                            pixel is above this threshold
     */
    static void accumulateBilateral(long delta, long xShift, long yShift, long *A, long *b, int maxDifference);

public:
    /**
     * @brief Computes quantized surface normals from 16-bit depth image.
     *
     * @param[in]  src           Source 16-bit depth image (in mm)
     * @param[out] dst           Destination 8-bit image, where each bit represents one bin of view cone
     * @param[in]  fx            Camera focal length in X direction
     * @param[in]  fy            Camera focal length in Y direction
     * @param[in]  maxDistance   Ignore pixels beyond this distance
     * @param[in]  maxDifference When computing surface normals, ignore contributions of
     *                           pixels whose depth difference with central pixel is above this threshold
     */
    static void quantizedNormals(const cv::Mat &src, cv::Mat &dst, float fx, float fy, int maxDistance, int maxDifference);

    /**
     * @brief Computes relative depths from 16-bit depth image on (triplet) given points.
     *
     * @param[in]  src    Source 16-bit depth image (in mm)
     * @param[in]  c      Center triplet point
     * @param[in]  p1     P1 triplet point
     * @param[in]  p2     P2 triplet point
     * @param[out] depths 2 value output array where: depths[0] = src.at(p1) - src.at(c); depths[1] = src.at(p2) - src.at(c);
     */
    static void relativeDepths(const cv::Mat &src, cv::Point c, cv::Point p1, cv::Point p2, int *depths);

    // Filters
    static void filterSobel(const cv::Mat &src, cv::Mat &dst, bool xFilter = true, bool yFilter = true);
    static void thresholdMinMax(const cv::Mat &src, cv::Mat &dst, float min, float max);

    // Computation
    static void quantizedOrientationGradients(const cv::Mat &srcGray, cv::Mat &quantizedOrientations, cv::Mat &magnitude);

    // Quantization & discretization functions
    static uchar quantizeOrientationGradient(float deg);
    static uchar quantizeDepth(float depth, std::vector<cv::Range> &ranges);
};

#endif //VSB_SEMESTRAL_PROJECT_PROCESSING_H
