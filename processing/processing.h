#ifndef VSB_SEMESTRAL_PROJECT_PROCESSING_H
#define VSB_SEMESTRAL_PROJECT_PROCESSING_H

#include <opencv2/core/mat.hpp>

class Processing {
public:
    // Filters
    static void filterSobel(cv::Mat &src, cv::Mat &dst, bool xFilter = true, bool yFilter = true);
    static void thresholdMinMax(cv::Mat &src, cv::Mat &dst, float min, float max);

    // Computation
    static void quantizedSurfaceNormals(cv::Mat &srcDepth, cv::Mat &quantizedSurfaceNormals);
    static void quantizedOrientationGradients(cv::Mat &srcGray, cv::Mat &quantizedOrientations, cv::Mat &magnitude);

    // Quantization & discretization functions
    static uchar quantizeOrientationGradient(float deg);
    static uchar quantizeSurfaceNormal(const cv::Vec3f &normal);
    static uchar quantizeDepth(float depth, std::vector<cv::Range> &ranges);
};

#endif //VSB_SEMESTRAL_PROJECT_PROCESSING_H
