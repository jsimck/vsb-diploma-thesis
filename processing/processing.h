#ifndef VSB_SEMESTRAL_PROJECT_PROCESSING_H
#define VSB_SEMESTRAL_PROJECT_PROCESSING_H

#include <opencv2/core/mat.hpp>

class Processing {
public:
    // Filters
    static void filterSobel(const cv::Mat &src, cv::Mat &dst, bool xFilter = true, bool yFilter = true);
    static void thresholdMinMax(const cv::Mat &src, cv::Mat &dst, float min, float max);

    // Computation
    static void quantizedSurfaceNormals(const cv::Mat &srcDepth, cv::Mat &quantizedSurfaceNormals);
    static void quantizedOrientationGradients(const cv::Mat &srcGray, cv::Mat &quantizedOrientations, cv::Mat &magnitude);

    // Quantization & discretization functions
    static uchar quantizeOrientationGradient(float deg);
    static uchar quantizeSurfaceNormal(const cv::Vec3f &normal);
    static uchar quantizeDepth(float depth, std::vector<cv::Range> &ranges);
    static cv::Vec2i relativeDepths(const cv::Mat &src, cv::Point &c, cv::Point &p1, cv::Point &p2);
};

#endif //VSB_SEMESTRAL_PROJECT_PROCESSING_H
