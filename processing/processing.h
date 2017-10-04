#ifndef VSB_SEMESTRAL_PROJECT_PROCESSING_H
#define VSB_SEMESTRAL_PROJECT_PROCESSING_H

#include <opencv2/core/mat.hpp>

class Processing {
public:
    static void filterSobel(const cv::Mat &src, cv::Mat &dst, bool xFilter = true, bool yFilter = true);
    static void thresholdMinMax(const cv::Mat &src, cv::Mat &dst, float min, float max);
};

#endif //VSB_SEMESTRAL_PROJECT_PROCESSING_H
