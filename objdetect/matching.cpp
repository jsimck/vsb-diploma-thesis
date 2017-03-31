#include "matching.h"

uint Matching::getFeaturePointsCount() const {
    return featurePointsCount;
}

void Matching::setFeaturePointsCount(uint featurePointsCount) {
    assert(featurePointsCount > 0);
    this->featurePointsCount = featurePointsCount;
}

void Matching::match(const cv::Mat &srcColor, const cv::Mat &srcGrayscale, const cv::Mat &srcDepth,
                     const std::vector<Window> &windows) {

}
