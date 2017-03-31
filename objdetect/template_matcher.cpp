#include "template_matcher.h"

uint TemplateMatcher::getFeaturePointsCount() const {
    return featurePointsCount;
}

void TemplateMatcher::setFeaturePointsCount(uint featurePointsCount) {
    assert(featurePointsCount > 0);
    this->featurePointsCount = featurePointsCount;
}

void TemplateMatcher::match(const cv::Mat &srcColor, const cv::Mat &srcGrayscale, const cv::Mat &srcDepth,
                     std::vector<Window> &windows, std::vector<TemplateMatch> &matches) {

}
