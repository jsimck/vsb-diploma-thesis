#ifndef VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
#define VSB_SEMESTRAL_PROJECT_OBJECTNESS_H

#include <string>
#include "../core/template.h"
#include "../core/template_group.h"

namespace objectness {
    void filterSobel(cv::Mat &src, cv::Mat &dst);
    void thresholdMinMax(cv::Mat &src, cv::Mat &dst, float min, float max);

    // Returns number of edgels of a template in template groups containing least amount of them and its dimensions [numOfEdgels, W, H]
    cv::Vec3i extractMinEdgels(std::vector<TemplateGroup> &templateGroups, float minThresh = 0.01f, float maxThresh = 0.1f);
    // Returns ROI of sliding windows that contain at least 30% of edgels as template containing least amount of them (see extractMinEdgels)
    cv::Rect objectness(cv::Mat &scene, cv::Mat &sceneDepthNormalized, cv::Mat &sceneColor, cv::Vec3i minEdgels, float minThresh = 0.01f, float maxThresh = 0.1f);
}

#endif //VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
