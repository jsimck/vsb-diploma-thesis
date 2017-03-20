#ifndef VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
#define VSB_SEMESTRAL_PROJECT_OBJECTNESS_H

#include <string>
#include "../core/template.h"
#include "../core/template_group.h"

#ifndef DEBUG
//#define DEBUG
#endif

void filterSobel(cv::Mat &src, cv::Mat &dst);
void thresholdMinMax(cv::Mat &src, cv::Mat &dst, double min, double max);

// Returns number of edgels of a template in template groups containing least amount of them and its dimensions [numOfEdgels, W, H]
cv::Vec3i extractMinEdgels(std::vector<TemplateGroup> &templateGroups, double minThresh = 0.01, double maxThresh = 0.1);
// Returns ROI of sliding windows that contain at least 30% of edgels as template containing least amount of them (see extractMinEdgels)
cv::Rect objectness(cv::Mat &scene, cv::Mat &sceneDepth, cv::Mat &sceneColor, std::vector<Template> &templates,
                    cv::Vec3i minEdgels, double minThresh = 0.01, double maxThresh = 0.1);

#endif //VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
