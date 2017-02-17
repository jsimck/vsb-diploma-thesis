#ifndef VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
#define VSB_SEMESTRAL_PROJECT_OBJECTNESS_H


#include <string>
#include "../core/template.h"

void filterSobel(cv::Mat &src, cv::Mat &dst);
void thresholdMinMax(cv::Mat &src, cv::Mat &dst, double min, double max);

cv::Vec4i edgeBasedObjectness(cv::Mat &scene, cv::Mat &sceneDepth, cv::Mat &sceneColor, std::vector<Template> &templates,
                              double thresh);

#endif //VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
