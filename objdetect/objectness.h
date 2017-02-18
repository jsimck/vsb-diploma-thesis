#ifndef VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
#define VSB_SEMESTRAL_PROJECT_OBJECTNESS_H

#include <string>
#include "../core/template.h"

#ifndef DEBUG
//#define DEBUG
#endif

void filterSobel(cv::Mat &src, cv::Mat &dst);
void thresholdMinMax(cv::Mat &src, cv::Mat &dst, double min, double max);

cv::Rect edgeBasedObjectness(cv::Mat &scene, cv::Mat &sceneDepth, cv::Mat &sceneColor, std::vector<Template> &templates, double thresh = 0.01);

#endif //VSB_SEMESTRAL_PROJECT_OBJECTNESS_H
