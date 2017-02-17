#ifndef VSB_SEMESTRAL_PROJECT_MATCHING_H
#define VSB_SEMESTRAL_PROJECT_MATCHING_H

#include <opencv2/opencv.hpp>
#include "../core/template.h"


void matchTemplate(cv::Mat &input, std::vector<Template> &templates, std::vector<cv::Rect> &matchBB);


#endif //VSB_SEMESTRAL_PROJECT_MATCHING_H