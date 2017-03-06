#ifndef VSB_SEMESTRAL_PROJECT_MATCHING_H
#define VSB_SEMESTRAL_PROJECT_MATCHING_H

#include <opencv2/opencv.hpp>
#include "../core/template.h"

std::vector<cv::Rect> nonMaximaSuppression(std::vector<cv::Rect> &matchBB, std::vector<double> &scoreBB, double overlapThresh = 0.5);
std::vector<cv::Rect> matchTemplate(cv::Mat &input, cv::Rect inputRoi, std::vector<Template> &templates);

#endif //VSB_SEMESTRAL_PROJECT_MATCHING_H